import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 0. CONFIGURATION & HYPERPARAMETERS
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on: {DEVICE}")

# Training Params
BATCH_SIZE = 128
EPOCHS = 300
LATENT_DIM = 32     # Generator Noise Dimension
NUM_BINS = 8        # Bins per feature (K)

# ⚠️ NEW: Warmup parameters
WARMUP_EPOCHS = 30  # Number of warmup epochs for student
WARMUP_SAMPLES = 500  # Number of random samples per warmup epoch

# ⚠️ NEW: Replay buffer parameters
REPLAY_BUFFER_SIZE = 1000  # Maximum samples to store
REPLAY_RATIO = 0.10  # 25% of batch will be replay samples

# Weights
LAMBDA_COV = 4.0    # Weight for Interaction Diversity (Entropy)
LAMBDA_HARD = 1.5   # Weight for Adversarial Hardness

# ==========================================
# 1. DATA PREPARATION (RAW RANGES)
# ==========================================
print("\n📊 Loading Data (Raw Values)...")

# ====== CHANGE DATASET HERE ======
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
# Alternatives:
# from sklearn.datasets import load_iris
# data = load_iris()
#from sklearn.datasets import load_wine
#data = load_wine()
# from sklearn.datasets import load_digits
# data = load_digits()
# =================================

X_raw = data.data
y_raw = data.target

# Auto-detect dimensions
DATA_DIM = X_raw.shape[1]
NUM_CLASSES = len(np.unique(y_raw))

print(f"📋 Dataset Info:")
print(f"   Samples: {X_raw.shape[0]}")
print(f"   Features (DATA_DIM): {DATA_DIM}")
print(f"   Classes (NUM_CLASSES): {NUM_CLASSES}")

# Statistics for Initialization
X_min = torch.tensor(X_raw.min(axis=0), dtype=torch.float32).to(DEVICE)
X_max = torch.tensor(X_raw.max(axis=0), dtype=torch.float32).to(DEVICE)
X_mean = torch.tensor(X_raw.mean(axis=0), dtype=torch.float32).to(DEVICE)
X_std = torch.tensor(X_raw.std(axis=0), dtype=torch.float32).to(DEVICE)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# Convert to Tensors
X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
y_test_tensor = torch.LongTensor(y_test).to(DEVICE)

# ==========================================
# 3.5 HARD VARIANCE CALCULATION UTILITY
# ==========================================

def calculate_hard_variance(membership, teacher_probs):
    """
    Calculates HARD-ASSIGNMENT variance for interpretability.
    
    Returns:
        intra_variance: Average variance within bins (lower = purer bins)
        inter_variance: Variance between bin centroids (higher = better separation)
        variance_map: (F, K) tensor showing per-bin intra-bin variances
        inter_variance_per_feature: (F,) tensor showing per-feature inter-bin variance
    
    REASONING: Hard assignments show true bin purity when samples 
    are definitively assigned. Useful for rule extraction and debugging.
    Per-feature inter-bin variance shows which features have well-separated bins.
    """
    N, F, K = membership.shape
    epsilon = 1e-8
    
    with torch.no_grad():
        # Hard Assignment: argmax membership per feature
        hard_assignments = membership.argmax(dim=2)  # (N, F)
        
        intra_variance_total = 0.0
        inter_variance_total = 0.0
        valid_bins = 0
        
        variance_map = []
        inter_variance_per_feature = []  # NEW: Track per-feature inter-bin variance
        
        for f in range(F):
            bin_variances = []
            bin_centroids = []
            
            for k in range(K):
                # Binary mask for samples assigned to this bin
                bin_mask = (hard_assignments[:, f] == k)
                n_in_bin = bin_mask.sum().item()
                
                if n_in_bin > 1:
                    # Extract teacher predictions for this bin
                    bin_probs = teacher_probs[bin_mask]  # (n_in_bin, C)
                    
                    # Compute centroid
                    bin_centroid = bin_probs.mean(dim=0)  # (C,)
                    bin_centroids.append(bin_centroid)
                    
                    # Compute variance: mean squared distance from centroid
                    sq_dists = ((bin_probs - bin_centroid) ** 2).sum(dim=1)
                    bin_var = sq_dists.mean().item()
                    bin_variances.append(bin_var)
                    
                    intra_variance_total += bin_var
                    valid_bins += 1
                    
                elif n_in_bin == 1:
                    # Single sample: 0 variance, but track centroid
                    bin_centroid = teacher_probs[bin_mask][0]
                    bin_centroids.append(bin_centroid)
                    bin_variances.append(0.0)
                else:
                    # Empty bin
                    bin_variances.append(0.0)
            
            variance_map.append(bin_variances)
            
            # Inter-bin variance: variance of centroids for this feature
            if len(bin_centroids) > 1:
                centroids_tensor = torch.stack(bin_centroids)  # (n_bins_used, C)
                mean_centroid = centroids_tensor.mean(dim=0)
                centroid_var = ((centroids_tensor - mean_centroid) ** 2).sum() / len(bin_centroids)
                inter_variance_per_feature.append(centroid_var.item())  # NEW: Store per-feature
                inter_variance_total += centroid_var.item()
            else:
                inter_variance_per_feature.append(0.0)  # NEW: No variance for single/no bins
        
        # Average variances
        intra_variance = intra_variance_total / (valid_bins + epsilon)
        inter_variance = inter_variance_total / F
        variance_map = torch.tensor(variance_map)  # (F, K)
        inter_variance_per_feature = torch.tensor(inter_variance_per_feature)  # (F,)
        
        return intra_variance, inter_variance, variance_map, inter_variance_per_feature

# ==========================================
# 1.5 RANDOM DATA GENERATION MODULE
# ==========================================

class RandomDataGenerator:
    """
    Generates random samples within valid feature ranges for warmup.
    Uses uniform distribution between min and max for each feature.
    """
    def __init__(self, min_vals, max_vals, device='cpu'):
        """
        Args:
            min_vals: Tensor of shape (num_features,) with minimum values
            max_vals: Tensor of shape (num_features,) with maximum values
            device: Device to generate samples on
        """
        self.min_vals = min_vals.to(device)
        self.max_vals = max_vals.to(device)
        self.device = device
        self.num_features = len(min_vals)
    
    def generate(self, num_samples):
        """
        Generate random samples uniformly within [min, max] for each feature.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tensor of shape (num_samples, num_features)
        """
        # Generate uniform random values in [0, 1]
        random_samples = torch.rand(num_samples, self.num_features, device=self.device)
        
        # Scale to [min, max] range
        range_vals = self.max_vals - self.min_vals
        scaled_samples = self.min_vals + random_samples * range_vals
        
        return scaled_samples

# Initialize random data generator
random_gen = RandomDataGenerator(X_min, X_max, device=DEVICE)

print(f"✅ Random Data Generator initialized")
print(f"   Feature ranges: [{X_min.min().item():.2f}, {X_max.max().item():.2f}]")

# Test generation
test_samples = random_gen.generate(5)
print(f"   Sample generated data shape: {test_samples.shape}")
print(f"   Sample values (first 3 features):\n{test_samples[:, :min(3, DATA_DIM)]}")

# ==========================================
# 1.6 REPLAY BUFFER
# ==========================================

class ReplayBuffer:
    """
    Experience replay buffer to store warmup samples.
    Prevents catastrophic forgetting during main training.
    """
    def __init__(self, max_size, device='cpu'):
        self.max_size = max_size
        self.device = device
        self.buffer = []
        self.position = 0
    
    def add(self, samples):
        """
        Add samples to the buffer. Overwrites old samples when full.
        
        Args:
            samples: Tensor of shape (batch_size, num_features)
        """
        samples = samples.detach()
        
        if len(self.buffer) < self.max_size:
            # Buffer not full yet, append
            for i in range(samples.shape[0]):
                if len(self.buffer) >= self.max_size:
                    break
                self.buffer.append(samples[i:i+1])
        else:
            # Buffer full, replace old samples (circular buffer)
            for i in range(samples.shape[0]):
                self.buffer[self.position] = samples[i:i+1]
                self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size):
        """
        Sample random batch from buffer.
        
        Args:
            batch_size: Number of samples to retrieve
            
        Returns:
            Tensor of shape (batch_size, num_features)
        """
        if len(self.buffer) == 0:
            return None
        
        batch_size = min(batch_size, len(self.buffer))
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        samples = torch.cat([self.buffer[i] for i in indices], dim=0)
        return samples.to(self.device)
    
    def __len__(self):
        return len(self.buffer)

# Initialize replay buffer
replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, device=DEVICE)
print(f"✅ Replay Buffer initialized (max size: {REPLAY_BUFFER_SIZE})")

# ==========================================
# 2. MODEL ARCHITECTURES
# ==========================================

class TeacherNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class StudentNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), # Lightweight
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, mean_vals):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim) 
            # Linear Output (Raw Values)
        )
        # Hack: Initialize bias to mean so generator starts in valid range
        with torch.no_grad():
            self.net[-1].bias.copy_(mean_vals.to(self.net[-1].weight.device))

    def forward(self, z):
        return self.net(z)

class BinLearner(nn.Module):
    def __init__(self, num_features, num_bins, min_vals, max_vals):
        super().__init__()
        self.num_bins = num_bins
        self.temperature = 1.0 # Annealed later
        
        self.register_buffer('min_val', min_vals.view(1, -1, 1))
        self.register_buffer('max_val', max_vals.view(1, -1, 1))
        
        # --- STABLE INITIALIZATION ---
        range_size = (max_vals - min_vals).view(1, -1, 1)  # (1, num_features, 1)
        target_width = range_size / num_bins                 # (1, num_features, 1) - single width value
        
        # Replicate to create num_bins widths per feature
        target_width = target_width.expand(-1, -1, num_bins)  # (1, num_features, num_bins)
        
        # Handle large raw values preventing exp() overflow
        threshold = 20.0
        is_large = target_width > threshold
        safe_param = torch.zeros_like(target_width)
        safe_param[is_large] = target_width[is_large]
        # Inverse softplus for small values
        small_val = target_width[~is_large]
        safe_param[~is_large] = torch.log(torch.exp(small_val) - 1)
        
        self.raw_widths = nn.Parameter(safe_param)

    def get_boundaries(self):
        widths = F.softplus(self.raw_widths)
        total_learned = widths.sum(dim=2, keepdim=True) + 1e-6
        range_size = self.max_val - self.min_val
        norm_widths = (widths / total_learned) * range_size
        
        cum_widths = torch.cumsum(norm_widths, dim=2)
        internal_bounds = cum_widths + self.min_val
        start_bounds = self.min_val.expand(-1, -1, 1)
        
        return torch.cat([start_bounds, internal_bounds], dim=2)

    def forward(self, x):
        x_in = x.unsqueeze(2)
        bounds = self.get_boundaries()
        sigmoids = torch.sigmoid(self.temperature * (x_in - bounds))
        membership = sigmoids[:, :, :-1] - sigmoids[:, :, 1:]
        membership = membership + 1e-8
        
        # NEW: Normalize to sum to 1.0 per feature
        membership = membership / (membership.sum(dim=2, keepdim=True) + 1e-8)
        
        return membership

# ==========================================
# 3. NEW ENTROPIC LOSS FUNCTIONS
# ==========================================

class EntropicBinLoss(nn.Module):
    """
    1. Intra-Bin: Minimize Entropy (Make bins Pure)
    2. Inter-Bin: Maximize Entropy of Mixture (Make neighbors Different)
    """
    def forward(self, membership, teacher_probs):
        epsilon = 1e-8
        N, F, K = membership.shape
        
        # Bin Mass & Centroids
        bin_mass = membership.sum(dim=0) + epsilon
        weighted_y = membership.unsqueeze(3) * teacher_probs.unsqueeze(1).unsqueeze(1)
        centroids = weighted_y.sum(dim=0) / bin_mass.unsqueeze(2) # (F, K, C)
        
        # 1. Intra-Bin Purity (Minimize Entropy)
        # H(p) = -sum(p log p). We minimize this.
        # This forces centroids to be [1, 0] or [0, 1], not [0.5, 0.5]
        intra_entropy = -torch.sum(centroids * torch.log(centroids + epsilon), dim=2)
        loss_intra = (intra_entropy * bin_mass).sum() / N
        
        # 2. Inter-Bin Repulsion (Maximize JS Divergence / Mixture Entropy)
        curr_bins = centroids[:, :-1, :]
        next_bins = centroids[:, 1:, :]
        
        # Mixture = Average of neighbors
        mixture = 0.5 * (curr_bins + next_bins)
        
        # We want Mixture to be chaotic (0.5, 0.5).
        # H(Mix) = -sum(mix * log(mix)). We want to MAXIMIZE this.
        # Equivalent to MINIMIZING: sum(mix * log(mix))
        loss_inter = torch.sum(mixture * torch.log(mixture + epsilon))
        
        return loss_intra + (0.5 * loss_inter)

class InteractionDiversityLoss(nn.Module):
    """
    Maximizes Entropy of the Pairwise Joint Distribution.
    Goal: Uniform distribution across all KxK tiles.
    """
    def forward(self, membership):
        N, F, K = membership.shape
        epsilon = 1e-8
        
        # Joint Co-occurrence: (N, i, a), (N, j, b) -> (i, j, a, b)
        joint_counts = torch.einsum('nia, njb -> ijab', membership, membership)
        joint_probs = joint_counts / N
        
        # Mask diagonal
        mask = torch.triu(torch.ones(F, F), diagonal=1).bool().to(membership.device)
        valid_probs = joint_probs[mask] # (Pairs, K, K)
        
        # Maximize Entropy = Minimize Sum(p log p)
        loss_entropy = torch.sum(valid_probs * torch.log(valid_probs + epsilon))
        
        return loss_entropy

class VarianceBasedBinLoss(nn.Module):
    """
    Variance-based bin loss using MSE:
    1. Intra-Bin: Minimize variance (make bins tight around their centroid)
    2. Inter-Bin: Maximize distance between adjacent bin centroids
    
    REASONING: Variance-based approach:
    - Intra-bin variance: Low variance = samples in same bin are similar
    - Inter-bin distance: Large distance = bins represent different regions
    - MSE naturally penalizes outliers more than entropy
    - More intuitive geometric interpretation
    """
    def forward(self, membership, teacher_probs):
        epsilon = 1e-8
        N, F, K = membership.shape
        
        # Bin Mass & Centroids
        bin_mass = membership.sum(dim=0) + epsilon  # (F, K)
        weighted_y = membership.unsqueeze(3) * teacher_probs.unsqueeze(1).unsqueeze(1)
        centroids = weighted_y.sum(dim=0) / bin_mass.unsqueeze(2)  # (F, K, C)
        
        # 1. Intra-Bin Loss: Minimize variance (MSE from centroid)
        centroids_expanded = centroids.unsqueeze(0).expand(N, -1, -1, -1)
        teacher_probs_expanded = teacher_probs.unsqueeze(1).unsqueeze(1).expand(-1, F, K, -1)
        
        squared_distances = ((teacher_probs_expanded - centroids_expanded) ** 2).sum(dim=3)  # (N, F, K)
        loss_intra = (membership * squared_distances).sum() / N
        
        # 2. Inter-Bin Loss: Maximize distance between adjacent centroids
        curr_bins = centroids[:, :-1, :]  # (F, K-1, C)
        next_bins = centroids[:, 1:, :]   # (F, K-1, C)
        
        inter_distances = ((curr_bins - next_bins) ** 2).sum(dim=2)  # (F, K-1)
        loss_inter = -inter_distances.sum() / F
        
        # 3. Compute HARD-ASSIGNMENT variances for monitoring
        intra_var, inter_var, _, _ = calculate_hard_variance(membership, teacher_probs)
        
        # Return total loss AND individual components for tracking
        return 2.0*loss_intra + (4.0 * loss_inter), loss_intra, loss_inter, intra_var, inter_var

# 4. TRAINING PIPELINE
# ==========================================

# A. Pre-train Teacher
print("\n🎓 Pre-training Teacher...")
teacher = TeacherNet(DATA_DIM, NUM_CLASSES).to(DEVICE)
t_opt = optim.Adam(teacher.parameters(), lr=0.001)
t_crit = nn.CrossEntropyLoss()

for e in range(50):
    t_opt.zero_grad()
    logits = teacher(X_train_tensor)
    loss = t_crit(logits, y_train_tensor)
    loss.backward()
    t_opt.step()

# Evaluate Teacher
teacher.eval()
with torch.no_grad():
    # Train accuracy
    teacher_train_logits = teacher(X_train_tensor)
    teacher_train_preds = torch.argmax(teacher_train_logits, dim=1)
    teacher_train_acc = accuracy_score(y_train, teacher_train_preds.cpu().numpy())
    
    # Test accuracy
    teacher_test_logits = teacher(X_test_tensor)
    teacher_test_preds = torch.argmax(teacher_test_logits, dim=1)
    teacher_test_acc = accuracy_score(y_test, teacher_test_preds.cpu().numpy())

print(f"✅ Teacher Frozen.")
print(f"   Train Accuracy: {teacher_train_acc*100:.2f}%")
print(f"   Test Accuracy:  {teacher_test_acc*100:.2f}%")

# B. Initialize Modules
bin_learner = BinLearner(DATA_DIM, NUM_BINS, X_min, X_max).to(DEVICE)
generator = Generator(LATENT_DIM, DATA_DIM, X_mean).to(DEVICE)
student = StudentNet(DATA_DIM, NUM_CLASSES).to(DEVICE)

# ==========================================
# 4.5 STUDENT WARMUP WITH RANDOM DATA + REPLAY BUFFER FILLING
# ==========================================
print("\n🔥 Warming up Student with Random Data...")
print(f"   Generating {WARMUP_SAMPLES} random samples per epoch for {WARMUP_EPOCHS} epochs")
print(f"   Storing samples in replay buffer (max: {REPLAY_BUFFER_SIZE})")

warmup_opt = optim.Adam(student.parameters(), lr=0.001)
loss_kl_fn = nn.KLDivLoss(reduction='batchmean')

for epoch in range(WARMUP_EPOCHS):
    # Generate random samples
    x_random = random_gen.generate(WARMUP_SAMPLES)
    
    # ⚠️ NEW: Add samples to replay buffer
    replay_buffer.add(x_random)
    
    # Get teacher predictions (soft labels)
    teacher.eval()
    with torch.no_grad():
        teacher_probs = F.softmax(teacher(x_random), dim=1)
    
    # Train student to mimic teacher on random data
    student.train()
    warmup_opt.zero_grad()
    
    student_logits = student(x_random)
    student_log_probs = F.log_softmax(student_logits, dim=1)
    
    # KL divergence loss
    loss = loss_kl_fn(student_log_probs, teacher_probs)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    warmup_opt.step()
    
    # Log progress
    if (epoch + 1) % 10 == 0:
        student.eval()
        with torch.no_grad():
            test_preds = torch.argmax(student(X_test_tensor), dim=1)
            test_acc = accuracy_score(y_test, test_preds.cpu().numpy())
        student.train()
        print(f"   Warmup Epoch {epoch+1:3d}/{WARMUP_EPOCHS}: Loss={loss.item():.4f}, Test Acc={test_acc*100:.2f}%, Buffer={len(replay_buffer)}")

# Final warmup evaluation
student.eval()
with torch.no_grad():
    warmup_test_preds = torch.argmax(student(X_test_tensor), dim=1)
    warmup_test_acc = accuracy_score(y_test, warmup_test_preds.cpu().numpy())
    
    warmup_train_preds = torch.argmax(student(X_train_tensor), dim=1)
    warmup_train_acc = accuracy_score(y_train, warmup_train_preds.cpu().numpy())

print(f"\n✅ Student Warmup Complete!")
print(f"   Train Accuracy: {warmup_train_acc*100:.2f}%")
print(f"   Test Accuracy:  {warmup_test_acc*100:.2f}%")
print(f"   Replay Buffer: {len(replay_buffer)} samples stored")

# C. Losses & Optimizers for Main Training
loss_bin_fn = VarianceBasedBinLoss()  # CHANGED: Now using variance-based MSE loss
loss_div_fn = InteractionDiversityLoss()

opt_bin = optim.Adam(bin_learner.parameters(), lr=0.01)
opt_gen = optim.Adam(generator.parameters(), lr=0.001)
opt_stu = optim.Adam(student.parameters(), lr=0.001)

history = {
    'bin': [], 'div': [], 'hard': [], 'stu': [], 'test_acc': [], 'agreement': [],
    'bin_intra': [],
    'bin_inter': [],
    'intra_variance': [],
    'inter_variance': [],
    'variance_maps': {},
    'inter_variance_per_feature': {},  # NEW: Store per-feature inter-bin variance
    'boundaries': {},
    'memberships': {}
}

# Store initial boundaries
with torch.no_grad():
    initial_boundaries = bin_learner.get_boundaries().detach().cpu().numpy()
    history['boundaries'][0] = initial_boundaries

print("\n⚔️ Starting Main Training...")
print(f"   Using replay buffer: {REPLAY_RATIO*100:.0f}% of batch from warmup samples")
print(f"{'Epoch':<6} | {'Bin':<8} | {'Div':<8} | {'Hard':<8} | {'Stu':<8} | {'Test Acc':<10} | {'Agreement':<10} | {'Intra Var':<10} | {'Inter Var':<10}")
print("-" * 135)

best_student_state = None
best_agreement_score = 0.0

for epoch in range(1, EPOCHS + 1):
    
    # Anneal Temperature: 0.5 -> 5.0
    bin_learner.temperature = 0.5 + (4.5 * (epoch / EPOCHS))
    z = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
    
    # ==========================
    # PHASE 1: BIN LEARNER (Purity + Repulsion)
    # ==========================
    opt_bin.zero_grad()
    x_gen = generator(z).detach()

    with torch.no_grad():
        t_probs = F.softmax(teacher(x_gen), dim=1)
        
    mship = bin_learner(x_gen)
    l_bin, l_intra, l_inter, intra_var, inter_var = loss_bin_fn(mship, t_probs)  # CHANGED: Unpack 5 values


    if not torch.isnan(l_bin):
        l_bin.backward()
        # Clip Gradients for Stability
        torch.nn.utils.clip_grad_norm_(bin_learner.parameters(), 1.0)
        opt_bin.step()
    
    # ==========================
    # PHASE 2: GENERATOR (Diversity + Hardness)
    # ==========================
    opt_gen.zero_grad()
    x_gen_2 = generator(z)
    
    # 1. Interaction Diversity (Maximize Entropy)
    mship_2 = bin_learner(x_gen_2)
    l_div = loss_div_fn(mship_2) 
    
    # 2. Hardness (Maximize Student Error)
    with torch.no_grad():
        t_probs_2 = F.softmax(teacher(x_gen_2), dim=1)
    s_log_probs = F.log_softmax(student(x_gen_2), dim=1)
    
    # Minimize Negative KL
    l_hard = -1.0 * loss_kl_fn(s_log_probs, t_probs_2)
    
    l_gen = (LAMBDA_COV * l_div) + (LAMBDA_HARD * l_hard)
    
    if not torch.isnan(l_gen):
        l_gen.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_gen.step()
    
    # ==========================
    # PHASE 3: STUDENT (Synthetic + Replay Buffer)
    # ==========================
    opt_stu.zero_grad()
    x_gen_3 = x_gen_2.detach()
    
    # ⚠️ NEW: Mix synthetic samples with replay buffer samples
    replay_batch_size = int(BATCH_SIZE * REPLAY_RATIO)
    x_replay = replay_buffer.sample(replay_batch_size)
    
    if x_replay is not None:
        x_mixed = torch.cat([x_gen_3, x_replay], dim=0)
    else:
        x_mixed = x_gen_3
    
    with torch.no_grad():
        t_probs_3 = F.softmax(teacher(x_mixed), dim=1)
    s_log_probs_3 = F.log_softmax(student(x_mixed), dim=1)
    
    l_stu = loss_kl_fn(s_log_probs_3, t_probs_3)
    
    if not torch.isnan(l_stu):
        l_stu.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt_stu.step()
    
    # Logging
    history['bin'].append(l_bin.item())
    history['bin_intra'].append(l_intra.item())  # NEW
    history['bin_inter'].append(l_inter.item())  # NEW
    history['div'].append(l_div.item())
    history['hard'].append(l_hard.item())
    history['stu'].append(l_stu.item())
    
    # Store prediction-space variances (already computed in forward pass)
    history['intra_variance'].append(intra_var)
    history['inter_variance'].append(inter_var)
    
    if epoch % 20 == 0:
        student.eval()
        with torch.no_grad():
            s_test_preds = torch.argmax(student(X_test_tensor), dim=1)
            t_test_preds = torch.argmax(teacher(X_test_tensor), dim=1)
            test_acc = accuracy_score(y_test, s_test_preds.cpu().numpy())
            agreement = (t_test_preds == s_test_preds).float().mean().item()
        history['test_acc'].append(test_acc)
        history['agreement'].append(agreement)
        student.train()
        print(f"{epoch:<6} | {l_bin.item():<8.3f} | {l_div.item():<8.1f} | {l_hard.item():<8.3f} | {l_stu.item():<8.3f} | {test_acc*100:<10.1f}% | {agreement*100:<10.1f}% | {intra_var:<10.4f} | {inter_var:<10.4f}")
        
        # Track best student model
        if agreement > best_agreement_score:
            best_agreement_score = agreement
            best_student_state = {
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'test_acc': test_acc,
                'agreement': agreement
            }
    
    # Store boundaries and membership info every 50 iterations
    if epoch % 50 == 0:
        with torch.no_grad():
            # Store boundaries
            boundaries = bin_learner.get_boundaries().detach().cpu().numpy()
            history['boundaries'][epoch] = boundaries
            
            # Store membership statistics
            z_sample = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
            x_sample = generator(z_sample)
            membership_sample = bin_learner(x_sample)
            
            # Compute statistics: mean membership per bin
            mean_membership = membership_sample.mean(axis=0).detach().cpu().numpy()  # (Features, Bins)
            std_membership = membership_sample.std(axis=0).detach().cpu().numpy()    # (Features, Bins)
            
            history['memberships'][epoch] = {
                'mean': mean_membership,
                'std': std_membership,
                'temperature': bin_learner.temperature
            }
            
            # NEW: Compute and store hard variance maps
            with torch.no_grad():
                t_probs_sample = F.softmax(teacher(x_sample), dim=1)
                _, _, var_map, inter_var_per_feat = calculate_hard_variance(membership_sample, t_probs_sample)
            
            history['variance_maps'][epoch] = var_map.cpu().numpy()
            history['inter_variance_per_feature'][epoch] = inter_var_per_feat.cpu().numpy()  # NEW

# ==========================================
# 5. FINAL EVALUATION
# ==========================================
print("\n" + "="*90)
print("FINAL EVALUATION")
print("="*90)

student.eval()
with torch.no_grad():
    # Student predictions on test set
    student_test_logits = student(X_test_tensor)
    student_test_preds = torch.argmax(student_test_logits, dim=1)
    student_test_acc = accuracy_score(y_test, student_test_preds.cpu().numpy())
    
    # Teacher-Student Agreement on Test Set
    teacher_test_preds = torch.argmax(teacher(X_test_tensor), dim=1)
    test_agreement = (teacher_test_preds == student_test_preds).float().mean().item()
    
    # Student predictions on train set
    student_train_logits = student(X_train_tensor)
    student_train_preds = torch.argmax(student_train_logits, dim=1)
    student_train_acc = accuracy_score(y_train, student_train_preds.cpu().numpy())
    
    # Teacher-Student Agreement on Train Set
    teacher_train_preds = torch.argmax(teacher(X_train_tensor), dim=1)
    train_agreement = (teacher_train_preds == student_train_preds).float().mean().item()

print(f"\n📊 ACCURACY COMPARISON:")
print(f"{'Stage':<20} | {'Train Acc':<12} | {'Test Acc':<12}")
print("-" * 50)
print(f"{'Teacher':<20} | {teacher_train_acc*100:>10.2f}% | {teacher_test_acc*100:>10.2f}%")
print(f"{'Student (Warmup)':<20} | {warmup_train_acc*100:>10.2f}% | {warmup_test_acc*100:>10.2f}%")
print(f"{'Student (Final)':<20} | {student_train_acc*100:>10.2f}% | {student_test_acc*100:>10.2f}%")
print(f"{'Gap (Final)':<20} | {abs(teacher_train_acc - student_train_acc)*100:>10.2f}% | {abs(teacher_test_acc - student_test_acc)*100:>10.2f}%")

print(f"\n🤝 TEACHER-STUDENT AGREEMENT:")
print(f"   Train Set: {train_agreement*100:.2f}%")
print(f"   Test Set:  {test_agreement*100:.2f}%")

print(f"\n💾 REPLAY BUFFER STATS:")
print(f"   Total samples stored: {len(replay_buffer)}")
print(f"   Samples used per batch: {int(BATCH_SIZE * REPLAY_RATIO)}")

# ==========================================
# 5.5. COMPREHENSIVE TEXT REPORT
# ==========================================
print("\n📝 Generating comprehensive text report...")

# Define test_acc_epochs BEFORE using it in the report
test_acc_epochs = [i*20 for i in range(len(history["test_acc"]))]

report_filename = f'experiment_report_{DATA_DIM}features_{NUM_CLASSES}classes.txt'

with open(report_filename, 'w') as f:
    f.write("="*90 + "\n")
    f.write("COMPREHENSIVE EXPERIMENT REPORT\n")
    f.write("="*90 + "\n\n")
    
    # Header Information
    f.write("EXPERIMENT CONFIGURATION\n")
    f.write("-"*90 + "\n")
    f.write(f"Dataset: {data.__class__.__name__}\n")
    f.write(f"Total Samples: {X_raw.shape[0]}\n")
    f.write(f"Features: {DATA_DIM}\n")
    f.write(f"Classes: {NUM_CLASSES}\n")
    f.write(f"Training Samples: {len(y_train)}\n")
    f.write(f"Test Samples: {len(y_test)}\n")
    f.write(f"\nHyperparameters:\n")
    f.write(f"  Batch Size: {BATCH_SIZE}\n")
    f.write(f"  Epochs: {EPOCHS}\n")
    f.write(f"  Num Bins: {NUM_BINS}\n")
    f.write(f"  Warmup Epochs: {WARMUP_EPOCHS}\n")
    f.write(f"  Replay Buffer Size: {REPLAY_BUFFER_SIZE}\n")
    f.write(f"  Replay Ratio: {REPLAY_RATIO*100:.1f}%\n")
    f.write(f"  Lambda Coverage: {LAMBDA_COV}\n")
    f.write(f"  Lambda Hardness: {LAMBDA_HARD}\n")
    f.write("\n\n")
    
    # Model Summary
    f.write("="*90 + "\n")
    f.write("MODEL SUMMARY\n")
    f.write("="*90 + "\n\n")
    
    f.write("Teacher Network:\n")
    f.write(f"  Architecture: {DATA_DIM} -> 128 -> 64 -> {NUM_CLASSES}\n")
    f.write(f"  Train Accuracy: {teacher_train_acc*100:.2f}%\n")
    f.write(f"  Test Accuracy: {teacher_test_acc*100:.2f}%\n\n")
    
    f.write("Student Network:\n")
    f.write(f"  Architecture: {DATA_DIM} -> 32 -> {NUM_CLASSES}\n")
    f.write(f"  Warmup Train Accuracy: {warmup_train_acc*100:.2f}%\n")
    f.write(f"  Warmup Test Accuracy: {warmup_test_acc*100:.2f}%\n")
    f.write(f"  Final Train Accuracy: {student_train_acc*100:.2f}%\n")
    f.write(f"  Final Test Accuracy: {student_test_acc*100:.2f}%\n\n")
    
    f.write("Agreement:\n")
    f.write(f"  Train Set: {train_agreement*100:.2f}%\n")
    f.write(f"  Test Set:  {test_agreement*100:.2f}%\n\n")
    
    # Best Agreement
    best_agreement_idx = np.argmax(history['agreement'])
    best_agreement = history['agreement'][best_agreement_idx]
    best_agreement_epoch = test_acc_epochs[best_agreement_idx]
    f.write(f"Best Agreement:\n")
    f.write(f"  Epoch: {best_agreement_epoch}\n")
    f.write(f"  Agreement: {best_agreement*100:.2f}%\n")
    f.write(f"  Test Accuracy at Best Agreement: {history['test_acc'][best_agreement_idx]*100:.2f}%\n\n\n")
    
    # 1. BIN BOUNDARIES - First 5 and Last 5 Iterations
    f.write("="*90 + "\n")
    f.write("1. BIN BOUNDARIES FOR SELECTED FEATURES\n")
    f.write("="*90 + "\n\n")
    
    # Get first 5 and last 5 epochs with boundaries
    boundary_epochs = sorted(history['boundaries'].keys())
    first_5_epochs = boundary_epochs[:min(5, len(boundary_epochs))]
    last_5_epochs = boundary_epochs[-min(5, len(boundary_epochs)):]
    
    # Show for first 5 features
    num_features_to_show = DATA_DIM  # Show all features
    
    f.write(f"FIRST 5 ITERATIONS (Epochs: {first_5_epochs})\n")
    f.write("-"*90 + "\n\n")
    for feat_idx in range(num_features_to_show):
        f.write(f"Feature {feat_idx}:\n")
        for epoch in first_5_epochs:
            bounds = history['boundaries'][epoch][0, feat_idx]
            f.write(f"  Epoch {epoch:3d}: [{', '.join([f'{b:10.4f}' for b in bounds])}]\n")
        f.write("\n")
    
    f.write(f"\nLAST 5 ITERATIONS (Epochs: {last_5_epochs})\n")
    f.write("-"*90 + "\n\n")
    for feat_idx in range(num_features_to_show):
        f.write(f"Feature {feat_idx}:\n")
        for epoch in last_5_epochs:
            bounds = history['boundaries'][epoch][0, feat_idx]
            f.write(f"  Epoch {epoch:3d}: [{', '.join([f'{b:10.4f}' for b in bounds])}]\n")
        f.write("\n")
    
    # 2. LOSS TRENDS EVERY 15 ITERATIONS
    f.write("\n" + "="*90 + "\n")
    f.write("2. LOSS TRENDS (Every 15 Iterations)\n")
    f.write("="*90 + "\n\n")
    
    f.write(f"{'Epoch':<8} | {'Bin Loss':<12} | {'Div Loss':<12} | {'Hard Loss':<12} | {'Stu Loss':<12}\n")
    f.write("-"*90 + "\n")
    
    for epoch in range(15, EPOCHS+1, 15):
        loss_idx = epoch - 1
        if loss_idx < len(history['bin']):
            f.write(f"{epoch:<8} | {history['bin'][loss_idx]:<12.4f} | "
                   f"{history['div'][loss_idx]:<12.2f} | "
                   f"{history['hard'][loss_idx]:<12.4f} | "
                   f"{history['stu'][loss_idx]:<12.4f}\n")
    
    # 3. ACCURACY AND AGREEMENT EVERY 15 ITERATIONS
    f.write("\n" + "="*90 + "\n")
    f.write("3. ACCURACY AND AGREEMENT TRENDS (Every 15 Iterations)\n")
    f.write("="*90 + "\n\n")
    
    f.write(f"{'Epoch':<8} | {'Test Accuracy':<15} | {'Agreement':<15}\n")
    f.write("-"*90 + "\n")
    
    # Find closest test accuracy measurements to every 15 iterations
    for target_epoch in range(15, EPOCHS+1, 15):
        # Find closest epoch in test_acc_epochs
        if len(test_acc_epochs) > 0:
            closest_idx = min(range(len(test_acc_epochs)), 
                            key=lambda i: abs(test_acc_epochs[i] - target_epoch))
            actual_epoch = test_acc_epochs[closest_idx]
            if abs(actual_epoch - target_epoch) <= 15:  # Within reasonable range
                f.write(f"{actual_epoch:<8} | {history['test_acc'][closest_idx]*100:<14.2f}% | "
                       f"{history['agreement'][closest_idx]*100:<14.2f}%\n")
    
    # 4. MEAN MEMBERSHIP FOR FIRST 5 FEATURES EVERY 50 ITERATIONS
    f.write("\n" + "="*90 + "\n")
    f.write("4. MEAN MEMBERSHIP PER BIN (All Features, Every 50 Iterations)\n")
    f.write("="*90 + "\n\n")
    
    membership_epochs = sorted(history['memberships'].keys())
    
    for epoch in membership_epochs:
        membership_data = history['memberships'][epoch]
        mean_mem = membership_data['mean']
        temp = membership_data['temperature']
        
        f.write(f"\nEpoch {epoch} (Temperature: {temp:.4f}):\n")
        f.write("-"*90 + "\n")
        
        for feat_idx in range(num_features_to_show):
            f.write(f"  Feature {feat_idx}: ")
            for bin_idx in range(NUM_BINS):
                f.write(f"Bin{bin_idx}={mean_mem[feat_idx][bin_idx]:.4f}  ")
            f.write("\n")
    
    # 5. VARIANCE MAPS (Every 50 Iterations) - ORGANIZED BY FEATURE
    f.write("\n" + "="*90 + "\n")
    f.write("5. PER-BIN VARIANCE MAPS (Hard Assignments, Every 50 Iterations)\n")
    f.write("="*90 + "\n\n")
    f.write("Shows variance within each bin (lower = purer bins)\n")
    f.write("Organized by feature to show evolution over epochs\n\n")
    
    variance_map_epochs = sorted(history['variance_maps'].keys())
    
    # Organize by FEATURE instead of epoch
    for feat_idx in range(num_features_to_show):
        f.write(f"\n{'='*90}\n")
        f.write(f"FEATURE {feat_idx}\n")
        f.write(f"{'='*90}\n\n")
        
        for epoch in variance_map_epochs:
            var_map = history['variance_maps'][epoch]  # (F, K)
            
            f.write(f"Epoch {epoch:3d}: ")
            for bin_idx in range(NUM_BINS):
                f.write(f"Bin{bin_idx}={var_map[feat_idx][bin_idx]:.6f}  ")
            f.write(f"| Mean={var_map[feat_idx].mean():.6f}\n")
        
        # Per-feature statistics across all epochs
        f.write(f"\n  Evolution Statistics for Feature {feat_idx}:\n")
        
        first_epoch_map = history['variance_maps'][variance_map_epochs[0]]
        last_epoch_map = history['variance_maps'][variance_map_epochs[-1]]
        
        initial_mean = first_epoch_map[feat_idx].mean()
        final_mean = last_epoch_map[feat_idx].mean()
        change = final_mean - initial_mean
        
        f.write(f"    Initial mean (Epoch {variance_map_epochs[0]}): {initial_mean:.6f}\n")
        f.write(f"    Final mean (Epoch {variance_map_epochs[-1]}):   {final_mean:.6f}\n")
        f.write(f"    Change: {change:+.6f} ")
        if change < 0:
            f.write("(✓ Improved - bins got purer)\n")
        elif change > 0:
            f.write("(✗ Degraded - bins got mixed)\n")
        else:
            f.write("(→ No change)\n")
        
        # Find best and worst bins
        final_bins = last_epoch_map[feat_idx]
        best_bin = final_bins.argmin()
        worst_bin = final_bins.argmax()
        
        f.write(f"    Best bin (lowest variance): Bin {best_bin} ({final_bins[best_bin]:.6f})\n")
        f.write(f"    Worst bin (highest variance): Bin {worst_bin} ({final_bins[worst_bin]:.6f})\n")
        f.write("\n")
    
    # Overall statistics (keep at the end)
    f.write(f"\n{'='*90}\n")
    f.write("OVERALL VARIANCE MAP STATISTICS\n")
    f.write(f"{'='*90}\n\n")
    
    for epoch in variance_map_epochs:
        var_map = history['variance_maps'][epoch]
        f.write(f"\nEpoch {epoch}:\n")
        f.write(f"  Mean variance across all bins: {var_map.mean():.6f}\n")
        f.write(f"  Max bin variance: {var_map.max():.6f}\n")
        f.write(f"  Min bin variance: {var_map.min():.6f}\n")
        f.write(f"  Std deviation: {var_map.std():.6f}\n")

# ==========================================
# 6. VISUALIZATION
# ==========================================
fig, axs = plt.subplots(3, 3, figsize=(18, 14))

# Row 1: Loss components
axs[0, 0].plot(history['bin'], label='Total', alpha=0.7)
axs[0, 0].plot(history['bin_intra'], label='Intra-Bin', alpha=0.7)
axs[0, 0].plot(history['bin_inter'], label='Inter-Bin', alpha=0.7)
axs[0, 0].set_title("Bin Loss Components")
axs[0, 0].set_xlabel("Epoch")
axs[0, 0].legend()
axs[0, 0].grid(True, alpha=0.3)

axs[0, 1].plot(history['div'])
axs[0, 1].set_title("Diversity Loss")
axs[0, 1].set_xlabel("Epoch")
axs[0, 1].grid(True, alpha=0.3)

axs[0, 2].plot(history['hard'])
axs[0, 2].set_title("Hardness Loss")
axs[0, 2].set_xlabel("Epoch")
axs[0, 2].grid(True, alpha=0.3)

# Row 2: Variance metrics and KD loss
axs[1, 0].plot(history['intra_variance'], label='Intra-Bin Variance', color='red', alpha=0.8)
axs[1, 0].set_title("Intra-Bin Variance (Raw)")
axs[1, 0].set_xlabel("Epoch")
axs[1, 0].set_ylabel("Variance")
axs[1, 0].legend()
axs[1, 0].grid(True, alpha=0.3)

axs[1, 1].plot(history['inter_variance'], label='Inter-Bin Variance', color='green', alpha=0.8)
axs[1, 1].set_title("Inter-Bin Variance (Raw)")
axs[1, 1].set_xlabel("Epoch")
axs[1, 1].set_ylabel("Variance")
axs[1, 1].legend()
axs[1, 1].grid(True, alpha=0.3)

# Variance ratio (quality indicator)
variance_ratio = [inter / (intra + 1e-8) for intra, inter in zip(history['intra_variance'], history['inter_variance'])]
axs[1, 2].plot(variance_ratio, label='Inter/Intra Ratio', color='purple', alpha=0.8)
axs[1, 2].axhline(y=2.0, color='orange', linestyle='--', label='Target (2.0)')
axs[1, 2].set_title("Variance Ratio (Quality Indicator)")
axs[1, 2].set_xlabel("Epoch")
axs[1, 2].set_ylabel("Ratio")
axs[1, 2].legend()
axs[1, 2].grid(True, alpha=0.3)

# Row 3: Student performance
axs[2, 0].plot(history['stu'])
axs[2, 0].set_title("Student KD Loss")
axs[2, 0].set_xlabel("Epoch")
axs[2, 0].grid(True, alpha=0.3)

epochs_plot = [i*20 for i in range(len(history['test_acc']))]
axs[2, 1].plot(epochs_plot, [acc*100 for acc in history['test_acc']], 'b-', label='Student')
axs[2, 1].axhline(y=teacher_test_acc*100, color='r', linestyle='--', label='Teacher')
axs[2, 1].axhline(y=warmup_test_acc*100, color='g', linestyle=':', label='Warmup')
axs[2, 1].set_title("Test Accuracy Over Time")
axs[2, 1].set_xlabel("Epoch")
axs[2, 1].set_ylabel("Accuracy (%)")
axs[2, 1].legend()
axs[2, 1].grid(True, alpha=0.3)

axs[2, 2].plot(epochs_plot, [agr*100 for agr in history['agreement']], 'purple')
axs[2, 2].set_title("Teacher-Student Agreement")
axs[2, 2].set_xlabel("Epoch")
axs[2, 2].set_ylabel("Agreement (%)")
axs[2, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_with_replay.png', dpi=300, bbox_inches='tight')
print("\n📈 Training history saved to 'training_history_with_replay.png'")
plt.show()

print("\nFinal Learned Boundaries (Feature 0):")
bounds = bin_learner.get_boundaries()[0, 0, :].detach().cpu().numpy()
print(np.round(bounds, 2))

print("\n" + "="*90)
print("FINAL LEARNED BOUNDARIES (ALL FEATURES) - DEBUG")
print("="*90)
with torch.no_grad():
    final_bounds = bin_learner.get_boundaries()
    print(f"Raw shape: {final_bounds.shape}")
    print(f"Expected shape: (1, {DATA_DIM}, {NUM_BINS+1})")
    
    final_bounds_np = final_bounds[0].detach().cpu().numpy()
    print(f"After indexing [0]: {final_bounds_np.shape}")
    print(f"\nAll {DATA_DIM} features with {NUM_BINS+1} boundaries each:\n")
    
    for feat_idx in range(DATA_DIM):
        bounds_array = final_bounds_np[feat_idx]
        print(f"Feature {feat_idx:2d} (len={len(bounds_array)}): [{', '.join([f'{b:10.4f}' for b in bounds_array])}]")
        
        # Additional debug: show if boundaries are collapsing
        if len(bounds_array) == NUM_BINS + 1:
            unique_vals = len(set(np.round(bounds_array, 4)))
            if unique_vals < NUM_BINS + 1:
                print(f"           ⚠️ WARNING: Only {unique_vals} unique boundaries!")

print("\n" + "="*90)
print("✅ EXPERIMENT COMPLETE!")
print("="*90)

# ==========================================
# 6. SAVE MODELS
# ==========================================
print("\n💾 Saving models...")

# Save teacher model
torch.save({
    'model_state_dict': teacher.state_dict(),
    'train_acc': teacher_train_acc,
    'test_acc': teacher_test_acc,
    'architecture': f'{DATA_DIM} -> 128 -> 64 -> {NUM_CLASSES}'
}, f'teacher_model_{DATA_DIM}features_{NUM_CLASSES}classes.pt')

# Save final student model
torch.save({
    'model_state_dict': student.state_dict(),
    'train_acc': student_train_acc,
    'test_acc': student_test_acc,
    'agreement_train': train_agreement,
    'agreement_test': test_agreement,
    'architecture': f'{DATA_DIM} -> 32 -> {NUM_CLASSES}',
    'epoch': EPOCHS
}, f'student_final_{DATA_DIM}features_{NUM_CLASSES}classes.pt')

# Save best student model (based on agreement)
if best_student_state is not None:
    torch.save(best_student_state, f'student_best_{DATA_DIM}features_{NUM_CLASSES}classes.pt')
    print(f"✓ Best student saved from epoch {best_student_state['epoch']} "
          f"(Agreement: {best_student_state['agreement']*100:.2f}%, "
          f"Test Acc: {best_student_state['test_acc']*100:.2f}%)")

print(f"✓ Teacher model saved: teacher_model_{DATA_DIM}features_{NUM_CLASSES}classes.pt")
print(f"✓ Final student saved: student_final_{DATA_DIM}features_{NUM_CLASSES}classes.pt")
print(f"✓ Best student saved: student_best_{DATA_DIM}features_{NUM_CLASSES}classes.pt")

# Optional: Also save generator and bin_learner
torch.save({
    'generator': generator.state_dict(),
    'bin_learner': bin_learner.state_dict(),
}, f'auxiliary_models_{DATA_DIM}features_{NUM_CLASSES}classes.pt')
print(f"✓ Auxiliary models saved: auxiliary_models_{DATA_DIM}features_{NUM_CLASSES}classes.pt")


# After training completes, before or after the model saving section
print("\n" + "="*90)
print("VARIANCE MAP VISUALIZATION")
print("="*90)

# Get the final variance map
final_epoch = max(history['variance_maps'].keys())
var_map = history['variance_maps'][final_epoch]  # Shape: (Features, Bins)

print(f"\nFinal Variance Map (Epoch {final_epoch}):")
print(f"Shape: {var_map.shape} (Features x Bins)")
print(f"\nPer-feature statistics:")

for feat_idx in range(DATA_DIM):
    print(f"\nFeature {feat_idx:2d}:")
    for bin_idx in range(NUM_BINS):
        print(f"  Bin {bin_idx}: {var_map[feat_idx][bin_idx]:.6f}", end="")
        if var_map[feat_idx][bin_idx] == 0.0:
            print(" (empty or single sample)", end="")
        print()
    print(f"  Mean: {var_map[feat_idx].mean():.6f}")
    print(f"  Max:  {var_map[feat_idx].max():.6f}")

# Visualize as heatmap

plt.figure(figsize=(12, 10))
plt.imshow(var_map, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
plt.colorbar(label='Variance (lower = purer)')
plt.xlabel('Bin Index')
plt.ylabel('Feature Index')
plt.title(f'Hard-Assignment Variance Map (Epoch {final_epoch})')
plt.xticks(range(NUM_BINS))
plt.yticks(range(DATA_DIM))
plt.tight_layout()
plt.savefig('variance_heatmap.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Variance heatmap saved to 'variance_heatmap.png'")
plt.show()

# NEW: Visualize per-feature inter-bin variance
print("\n" + "="*90)
print("PER-FEATURE INTER-BIN VARIANCE (BIN SEPARATION QUALITY)")
print("="*90)

final_epoch = max(history['inter_variance_per_feature'].keys())
inter_var_per_feat = history['inter_variance_per_feature'][final_epoch]

print(f"\nPer-feature inter-bin variance (Epoch {final_epoch}):")
print(f"Higher values = better separated bins for that feature\n")

# Sort features by inter-bin variance
sorted_indices = np.argsort(inter_var_per_feat)[::-1]  # Descending order

print("Top 10 features with best bin separation:")
for i, feat_idx in enumerate(sorted_indices[:10], 1):
    print(f"  {i:2d}. Feature {feat_idx:2d}: {inter_var_per_feat[feat_idx]:.6f}")

print("\nBottom 10 features with worst bin separation:")
for i, feat_idx in enumerate(sorted_indices[-10:][::-1], 1):
    print(f"  {i:2d}. Feature {feat_idx:2d}: {inter_var_per_feat[feat_idx]:.6f}")

# Bar chart visualization
plt.figure(figsize=(14, 6))
plt.bar(range(DATA_DIM), inter_var_per_feat, color='steelblue', alpha=0.7)
plt.xlabel('Feature Index')
plt.ylabel('Inter-Bin Variance')
plt.title(f'Per-Feature Inter-Bin Variance (Epoch {final_epoch}) - Higher = Better Separation')
plt.xticks(range(DATA_DIM))
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('inter_variance_per_feature.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Per-feature inter-bin variance plot saved to 'inter_variance_per_feature.png'")
plt.show()

# ==========================================
# VARIANCE HEATMAP EVOLUTION COMPARISON
# ==========================================
print("\n" + "="*90)
print("VARIANCE HEATMAP EVOLUTION ACROSS EPOCHS")
print("="*90)

var_map_epochs = sorted(history['variance_maps'].keys())
num_epochs = len(var_map_epochs)

# Create subplot grid showing all epochs
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, epoch in enumerate(var_map_epochs):
    if idx >= 6:  # Limit to 6 subplots
        break
    
    var_map = history['variance_maps'][epoch]
    
    ax = axes[idx]
    im = ax.imshow(var_map, aspect='auto', cmap='RdYlGn_r', 
                   interpolation='nearest', vmin=0, vmax=0.5)  # Fixed scale for comparison
    ax.set_xlabel('Bin Index')
    ax.set_ylabel('Feature Index')
    ax.set_title(f'Epoch {epoch}')
    ax.set_xticks(range(NUM_BINS))
    ax.set_yticks(range(0, DATA_DIM, 5))  # Every 5 features

# Add shared colorbar
fig.colorbar(im, ax=axes, label='Intra-Bin Variance (lower = purer)', 
             orientation='horizontal', pad=0.05, aspect=40)

plt.tight_layout()
plt.savefig('variance_heatmap_evolution.png', dpi=300, bbox_inches='tight')
print("✅ Variance heatmap evolution saved to 'variance_heatmap_evolution.png'")
plt.show()

# Alternative: Create animated comparison with consistent scale
print("\nCreating individual epoch heatmaps for comparison...")

# Find global min/max for consistent color scale
all_values = [history['variance_maps'][epoch] for epoch in var_map_epochs]
vmin = min(vm.min() for vm in all_values)
vmax = max(vm.max() for vm in all_values)

for epoch in var_map_epochs:
    var_map = history['variance_maps'][epoch]
    
    plt.figure(figsize=(12, 10))
    plt.imshow(var_map, aspect='auto', cmap='RdYlGn_r', 
               interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Variance (lower = purer)')
    plt.xlabel('Bin Index')
    plt.ylabel('Feature Index')
    plt.title(f'Hard-Assignment Variance Map (Epoch {epoch})')
    plt.xticks(range(NUM_BINS))
    plt.yticks(range(DATA_DIM))
    plt.tight_layout()
    plt.savefig(f'variance_heatmap_epoch_{epoch:04d}.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"✅ Saved {len(var_map_epochs)} individual heatmaps: variance_heatmap_epoch_XXXX.png")

# Create difference heatmaps (change from previous epoch)
print("\nCreating difference heatmaps...")

for i in range(1, len(var_map_epochs)):
    prev_epoch = var_map_epochs[i-1]
    curr_epoch = var_map_epochs[i]
    
    prev_map = history['variance_maps'][prev_epoch]
    curr_map = history['variance_maps'][curr_epoch]
    
    diff_map = curr_map - prev_map
    
    plt.figure(figsize=(12, 10))
    im = plt.imshow(diff_map, aspect='auto', cmap='RdBu_r', 
                    interpolation='nearest', vmin=-0.1, vmax=0.1)
    plt.colorbar(label='Variance Change (blue = decreased, red = increased)')
    plt.xlabel('Bin Index')
    plt.ylabel('Feature Index')
    plt.title(f'Variance Change: Epoch {prev_epoch} → {curr_epoch}')
    plt.xticks(range(NUM_BINS))
    plt.yticks(range(DATA_DIM))
    plt.tight_layout()
    plt.savefig(f'variance_diff_{prev_epoch}_to_{curr_epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"✅ Saved {len(var_map_epochs)-1} difference heatmaps")