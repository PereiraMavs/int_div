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
EPOCHS = 400
LATENT_DIM = 32     # Generator Noise Dimension
NUM_BINS = 8        # Bins per feature (K)

# Warmup parameters
WARMUP_EPOCHS = 30  # Number of warmup epochs for student
WARMUP_SAMPLES = 500  # Number of random samples per warmup epoch

# Replay buffer parameters
REPLAY_BUFFER_SIZE = 1000  # Maximum samples to store
REPLAY_RATIO = 0.10  # 10% of batch will be replay samples  # FIXED: Comment now matches value

# Weights
LAMBDA_COV = 3.0    # Weight for Interaction Diversity (Entropy)
LAMBDA_HARD = 2.0   # Weight for Adversarial Hardness

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
# from sklearn.datasets import load_wine
# data = load_wine()
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
# 1.5 HARD VARIANCE CALCULATION UTILITY
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
# 1.6 RANDOM DATA GENERATION MODULE
# ==========================================

class RandomDataGenerator:
    """
    Generates random samples within valid feature ranges for warmup.
    Uses uniform distribution between min and max for each feature.
    """
    def __init__(self, min_vals, max_vals, device='cpu'):
        self.min_vals = min_vals.to(device)
        self.max_vals = max_vals.to(device)
        self.device = device
        self.num_features = len(min_vals)
    
    def generate(self, num_samples):
        """Generate random samples uniformly within [min, max] for each feature."""
        random_samples = torch.rand(num_samples, self.num_features, device=self.device)
        range_vals = self.max_vals - self.min_vals
        scaled_samples = self.min_vals + random_samples * range_vals
        return scaled_samples

# Initialize random data generator
random_gen = RandomDataGenerator(X_min, X_max, device=DEVICE)
print(f"✅ Random Data Generator initialized")

# ==========================================
# 1.7 REPLAY BUFFER
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
        """Add samples to the buffer. Overwrites old samples when full."""
        samples = samples.detach()
        
        if len(self.buffer) < self.max_size:
            for i in range(samples.shape[0]):
                if len(self.buffer) >= self.max_size:
                    break
                self.buffer.append(samples[i:i+1])
        else:
            for i in range(samples.shape[0]):
                self.buffer[self.position] = samples[i:i+1]
                self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size):
        """Sample random batch from buffer."""
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
            nn.Linear(input_dim, 32),  # Lightweight
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    """
    FIXED: Added clamping to ensure outputs stay within valid data range.
    """
    def __init__(self, latent_dim, output_dim, mean_vals, min_vals, max_vals):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim)
        )
        # Store min/max for clamping
        self.register_buffer('min_vals', min_vals)
        self.register_buffer('max_vals', max_vals)
        
        # Initialize bias to mean so generator starts in valid range
        with torch.no_grad():
            self.net[-1].bias.copy_(mean_vals.to(self.net[-1].weight.device))

    def forward(self, z):
        x = self.net(z)
        # FIXED: Clamp output to valid data range
        x = torch.clamp(x, self.min_vals, self.max_vals)
        return x

class BinLearner(nn.Module):
    def __init__(self, num_features, num_bins, min_vals, max_vals):
        super().__init__()
        self.num_bins = num_bins
        self.temperature = 1.0  # Annealed later
        
        self.register_buffer('min_val', min_vals.view(1, -1, 1))
        self.register_buffer('max_val', max_vals.view(1, -1, 1))
        
        # Stable initialization
        range_size = (max_vals - min_vals).view(1, -1, 1)
        target_width = range_size / num_bins
        target_width = target_width.expand(-1, -1, num_bins)
        
        # Handle large raw values preventing exp() overflow
        threshold = 20.0
        is_large = target_width > threshold
        safe_param = torch.zeros_like(target_width)
        safe_param[is_large] = target_width[is_large]
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
        sigmoids = torch.sigmoid((x_in - bounds) / self.temperature)
        membership = sigmoids[:, :, :-1] - sigmoids[:, :, 1:]
        membership = membership + 1e-8
        
        # Normalize to sum to 1.0 per feature
        membership = membership / (membership.sum(dim=2, keepdim=True) + 1e-8)
        
        return membership

# ==========================================
# 3. LOSS FUNCTIONS
# ==========================================

class InteractionDiversityLoss(nn.Module):
    """
    Maximizes Entropy of the Pairwise Joint Distribution.
    Goal: Uniform distribution across all KxK tiles.
    
    NOTE: Currently implements T=2 (pairwise) interactions only.
    For T>2, would need higher-order einsum or explicit loops.
    """
    def __init__(self, t_way=2):
        super().__init__()
        self.t_way = t_way
        if t_way != 2:
            raise NotImplementedError("Currently only T=2 (pairwise) is supported")
    
    def forward(self, membership):
        N, F, K = membership.shape
        epsilon = 1e-8
        
        # Joint Co-occurrence: (N, i, a), (N, j, b) -> (i, j, a, b)
        joint_counts = torch.einsum('nia, njb -> ijab', membership, membership)
        joint_probs = joint_counts / N
        
        # Mask diagonal
        mask = torch.triu(torch.ones(F, F), diagonal=1).bool().to(membership.device)
        valid_probs = joint_probs[mask]  # (Pairs, K, K)
        
        # Maximize Entropy = Minimize Sum(p log p)
        loss_entropy = torch.sum(valid_probs * torch.log(valid_probs + epsilon))
        
        return loss_entropy

class VarianceBasedBinLoss(nn.Module):
    """
    Variance-based bin loss using MSE:
    1. Intra-Bin: Minimize variance (make bins tight around their centroid)
    2. Inter-Bin: Maximize distance between adjacent bin centroids
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
        
        squared_distances = ((teacher_probs_expanded - centroids_expanded) ** 2).sum(dim=3)
        loss_intra = (membership * squared_distances).sum() / N
        
        # 2. Inter-Bin Loss: Maximize distance between adjacent centroids
        curr_bins = centroids[:, :-1, :]  # (F, K-1, C)
        next_bins = centroids[:, 1:, :]   # (F, K-1, C)
        
        inter_distances = ((curr_bins - next_bins) ** 2).sum(dim=2)  # (F, K-1)
        loss_inter = -1.0*inter_distances.sum() / F
        
        # 3. Compute HARD-ASSIGNMENT variances for monitoring
        intra_var, inter_var, _, _ = calculate_hard_variance(membership, teacher_probs)  # UPDATED
        
        return 1.0 * loss_intra + (3.0 * loss_inter), loss_intra, loss_inter, intra_var, inter_var

# ==========================================
# 3.5 COVERAGE CERTIFICATION
# ==========================================

def compute_coverage(membership, threshold=5):
    """
    Compute fraction of K×K bin pairs with ≥threshold samples.
    
    This provides an interpretable robustness metric:
    "This Student was stress-tested on X% of all pairwise feature interactions."
    
    Args:
        membership: Tensor of shape (N, F, K) - soft membership values
        threshold: Minimum sample count to consider a bin pair "covered"
    
    Returns:
        coverage_ratio: Fraction of bin pairs covered
        coverage_matrix: (F, F) matrix showing coverage per feature pair
    """
    N, F, K = membership.shape
    
    with torch.no_grad():
        hard_assignments = membership.argmax(dim=2)  # (N, F)
        
        coverage_matrix = torch.zeros(F, F)
        total_pairs = 0
        covered_pairs = 0
        
        for i in range(F):
            for j in range(i + 1, F):
                counts = torch.zeros(K, K, device=membership.device)
                for n in range(N):
                    bin_i = hard_assignments[n, i]
                    bin_j = hard_assignments[n, j]
                    counts[bin_i, bin_j] += 1
                
                pair_covered = (counts >= threshold).sum().item()
                pair_total = K * K
                
                coverage_matrix[i, j] = pair_covered / pair_total
                coverage_matrix[j, i] = pair_covered / pair_total
                
                covered_pairs += pair_covered
                total_pairs += pair_total
        
        coverage_ratio = covered_pairs / total_pairs if total_pairs > 0 else 0.0
        
    return coverage_ratio, coverage_matrix

# ==========================================
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
    teacher_train_logits = teacher(X_train_tensor)
    teacher_train_preds = torch.argmax(teacher_train_logits, dim=1)
    teacher_train_acc = accuracy_score(y_train, teacher_train_preds.cpu().numpy())
    
    teacher_test_logits = teacher(X_test_tensor)
    teacher_test_preds = torch.argmax(teacher_test_logits, dim=1)
    teacher_test_acc = accuracy_score(y_test, teacher_test_preds.cpu().numpy())

print(f"✅ Teacher Frozen.")
print(f"   Train Accuracy: {teacher_train_acc*100:.2f}%")
print(f"   Test Accuracy:  {teacher_test_acc*100:.2f}%")

# B. Initialize Modules
bin_learner = BinLearner(DATA_DIM, NUM_BINS, X_min, X_max).to(DEVICE)
generator = Generator(LATENT_DIM, DATA_DIM, X_mean, X_min, X_max).to(DEVICE)  # FIXED: Pass min/max
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
    x_random = random_gen.generate(WARMUP_SAMPLES)
    replay_buffer.add(x_random)
    
    teacher.eval()
    with torch.no_grad():
        teacher_probs = F.softmax(teacher(x_random), dim=1)
    
    student.train()
    warmup_opt.zero_grad()
    
    student_logits = student(x_random)
    student_log_probs = F.log_softmax(student_logits, dim=1)
    
    loss = loss_kl_fn(student_log_probs, teacher_probs)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    warmup_opt.step()
    
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
loss_bin_fn = VarianceBasedBinLoss()
loss_div_fn = InteractionDiversityLoss(t_way=2)

opt_bin = optim.Adam(bin_learner.parameters(), lr=0.01)
opt_gen = optim.Adam(generator.parameters(), lr=0.001)
opt_stu = optim.Adam(student.parameters(), lr=0.001)

# ADDED: Learning rate schedulers
scheduler_bin = optim.lr_scheduler.CosineAnnealingLR(opt_bin, T_max=EPOCHS)
scheduler_gen = optim.lr_scheduler.CosineAnnealingLR(opt_gen, T_max=EPOCHS)
scheduler_stu = optim.lr_scheduler.CosineAnnealingLR(opt_stu, T_max=EPOCHS)

history = {
    'bin': [], 'div': [], 'hard': [], 'stu': [], 'test_acc': [], 'agreement': [],
    'bin_intra': [], 'bin_inter': [],
    'intra_variance': [], 'inter_variance': [],
    'inter_variance_per_feature': {},  # NEW: Store per-feature inter-bin variance
    'coverage': [],
    'variance_maps': {},
    'boundaries': {},
    'memberships': {}
}

# Store initial boundaries
with torch.no_grad():
    initial_boundaries = bin_learner.get_boundaries().detach().cpu().numpy()
    history['boundaries'][0] = initial_boundaries

print("\n⚔️ Starting Main Training...")
print(f"   Using replay buffer: {REPLAY_RATIO*100:.0f}% of batch from warmup samples")
print(f"{'Epoch':<6} | {'Bin':<8} | {'Div':<8} | {'Hard':<8} | {'Stu':<8} | {'Test Acc':<10} | {'Agreement':<10} | {'Coverage':<10}")
print("-" * 110)

best_student_state = None
best_agreement_score = 0.0

for epoch in range(1, EPOCHS + 1):
    
    # Anneal Temperature: 0.5 -> 5.0
    bin_learner.temperature = 4.0 - (3.5 * (epoch / EPOCHS))  # 5.0 -> 2.0 (hard to soft boundaries)
    z = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
    
    # ==========================
    # PHASE 1: BIN LEARNER (Purity + Repulsion)
    # ==========================
    opt_bin.zero_grad()
    x_gen = generator(z).detach()

    with torch.no_grad():
        t_probs = F.softmax(teacher(x_gen), dim=1)
        
    mship = bin_learner(x_gen)
    l_bin, l_intra, l_inter, intra_var, inter_var = loss_bin_fn(mship, t_probs)

    if not torch.isnan(l_bin):
        l_bin.backward()
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
    # FIXED: Both Teacher and Student are now frozen for gradient computation
    with torch.no_grad():
        t_probs_2 = F.softmax(teacher(x_gen_2), dim=1)
        s_log_probs = F.log_softmax(student(x_gen_2), dim=1)  # FIXED: Now inside no_grad
    
    # Minimize Negative KL (equivalent to maximizing KL divergence)
    l_hard = -1.0 * F.kl_div(s_log_probs, t_probs_2, reduction='batchmean')
    
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
    
    # Mix synthetic samples with replay buffer samples
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
    
    # Step schedulers
    scheduler_bin.step()
    scheduler_gen.step()
    scheduler_stu.step()
    
    # Logging
    history['bin'].append(l_bin.item())
    history['bin_intra'].append(l_intra.item())
    history['bin_inter'].append(l_inter.item())
    history['div'].append(l_div.item())
    history['hard'].append(l_hard.item())
    history['stu'].append(l_stu.item())
    history['intra_variance'].append(intra_var)
    history['inter_variance'].append(inter_var)
    
    if epoch % 20 == 0:
        student.eval()
        with torch.no_grad():
            s_test_preds = torch.argmax(student(X_test_tensor), dim=1)
            t_test_preds = torch.argmax(teacher(X_test_tensor), dim=1)
            test_acc = accuracy_score(y_test, s_test_preds.cpu().numpy())
            agreement = (t_test_preds == s_test_preds).float().mean().item()
            
            # ADDED: Compute coverage certification
            z_cov = torch.randn(1000, LATENT_DIM).to(DEVICE)
            x_cov = generator(z_cov)
            mship_cov = bin_learner(x_cov)
            coverage_ratio, _ = compute_coverage(mship_cov, threshold=5)
            
        history['test_acc'].append(test_acc)
        history['agreement'].append(agreement)
        history['coverage'].append(coverage_ratio)
        student.train()
        
        print(f"{epoch:<6} | {l_bin.item():<8.3f} | {l_div.item():<8.1f} | {l_hard.item():<8.3f} | {l_stu.item():<8.3f} | {test_acc*100:<10.1f}% | {agreement*100:<10.1f}% | {coverage_ratio*100:<10.1f}%")
        
        # Track best student model
        if agreement > best_agreement_score:
            best_agreement_score = agreement
            best_student_state = {
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'test_acc': test_acc,
                'agreement': agreement,
                'coverage': coverage_ratio
            }
    
    # Store boundaries and membership info every 50 iterations
    if epoch % 50 == 0:
        with torch.no_grad():
            boundaries = bin_learner.get_boundaries().detach().cpu().numpy()
            history['boundaries'][epoch] = boundaries
            
            z_sample = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
            x_sample = generator(z_sample)
            membership_sample = bin_learner(x_sample)
            
            mean_membership = membership_sample.mean(axis=0).detach().cpu().numpy()
            std_membership = membership_sample.std(axis=0).detach().cpu().numpy()
            
            history['memberships'][epoch] = {
                'mean': mean_membership,
                'std': std_membership,
                'temperature': bin_learner.temperature
            }
            
            t_probs_sample = F.softmax(teacher(x_sample), dim=1)
            _, _, var_map, inter_var_per_feat = calculate_hard_variance(membership_sample, t_probs_sample)  # UPDATED
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
    student_test_logits = student(X_test_tensor)
    student_test_preds = torch.argmax(student_test_logits, dim=1)
    student_test_acc = accuracy_score(y_test, student_test_preds.cpu().numpy())
    
    teacher_test_preds = torch.argmax(teacher(X_test_tensor), dim=1)
    test_agreement = (teacher_test_preds == student_test_preds).float().mean().item()
    
    student_train_logits = student(X_train_tensor)
    student_train_preds = torch.argmax(student_train_logits, dim=1)
    student_train_acc = accuracy_score(y_train, student_train_preds.cpu().numpy())
    
    teacher_train_preds = torch.argmax(teacher(X_train_tensor), dim=1)
    train_agreement = (teacher_train_preds == student_train_preds).float().mean().item()
    
    # Final coverage certification
    z_final = torch.randn(2000, LATENT_DIM).to(DEVICE)
    x_final = generator(z_final)
    mship_final = bin_learner(x_final)
    final_coverage, coverage_matrix = compute_coverage(mship_final, threshold=5)

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

print(f"\n📋 COVERAGE CERTIFICATION:")
print(f"   Final Coverage: {final_coverage*100:.2f}% of pairwise feature interactions")
print(f"   (Threshold: 5 samples per bin pair)")

print(f"\n💾 REPLAY BUFFER STATS:")
print(f"   Total samples stored: {len(replay_buffer)}")
print(f"   Samples used per batch: {int(BATCH_SIZE * REPLAY_RATIO)}")

# ==========================================
# 6. VISUALIZATION
# ==========================================
fig, axs = plt.subplots(4, 3, figsize=(15, 16))  # Changed to 4 rows

# Row 1: Loss curves
axs[0, 0].plot(history['bin'], label='Total', alpha=0.8, color='blue')
axs[0, 0].set_title("BinLearner Total Loss")
axs[0, 0].set_xlabel("Epoch")
axs[0, 0].set_ylabel("Loss")
axs[0, 0].legend()
axs[0, 0].grid(True, alpha=0.3)

# NEW: Separate Intra Loss plot
axs[0, 1].plot(history['bin_intra'], label='Intra Loss', alpha=0.8, color='red')
axs[0, 1].set_title("Intra-Bin Loss (minimize variance)")
axs[0, 1].set_xlabel("Epoch")
axs[0, 1].set_ylabel("Loss")
axs[0, 1].legend()
axs[0, 1].grid(True, alpha=0.3)

# NEW: Separate Inter Loss plot (negated for clarity)
axs[0, 2].plot([-x for x in history['bin_inter']], label='Inter Distance', alpha=0.8, color='green')
axs[0, 2].set_title("Inter-Bin Distance (higher = better)")
axs[0, 2].set_xlabel("Epoch")
axs[0, 2].set_ylabel("Distance")
axs[0, 2].legend()
axs[0, 2].grid(True, alpha=0.3)

# Row 2: Generator losses
axs[1, 0].plot(history['div'], color='orange')
axs[1, 0].set_title("Diversity Loss (Entropy)")
axs[1, 0].set_xlabel("Epoch")
axs[1, 0].set_ylabel("Loss")
axs[1, 0].grid(True, alpha=0.3)

axs[1, 1].plot(history['hard'], color='brown')
axs[1, 1].set_title("Hardness Loss")
axs[1, 1].set_xlabel("Epoch")
axs[1, 1].set_ylabel("Loss")
axs[1, 1].grid(True, alpha=0.3)

axs[1, 2].plot(history['stu'], color='cyan')
axs[1, 2].set_title("Student KD Loss")
axs[1, 2].set_xlabel("Epoch")
axs[1, 2].set_ylabel("Loss")
axs[1, 2].grid(True, alpha=0.3)

# Row 3: Variance metrics
axs[2, 0].plot(history['intra_variance'], label='Intra-Bin Variance', color='red', alpha=0.8)
axs[2, 0].set_title("Intra-Bin Variance (Raw)")
axs[2, 0].set_xlabel("Epoch")
axs[2, 0].set_ylabel("Variance")
axs[2, 0].legend()
axs[2, 0].grid(True, alpha=0.3)

axs[2, 1].plot(history['inter_variance'], label='Inter-Bin Variance', color='green', alpha=0.8)
axs[2, 1].set_title("Inter-Bin Variance (Raw)")
axs[2, 1].set_xlabel("Epoch")
axs[2, 1].set_ylabel("Variance")
axs[2, 1].legend()
axs[2, 1].grid(True, alpha=0.3)

# Coverage over time
epochs_plot = [i*20 for i in range(len(history['coverage']))]
axs[2, 2].plot(epochs_plot, [c*100 for c in history['coverage']], color='purple', alpha=0.8)
axs[2, 2].set_title("Coverage Certification")
axs[2, 2].set_xlabel("Epoch")
axs[2, 2].set_ylabel("Coverage (%)")
axs[2, 2].grid(True, alpha=0.3)

# Row 4: Student performance
epochs_plot = [i*20 for i in range(len(history['test_acc']))]
axs[3, 0].plot(epochs_plot, [acc*100 for acc in history['test_acc']], 'b-', label='Student')
axs[3, 0].axhline(y=teacher_test_acc*100, color='r', linestyle='--', label='Teacher')
axs[3, 0].axhline(y=warmup_test_acc*100, color='g', linestyle=':', label='Warmup')
axs[3, 0].set_title("Test Accuracy Over Time")
axs[3, 0].set_xlabel("Epoch")
axs[3, 0].set_ylabel("Accuracy (%)")
axs[3, 0].legend()
axs[3, 0].grid(True, alpha=0.3)

axs[3, 1].plot(epochs_plot, [agr*100 for agr in history['agreement']], 'purple')
axs[3, 1].set_title("Teacher-Student Agreement")
axs[3, 1].set_xlabel("Epoch")
axs[3, 1].set_ylabel("Agreement (%)")
axs[3, 1].grid(True, alpha=0.3)

# NEW: Loss ratio plot
loss_ratio = [history['bin_intra'][i] / (abs(history['bin_inter'][i]) + 1e-8) for i in range(len(history['bin_intra']))]
axs[3, 2].plot(loss_ratio, color='magenta', alpha=0.8)
axs[3, 2].set_title("Intra/Inter Loss Ratio (lower = better)")
axs[3, 2].set_xlabel("Epoch")
axs[3, 2].set_ylabel("Ratio")
axs[3, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_corrected.png', dpi=300, bbox_inches='tight')
print("\n📈 Training history saved to 'training_history_corrected.png'")
plt.show()

# ==========================================
# 7. SAVE MODELS
# ==========================================
print("\n💾 Saving models...")

torch.save({
    'model_state_dict': teacher.state_dict(),
    'train_acc': teacher_train_acc,
    'test_acc': teacher_test_acc,
    'architecture': f'{DATA_DIM} -> 128 -> 64 -> {NUM_CLASSES}'
}, f'teacher_model_{DATA_DIM}features_{NUM_CLASSES}classes.pt')

torch.save({
    'model_state_dict': student.state_dict(),
    'train_acc': student_train_acc,
    'test_acc': student_test_acc,
    'agreement_train': train_agreement,
    'agreement_test': test_agreement,
    'coverage': final_coverage,
    'architecture': f'{DATA_DIM} -> 32 -> {NUM_CLASSES}',
    'epoch': EPOCHS
}, f'student_final_{DATA_DIM}features_{NUM_CLASSES}classes.pt')

if best_student_state is not None:
    torch.save(best_student_state, f'student_best_{DATA_DIM}features_{NUM_CLASSES}classes.pt')
    print(f"✓ Best student saved from epoch {best_student_state['epoch']} "
          f"(Agreement: {best_student_state['agreement']*100:.2f}%, "
          f"Test Acc: {best_student_state['test_acc']*100:.2f}%, "
          f"Coverage: {best_student_state['coverage']*100:.2f}%)")

torch.save({
    'generator': generator.state_dict(),
    'bin_learner': bin_learner.state_dict(),
}, f'auxiliary_models_{DATA_DIM}features_{NUM_CLASSES}classes.pt')

print(f"✓ All models saved successfully")

# ==========================================
# 8. COMPREHENSIVE VARIANCE REPORT
# ==========================================
print("\n📝 Generating comprehensive variance report...")

variance_report_filename = f'variance_report_{DATA_DIM}features_{NUM_CLASSES}classes.txt'

with open(variance_report_filename, 'w') as f:
    f.write("="*90 + "\n")
    f.write("COMPREHENSIVE VARIANCE ANALYSIS REPORT\n")
    f.write("="*90 + "\n\n")
    
    f.write("REPORT OVERVIEW\n")
    f.write("-"*90 + "\n")
    f.write(f"Dataset: {data.__class__.__name__}\n")
    f.write(f"Features: {DATA_DIM}\n")
    f.write(f"Classes: {NUM_CLASSES}\n")
    f.write(f"Bins per feature: {NUM_BINS}\n")
    f.write(f"Training epochs: {EPOCHS}\n")
    f.write(f"Variance snapshots: Every 50 epochs\n\n")
    
    # ==========================================
    # SECTION 1: GLOBAL VARIANCE EVOLUTION
    # ==========================================
    f.write("="*90 + "\n")
    f.write("SECTION 1: GLOBAL VARIANCE AND LOSS EVOLUTION (ALL EPOCHS)\n")
    f.write("="*90 + "\n\n")

    # SECTION 1A: Loss Evolution
    f.write("SECTION 1A: BIN LOSS EVOLUTION\n")
    f.write("-"*90 + "\n\n")

    f.write(f"{'Epoch':<8} | {'Total Bin Loss':<15} | {'Intra Loss':<15} | {'Inter Loss':<15} | {'Inter Distance':<15}\n")
    f.write("-"*90 + "\n")

    # Print every 20 epochs
    for epoch in range(1, EPOCHS + 1):
        if epoch % 20 == 0 or epoch == 1:
            total = history['bin'][epoch-1]
            intra_loss = history['bin_intra'][epoch-1]
            inter_loss = history['bin_inter'][epoch-1]
            inter_dist = -inter_loss  # Negate to show as distance
            f.write(f"{epoch:<8} | {total:<15.6f} | {intra_loss:<15.6f} | {inter_loss:<15.6f} | {inter_dist:<15.6f}\n")

    # Loss summary statistics
    f.write("\nLoss Summary Statistics:\n")
    total_init = history['bin'][0]
    total_final = history['bin'][-1]
    intra_loss_init = history['bin_intra'][0]
    intra_loss_final = history['bin_intra'][-1]
    inter_loss_init = history['bin_inter'][0]
    inter_loss_final = history['bin_inter'][-1]

    f.write(f"  Total Bin Loss:\n")
    f.write(f"    Initial: {total_init:.6f}\n")
    f.write(f"    Final:   {total_final:.6f}\n")
    f.write(f"    Change:  {total_final - total_init:+.6f} ({((total_final - total_init) / abs(total_init) * 100):+.2f}%)\n")
    f.write(f"    Min:     {min(history['bin']):.6f}\n")
    f.write(f"    Max:     {max(history['bin']):.6f}\n\n")

    f.write(f"  Intra-Bin Loss (minimize variance within bins):\n")
    f.write(f"    Initial: {intra_loss_init:.6f}\n")
    f.write(f"    Final:   {intra_loss_final:.6f}\n")
    f.write(f"    Change:  {intra_loss_final - intra_loss_init:+.6f} ({((intra_loss_final - intra_loss_init) / intra_loss_init * 100):+.2f}%)\n")
    f.write(f"    Min:     {min(history['bin_intra']):.6f}\n")
    f.write(f"    Max:     {max(history['bin_intra']):.6f}\n\n")

    f.write(f"  Inter-Bin Loss (maximize distance between bins, negative value):\n")
    f.write(f"    Initial: {inter_loss_init:.6f}\n")
    f.write(f"    Final:   {inter_loss_final:.6f}\n")
    f.write(f"    Change:  {inter_loss_final - inter_loss_init:+.6f} ({((inter_loss_final - inter_loss_init) / abs(inter_loss_init) * 100):+.2f}%)\n")
    f.write(f"    Min:     {min(history['bin_inter']):.6f}\n")
    f.write(f"    Max:     {max(history['bin_inter']):.6f}\n\n")

    f.write(f"  Inter-Bin Distance (negated inter loss, higher = better):\n")
    f.write(f"    Initial: {-inter_loss_init:.6f}\n")
    f.write(f"    Final:   {-inter_loss_final:.6f}\n")
    f.write(f"    Change:  {-inter_loss_final - (-inter_loss_init):+.6f}\n\n")

    # Loss ratio evolution
    loss_ratio_init = intra_loss_init / (abs(inter_loss_init) + 1e-8)
    loss_ratio_final = intra_loss_final / (abs(inter_loss_final) + 1e-8)
    f.write(f"  Loss Balance Ratio (Intra/|Inter|, lower = better separation):\n")
    f.write(f"    Initial: {loss_ratio_init:.2f}\n")
    f.write(f"    Final:   {loss_ratio_final:.2f}\n")
    f.write(f"    Change:  {loss_ratio_final - loss_ratio_init:+.2f} ({((loss_ratio_final - loss_ratio_init) / loss_ratio_init * 100):+.2f}%)\n\n")

    # SECTION 1B: Variance Monitoring (existing code)
    f.write("\n" + "-"*90 + "\n")
    f.write("SECTION 1B: VARIANCE MONITORING METRICS\n")
    f.write("-"*90 + "\n\n")

    f.write(f"{'Epoch':<8} | {'Intra Variance':<15} | {'Inter Variance':<15} | {'Ratio (Inter/Intra)':<20}\n")
    f.write("-"*90 + "\n")

    # Print every 20 epochs
    for epoch in range(1, EPOCHS + 1):
        if epoch % 20 == 0 or epoch == 1:
            intra = history['intra_variance'][epoch-1]
            inter = history['inter_variance'][epoch-1]
            ratio = inter / (intra + 1e-8)
            f.write(f"{epoch:<8} | {intra:<15.6f} | {inter:<15.6f} | {ratio:<20.2f}\n")

    # Summary statistics (existing code continues...)
    f.write("\nSummary Statistics:\n")
    intra_init = history['intra_variance'][0]
    intra_final = history['intra_variance'][-1]
    inter_init = history['inter_variance'][0]
    inter_final = history['inter_variance'][-1]
    
    f.write(f"  Intra-Bin Variance:\n")
    f.write(f"    Initial: {intra_init:.6f}\n")
    f.write(f"    Final:   {intra_final:.6f}\n")
    f.write(f"    Change:  {intra_final - intra_init:+.6f} ({((intra_final - intra_init) / intra_init * 100):+.2f}%)\n")
    f.write(f"    Min:     {min(history['intra_variance']):.6f}\n")
    f.write(f"    Max:     {max(history['intra_variance']):.6f}\n\n")
    
    f.write(f"  Inter-Bin Variance:\n")
    f.write(f"    Initial: {inter_init:.6f}\n")
    f.write(f"    Final:   {inter_final:.6f}\n")
    f.write(f"    Change:  {inter_final - inter_init:+.6f} ({((inter_final - inter_init) / inter_init * 100):+.2f}%)\n")
    f.write(f"    Min:     {min(history['inter_variance']):.6f}\n")
    f.write(f"    Max:     {max(history['inter_variance']):.6f}\n\n")
    
    ratio_init = inter_init / (intra_init + 1e-8)
    ratio_final = inter_final / (intra_final + 1e-8)
    f.write(f"  Quality Ratio (Inter/Intra):\n")
    f.write(f"    Initial: {ratio_init:.2f}\n")
    f.write(f"    Final:   {ratio_final:.2f}\n")
    f.write(f"    Change:  {ratio_final - ratio_init:+.2f} ({((ratio_final - ratio_init) / ratio_init * 100):+.2f}%)\n\n")
    
    # ==========================================
    # SECTION 2: PER-EPOCH DETAILED ANALYSIS
    # ==========================================
    f.write("\n" + "="*90 + "\n")
    f.write("SECTION 2: DETAILED VARIANCE MAPS (EVERY 50 EPOCHS)\n")
    f.write("="*90 + "\n\n")
    
    var_map_epochs = sorted(history['variance_maps'].keys())
    
    for epoch in var_map_epochs:
        f.write(f"\n{'='*90}\n")
        f.write(f"EPOCH {epoch}\n")
        f.write(f"{'='*90}\n\n")
        
        var_map = history['variance_maps'][epoch]
        inter_var_per_feat = history['inter_variance_per_feature'][epoch]
        
        # Overall statistics
        f.write(f"Overall Statistics:\n")
        f.write(f"  Intra-bin variance (all bins):\n")
        f.write(f"    Mean: {var_map.mean():.6f}\n")
        f.write(f"    Std:  {var_map.std():.6f}\n")
        f.write(f"    Min:  {var_map.min():.6f}\n")
        f.write(f"    Max:  {var_map.max():.6f}\n\n")
        
        f.write(f"  Inter-bin variance (per feature):\n")
        f.write(f"    Mean: {inter_var_per_feat.mean():.6f}\n")
        f.write(f"    Std:  {inter_var_per_feat.std():.6f}\n")
        f.write(f"    Min:  {inter_var_per_feat.min():.6f}\n")
        f.write(f"    Max:  {inter_var_per_feat.max():.6f}\n\n")
        
        # Count quality categories
        excellent = ((inter_var_per_feat > 0.2) & (var_map.mean(axis=1) < 0.01)).sum()
        good = ((inter_var_per_feat > 0.1) & (var_map.mean(axis=1) < 0.02)).sum()
        collapsed = (inter_var_per_feat == 0).sum()
        
        f.write(f"  Quality Categories:\n")
        f.write(f"    Excellent (inter>0.2, intra<0.01): {excellent}/{DATA_DIM}\n")
        f.write(f"    Good (inter>0.1, intra<0.02):      {good}/{DATA_DIM}\n")
        f.write(f"    Collapsed (inter=0):                {collapsed}/{DATA_DIM}\n\n")
        
        # Per-feature table
        f.write(f"Per-Feature Analysis:\n")
        f.write(f"{'Feat':<6} | {'Intra Mean':<12} | {'Intra Min':<12} | {'Intra Max':<12} | {'Inter':<12} | {'Quality':<10}\n")
        f.write("-"*90 + "\n")
        
        for feat_idx in range(DATA_DIM):
            intra_mean = var_map[feat_idx].mean()
            intra_min = var_map[feat_idx].min()
            intra_max = var_map[feat_idx].max()
            inter_val = inter_var_per_feat[feat_idx]
            
            # Determine quality
            if inter_val == 0:
                quality = "COLLAPSED"
            elif inter_val > 0.2 and intra_mean < 0.01:
                quality = "EXCELLENT"
            elif inter_val > 0.1 and intra_mean < 0.02:
                quality = "GOOD"
            elif inter_val < 0.05:
                quality = "POOR"
            else:
                quality = "FAIR"
            
            f.write(f"{feat_idx:<6} | {intra_mean:<12.6f} | {intra_min:<12.6f} | {intra_max:<12.6f} | {inter_val:<12.6f} | {quality:<10}\n")
        
        # Top/Bottom features
        sorted_indices = np.argsort(inter_var_per_feat)[::-1]
        
        f.write(f"\n🏆 Top 10 Features (Best Separation):\n")
        for i, feat_idx in enumerate(sorted_indices[:10], 1):
            f.write(f"  {i:2d}. Feature {feat_idx:2d}: Inter={inter_var_per_feat[feat_idx]:.6f}, Intra={var_map[feat_idx].mean():.6f}\n")
        
        f.write(f"\n⚠️  Bottom 10 Features (Worst Separation):\n")
        for i, feat_idx in enumerate(sorted_indices[-10:][::-1], 1):
            f.write(f"  {i:2d}. Feature {feat_idx:2d}: Inter={inter_var_per_feat[feat_idx]:.6f}, Intra={var_map[feat_idx].mean():.6f}\n")
    
    # ==========================================
    # SECTION 3: PER-FEATURE EVOLUTION
    # ==========================================
    f.write("\n\n" + "="*90 + "\n")
    f.write("SECTION 3: PER-FEATURE VARIANCE EVOLUTION\n")
    f.write("="*90 + "\n\n")
    
    for feat_idx in range(DATA_DIM):
        f.write(f"\n{'='*90}\n")
        f.write(f"FEATURE {feat_idx}\n")
        f.write(f"{'='*90}\n\n")
        
        # Evolution table
        f.write(f"{'Epoch':<8} | {'Intra Mean':<12} | {'Intra Min':<12} | {'Intra Max':<12} | {'Inter':<12} | {'Bin Variances':<40}\n")
        f.write("-"*90 + "\n")
        
        for epoch in var_map_epochs:
            var_map = history['variance_maps'][epoch]
            inter_var_per_feat = history['inter_variance_per_feature'][epoch]
            
            intra_mean = var_map[feat_idx].mean()
            intra_min = var_map[feat_idx].min()
            intra_max = var_map[feat_idx].max()
            inter_val = inter_var_per_feat[feat_idx]
            
            bin_vals = ", ".join([f"{var_map[feat_idx][k]:.4f}" for k in range(NUM_BINS)])
            
            f.write(f"{epoch:<8} | {intra_mean:<12.6f} | {intra_min:<12.6f} | {intra_max:<12.6f} | {inter_val:<12.6f} | [{bin_vals}]\n")
        
        # Evolution statistics
        first_epoch = var_map_epochs[0]
        last_epoch = var_map_epochs[-1]
        
        first_intra = history['variance_maps'][first_epoch][feat_idx].mean()
        last_intra = history['variance_maps'][last_epoch][feat_idx].mean()
        first_inter = history['inter_variance_per_feature'][first_epoch][feat_idx]
        last_inter = history['inter_variance_per_feature'][last_epoch][feat_idx]
        
        f.write(f"\nEvolution Summary:\n")
        f.write(f"  Intra Variance: {first_intra:.6f} → {last_intra:.6f} (Change: {last_intra - first_intra:+.6f})\n")
        f.write(f"  Inter Variance: {first_inter:.6f} → {last_inter:.6f} (Change: {last_inter - first_inter:+.6f})\n")
        
        if last_intra < first_intra and last_inter > first_inter:
            f.write(f"  Status: ✅ IMPROVED (purer bins, better separation)\n")
        elif last_intra > first_intra and last_inter < first_inter:
            f.write(f"  Status: ❌ DEGRADED (mixed bins, worse separation)\n")
        elif last_intra < first_intra:
            f.write(f"  Status: ⚠️  PARTIAL (purer bins, but separation unchanged)\n")
        elif last_inter > first_inter:
            f.write(f"  Status: ⚠️  PARTIAL (better separation, but bins not purer)\n")
        else:
            f.write(f"  Status: → NO CHANGE\n")
    
    # ==========================================
    # SECTION 4: SUMMARY AND RECOMMENDATIONS
    # ==========================================
    f.write("\n\n" + "="*90 + "\n")
    f.write("SECTION 4: SUMMARY AND RECOMMENDATIONS\n")
    f.write("="*90 + "\n\n")
    
    final_var_map = history['variance_maps'][var_map_epochs[-1]]
    final_inter_var = history['inter_variance_per_feature'][var_map_epochs[-1]]
    
    excellent_features = ((final_inter_var > 0.2) & (final_var_map.mean(axis=1) < 0.01)).sum()
    good_features = ((final_inter_var > 0.1) & (final_var_map.mean(axis=1) < 0.02)).sum()
    collapsed_features = (final_inter_var == 0).sum()
    
    f.write(f"Final Quality Assessment:\n")
    f.write(f"  Excellent features: {excellent_features}/{DATA_DIM} ({excellent_features/DATA_DIM*100:.1f}%)\n")
    f.write(f"  Good features:      {good_features}/{DATA_DIM} ({good_features/DATA_DIM*100:.1f}%)\n")
    f.write(f"  Collapsed features: {collapsed_features}/{DATA_DIM} ({collapsed_features/DATA_DIM*100:.1f}%)\n\n")
    
    if excellent_features >= DATA_DIM * 0.5:
        f.write("Overall Assessment: ✅ EXCELLENT - Majority of features have well-separated, pure bins\n")
    elif good_features >= DATA_DIM * 0.5:
        f.write("Overall Assessment: ✅ GOOD - Majority of features have reasonable bin quality\n")
    elif collapsed_features >= DATA_DIM * 0.3:
        f.write("Overall Assessment: ⚠️  NEEDS IMPROVEMENT - Many features have collapsed bins\n")
    else:
        f.write("Overall Assessment: 🔧 FAIR - Mixed results across features\n")
    
    f.write("\nRecommendations:\n")
    if collapsed_features > 0:
        collapsed_feat_list = [str(i) for i in range(DATA_DIM) if final_inter_var[i] == 0]
        f.write(f"  - Features {', '.join(collapsed_feat_list)} have collapsed bins. Consider:\n")
        f.write(f"    * Reducing number of bins\n")
        f.write(f"    * Adjusting temperature annealing schedule\n")
        f.write(f"    * These features may not be discriminative for this task\n\n")
    
    if excellent_features < DATA_DIM * 0.3:
        f.write(f"  - Only {excellent_features} features achieved excellent quality. Consider:\n")
        f.write(f"    * Increasing bin loss weight\n")
        f.write(f"    * Training for more epochs\n")
        f.write(f"    * Adjusting inter-bin loss weight\n\n")
    
    f.write("\n" + "="*90 + "\n")
    f.write("END OF VARIANCE REPORT\n")
    f.write("="*90 + "\n")

print(f"✅ Comprehensive variance report saved to '{variance_report_filename}'")

# Also save the raw data
torch.save({
    'variance_maps': history['variance_maps'],
    'inter_variance_per_feature': history['inter_variance_per_feature'],
    'intra_variance': history['intra_variance'],
    'inter_variance': history['inter_variance']
}, f'variance_data_{DATA_DIM}features_{NUM_CLASSES}classes.pt')

print(f"✅ Variance data saved to 'variance_data_{DATA_DIM}features_{NUM_CLASSES}classes.pt'")

print("\n" + "="*90)
print("✅ EXPERIMENT COMPLETE!")
print("="*90)
