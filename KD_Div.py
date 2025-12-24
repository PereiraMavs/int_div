import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import datasets  # Custom dataset loading module
import visualization  # Custom visualization module

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
BIN_WARMUP_EPOCHS = 30  # Number of warmup epochs for bin learner (Stage 1)
GEN_WARMUP_EPOCHS = 50  # Number of warmup epochs for generator (Stage 2)
WARMUP_SAMPLES = 500  # Number of random samples per warmup epoch

# Replay buffer parameters
REPLAY_BUFFER_SIZE = 1000  # Maximum samples to store
REPLAY_RATIO = 0.10  # 10% of batch will be replay samples

# Weights
LAMBDA_COV = 18.0    # Weight for Interaction Diversity (Entropy)
LAMBDA_HARD = 2.0   # Weight for Adversarial Hardness

# ==========================================
# 1. DATA PREPARATION
# ==========================================
print("\n📊 Loading Data...")

# ====== CHANGE DATASET HERE ======
DATASET_NAME = 'credit'  # Store dataset name for graph filenames
X_train, X_test, y_train, y_test, X_min, X_max, col_names, scaler, ht, NUM_CLASSES = datasets.load_dataset(DATASET_NAME)

# Auto-detect dimensions
DATA_DIM = X_train.shape[1]
NUM_CLASSES = len(np.unique(y_train))

print(f"📋 Dataset Info:")
print(f"   Dataset: {DATASET_NAME}")
print(f"   Samples: {X_train.shape[0]}")
print(f"   Features (DATA_DIM): {DATA_DIM}")
print(f"   Classes (NUM_CLASSES): {NUM_CLASSES}")

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
        inter_variance_per_feature = []
        
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
                inter_variance_per_feature.append(centroid_var.item())
                inter_variance_total += centroid_var.item()
            else:
                inter_variance_per_feature.append(0.0)
        
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
    Generator with clamping to ensure outputs stay within valid data range.
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
        intra_var, inter_var, _, _ = calculate_hard_variance(membership, teacher_probs)
        
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
# 4. WARMUP FUNCTIONS
# ==========================================

def warmup_bin_learner(bin_learner, teacher, random_gen, loss_bin_fn, epochs, batch_size, device):
    """
    Stage 1: Warm up bin learner with random data to initialize bin boundaries.
    
    Args:
        bin_learner: BinLearner model
        teacher: Pretrained teacher model
        random_gen: RandomDataGenerator instance
        loss_bin_fn: VarianceBasedBinLoss instance
        epochs: Number of warmup epochs
        batch_size: Batch size for training
        device: Device to run on
    
    Returns:
        opt_bin: Optimizer with warmed-up state
    """
    print("\n📊 Stage 1: Warming up Bin Learner with Random Data...")
    print(f"   Training bins for {epochs} epochs on random samples")
    print(f"   Goal: Initialize bin boundaries to cover full feature space")
    
    opt_bin = optim.Adam(bin_learner.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        bin_learner.train()
        
        # Generate random samples from feature space
        x_random = random_gen.generate(batch_size)
        
        # Get teacher predictions (detached, no teacher updates)
        teacher.eval()
        with torch.no_grad():
            teacher_probs = F.softmax(teacher(x_random), dim=1)
        
        # Compute membership and bin loss
        mship_warmup = bin_learner(x_random)
        loss_bin, loss_intra, loss_inter, _, _ = loss_bin_fn(mship_warmup, teacher_probs)
        
        opt_bin.zero_grad()
        loss_bin.backward()
        opt_bin.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs}: "
                  f"Bin Loss = {loss_bin.item():.4f}, "
                  f"Intra = {loss_intra.item():.4f}, "
                  f"Inter = {loss_inter.item():.4f}")
    
    print("✅ Bin Learner Warmup Complete - Bins initialized across feature space\n")
    return opt_bin


def warmup_student(student, teacher, random_gen, replay_buffer, X_test_tensor, y_test, 
                   epochs, warmup_samples, device):
    """
    Stage 2: Warm up student with random data and fill replay buffer.
    
    Args:
        student: StudentNet model
        teacher: Pretrained teacher model
        random_gen: RandomDataGenerator instance
        replay_buffer: ReplayBuffer instance to fill
        X_test_tensor: Test data for evaluation
        y_test: Test labels
        epochs: Number of warmup epochs
        warmup_samples: Samples per epoch
        device: Device to run on
    
    Returns:
        warmup_opt: Optimizer with warmed-up state
        warmup_train_acc: Training accuracy after warmup
        warmup_test_acc: Test accuracy after warmup
    """
    print("\n🔥 Warming up Student with Random Data...")
    print(f"   Generating {warmup_samples} random samples per epoch for {epochs} epochs")
    print(f"   Storing samples in replay buffer (max: {replay_buffer.max_size})")
    
    warmup_opt = optim.Adam(student.parameters(), lr=0.001)
    loss_kl_fn = nn.KLDivLoss(reduction='batchmean')
    
    for epoch in range(epochs):
        x_random = random_gen.generate(warmup_samples)
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
            print(f"   Warmup Epoch {epoch+1:3d}/{epochs}: Loss={loss.item():.4f}, "
                  f"Test Acc={test_acc*100:.2f}%, Buffer={len(replay_buffer)}")
    
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
    
    return warmup_opt, warmup_train_acc, warmup_test_acc


def warmup_generator(generator, bin_learner, loss_div_fn, epochs, latent_dim, batch_size, device):
    """
    Stage 3: Warm up generator for coverage with frozen bins.
    
    Args:
        generator: Generator model
        bin_learner: BinLearner model (will be frozen)
        loss_div_fn: InteractionDiversityLoss instance
        epochs: Number of warmup epochs
        latent_dim: Latent dimension for noise
        batch_size: Batch size for training
        device: Device to run on
    
    Returns:
        opt_gen: Optimizer with warmed-up state
    """
    print("\n🎨 Stage 3: Warming up Generator for Coverage...")
    print(f"   Training generator for {epochs} epochs to maximize diversity")
    print(f"   Goal: Learn to generate diverse samples covering all bin combinations")
    
    opt_gen = optim.Adam(generator.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        generator.train()
        
        # Generate synthetic samples
        z_gen = torch.randn(batch_size, latent_dim, device=device)
        x_gen = generator(z_gen)
        
        # Get bin assignments (frozen bins)
        with torch.no_grad():
            mship_gen = bin_learner(x_gen)
        
        # Maximize diversity (minimize entropy loss)
        l_div = loss_div_fn(mship_gen)
        
        opt_gen.zero_grad()
        l_div.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_gen.step()
        
        if (epoch + 1) % 10 == 0:
            # Compute coverage on generated samples
            with torch.no_grad():
                z_cov = torch.randn(1000, latent_dim, device=device)
                x_cov = generator(z_cov)
                mship_cov = bin_learner(x_cov)
                coverage_ratio, _ = compute_coverage(mship_cov, threshold=5)
            
            print(f"   Epoch {epoch+1}/{epochs}: "
                  f"Diversity Loss = {l_div.item():.2f}, "
                  f"Coverage = {coverage_ratio*100:.1f}%")
    
    print("✅ Generator Warmup Complete - Generator learned to produce diverse samples\n")
    return opt_gen


# ==========================================
# 5. TRAINING PIPELINE
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
X_mean = torch.tensor(X_train.mean(axis=0), dtype=torch.float32).to(DEVICE)
bin_learner = BinLearner(DATA_DIM, NUM_BINS, X_min, X_max).to(DEVICE)
generator = Generator(LATENT_DIM, DATA_DIM, X_mean, X_min, X_max).to(DEVICE)
student = StudentNet(DATA_DIM, NUM_CLASSES).to(DEVICE)

# Instantiate loss functions
loss_bin_fn = VarianceBasedBinLoss()
loss_div_fn = InteractionDiversityLoss(t_way=2)

# C. Run Warmup Stages
opt_bin_warmup = optim.Adam(bin_learner.parameters(), lr=0.001)
'''
opt_bin_warmup = warmup_bin_learner(
    bin_learner, teacher, random_gen, loss_bin_fn, 
    BIN_WARMUP_EPOCHS, BATCH_SIZE, DEVICE
)
'''
#opt_stu_warmup = optim.Adam(student.parameters(), lr=0.001)

opt_stu_warmup, warmup_train_acc, warmup_test_acc = warmup_student(
    student, teacher, random_gen, replay_buffer, 
    X_test_tensor, y_test, WARMUP_EPOCHS, WARMUP_SAMPLES, DEVICE
)

opt_gen_warmup = optim.Adam(generator.parameters(), lr=0.001)
'''
opt_gen_warmup = warmup_generator(
    generator, bin_learner, loss_div_fn, 
    GEN_WARMUP_EPOCHS, LATENT_DIM, BATCH_SIZE, DEVICE
)
'''

# D. Losses & Optimizers for Main Training
loss_kl_fn = nn.KLDivLoss(reduction='batchmean')

# Reuse warmed-up optimizers (preserves Adam momentum)
opt_bin = opt_bin_warmup  # Reuse bin optimizer
for param_group in opt_bin.param_groups:
    param_group['lr'] = 0.01  # Increase LR for main training

opt_gen = opt_gen_warmup  # Reuse generator optimizer
opt_stu = opt_stu_warmup  # Reuse student optimizer

# Learning rate schedulers
scheduler_bin = optim.lr_scheduler.CosineAnnealingLR(opt_bin, T_max=EPOCHS)
scheduler_gen = optim.lr_scheduler.CosineAnnealingLR(opt_gen, T_max=EPOCHS)
scheduler_stu = optim.lr_scheduler.CosineAnnealingLR(opt_stu, T_max=EPOCHS)

# History tracking for all losses
history = {
    'bin_total': [],       # Total bin loss
    'bin_intra': [],       # Intra-bin loss component
    'bin_inter': [],       # Inter-bin loss component
    'diversity': [],       # Diversity loss
    'hardness': [],        # Hardness loss
    'student': [],         # Student KD loss
    'test_acc': [],
    'agreement': [],
    'coverage': [],
    'variance_maps': [],   # Per-feature, per-bin intra-variance (F, K)
    'inter_variance_per_feature': []  # Per-feature inter-bin variance (F,)
}

print("\n⚔️ Starting Main Training...")
print(f"   Using replay buffer: {REPLAY_RATIO*100:.0f}% of batch from warmup samples")
print(f"{'Epoch':<6} | {'Bin Total':<10} | {'Div':<8} | {'Hard':<8} | {'Stu':<8} | {'Test Acc':<10} | {'Agreement':<10} | {'Coverage':<10}")
print("-" * 120)

best_student_state = None
best_agreement_score = 0.0

for epoch in range(1, EPOCHS + 1):
    
    # Anneal Temperature: 0.5 -> 5.0
    bin_learner.temperature = 2.0 - (1.7 * (epoch / EPOCHS))  # 5.0 -> 2.0 (hard to soft boundaries)
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
    with torch.no_grad():
        t_probs_2 = F.softmax(teacher(x_gen_2), dim=1)
        s_log_probs = F.log_softmax(student(x_gen_2), dim=1)
    
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
    
    # Store losses in history
    history['bin_total'].append(l_bin.item())
    history['bin_intra'].append(l_intra.item())
    history['bin_inter'].append(l_inter.item())
    history['diversity'].append(l_div.item())
    history['hardness'].append(l_hard.item())
    history['student'].append(l_stu.item())
    
    # Compute and store detailed variance metrics
    with torch.no_grad():
        z_var = torch.randn(500, LATENT_DIM).to(DEVICE)
        x_var = generator(z_var)
        t_probs_var = F.softmax(teacher(x_var), dim=1)
        mship_var = bin_learner(x_var)
        _, _, variance_map, inter_var_per_feature = calculate_hard_variance(mship_var, t_probs_var)
        history['variance_maps'].append(variance_map.cpu())
        history['inter_variance_per_feature'].append(inter_var_per_feature.cpu())
    
    if epoch % 20 == 0:
        student.eval()
        with torch.no_grad():
            s_test_preds = torch.argmax(student(X_test_tensor), dim=1)
            t_test_preds = torch.argmax(teacher(X_test_tensor), dim=1)
            test_acc = accuracy_score(y_test, s_test_preds.cpu().numpy())
            agreement = (t_test_preds == s_test_preds).float().mean().item()
            
            # Compute coverage certification
            z_cov = torch.randn(1000, LATENT_DIM).to(DEVICE)
            x_cov = generator(z_cov)
            mship_cov = bin_learner(x_cov)
            coverage_ratio, _ = compute_coverage(mship_cov, threshold=5)
            
        history['test_acc'].append(test_acc)
        history['agreement'].append(agreement)
        history['coverage'].append(coverage_ratio)
        
        student.train()
        
        print(f"{epoch:<6} | {l_bin.item():<10.5f} | {l_div.item():<8.1f} | {l_hard.item():<8.3f} | {l_stu.item():<8.3f} | {test_acc*100:<10.1f}% | {agreement*100:<10.1f}% | {coverage_ratio*100:<10.1f}%")
        
        # Print detailed variance report
        print(f"\n   📊 Per-Feature Bin Variance Report (Epoch {epoch}):")
        print(f"   {'Feature':<10} | {'Inter-Var':<12} | Intra-Variance per Bin (K={NUM_BINS})")
        print("   " + "-" * 80)
        for f in range(DATA_DIM):
            intra_vals = [f"{variance_map[f, k].item():.4f}" for k in range(NUM_BINS)]
            inter_val = inter_var_per_feature[f].item()
            print(f"   Feature {f:<3} | {inter_val:<12.4f} | {' '.join(intra_vals)}")
        print()
        
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
# 6. VISUALIZATION - Using Modular Functions
# ==========================================
print("\n📈 Generating visualizations and reports...")

# Generate loss curves
visualization.plot_loss_curves(history, DATASET_NAME, EPOCHS)

# Generate performance metrics
visualization.plot_performance_metrics(history, DATASET_NAME, teacher_test_acc, warmup_test_acc, EPOCHS)

# Generate combined dashboard
visualization.plot_combined_dashboard(history, DATASET_NAME, teacher_test_acc, warmup_test_acc, EPOCHS)

# Generate text report
visualization.generate_loss_report(history, DATASET_NAME, EPOCHS)

# Generate variance report
visualization.generate_variance_report(history, DATASET_NAME, NUM_BINS, DATA_DIM)

# Save training history
visualization.save_history(history, DATASET_NAME)

# ==========================================
# 7. SAVE MODELS
# ==========================================
print("\n💾 Saving models...")

torch.save({
    'model_state_dict': teacher.state_dict(),
    'train_acc': teacher_train_acc,
    'test_acc': teacher_test_acc,
    'architecture': f'{DATA_DIM} -> 128 -> 64 -> {NUM_CLASSES}'
}, f'teacher_model_{DATASET_NAME}.pt')

torch.save({
    'model_state_dict': student.state_dict(),
    'train_acc': student_train_acc,
    'test_acc': student_test_acc,
    'agreement_train': train_agreement,
    'agreement_test': test_agreement,
    'coverage': final_coverage,
    'architecture': f'{DATA_DIM} -> 32 -> {NUM_CLASSES}',
    'epoch': EPOCHS
}, f'student_final_{DATASET_NAME}.pt')

if best_student_state is not None:
    torch.save(best_student_state, f'student_best_{DATASET_NAME}.pt')
    print(f"✓ Best student saved from epoch {best_student_state['epoch']} "
          f"(Agreement: {best_student_state['agreement']*100:.2f}%, "
          f"Test Acc: {best_student_state['test_acc']*100:.2f}%, "
          f"Coverage: {best_student_state['coverage']*100:.2f}%)")

torch.save({
    'generator': generator.state_dict(),
    'bin_learner': bin_learner.state_dict(),
}, f'auxiliary_models_{DATASET_NAME}.pt')

print(f"✓ All models saved successfully")

print("\n" + "="*90)
print("✅ EXPERIMENT COMPLETE!")
print("="*90)
