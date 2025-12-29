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
BATCH_SIZE = 256
EPOCHS = 400
LATENT_DIM = 32     # Generator Noise Dimension
NUM_BINS = 8        # Bins per feature (K)

# Warmup parameters
WARMUP_EPOCHS = 30  # Number of warmup epochs for student
WARMUP_SAMPLES = 500  # Number of random samples per warmup epoch

# Replay buffer parameters
REPLAY_BUFFER_SIZE = 1000  # Maximum samples to store
REPLAY_RATIO = 0.10  # 10% of batch will be replay samples

# Weights
LAMBDA_COV = 10.0    # Weight for Interaction Diversity (Entropy)
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
        
        return 2.0 * loss_intra + (3.0 * loss_inter), loss_intra, loss_inter, intra_var, inter_var

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

def update_cumulative_coverage(membership, cumulative_tracker):
    """
    Update cumulative coverage tracker with new samples.
    
    Args:
        membership: Tensor of shape (N, F, K) - soft membership values
        cumulative_tracker: Boolean tensor (F, F, K, K) tracking visited bin pairs
    
    Returns:
        cumulative_coverage_ratio: Fraction of all possible bin pairs visited so far
    """
    N, F, K = membership.shape
    
    with torch.no_grad():
        hard_assignments = membership.argmax(dim=2)  # (N, F)
        
        # Update tracker for each sample
        for n in range(N):
            for i in range(F):
                for j in range(i + 1, F):
                    bin_i = hard_assignments[n, i].item()
                    bin_j = hard_assignments[n, j].item()
                    cumulative_tracker[i, j, bin_i, bin_j] = True
                    cumulative_tracker[j, i, bin_j, bin_i] = True  # Symmetric
        
        # Count covered pairs
        num_feature_pairs = (F * (F - 1)) // 2
        total_bin_pairs = num_feature_pairs * K * K
        covered_bin_pairs = 0
        
        for i in range(F):
            for j in range(i + 1, F):
                covered_bin_pairs += cumulative_tracker[i, j].sum().item()
        
        cumulative_coverage_ratio = covered_bin_pairs / total_bin_pairs if total_bin_pairs > 0 else 0.0
        
    return cumulative_coverage_ratio

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
X_mean = torch.tensor(X_train.mean(axis=0), dtype=torch.float32).to(DEVICE)
bin_learner = BinLearner(DATA_DIM, NUM_BINS, X_min, X_max).to(DEVICE)
generator = Generator(LATENT_DIM, DATA_DIM, X_mean, X_min, X_max).to(DEVICE)
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

# ==========================================
# C. PHASE 1: BIN LEARNER TRAINING (Stabilize Bin Boundaries)
# ==========================================
print("\n📍 PHASE 1: Training Bin Learner on Random Data...")
print(f"   Goal: Learn stable bin boundaries before distillation")

BIN_LEARNER_EPOCHS = 400
loss_bin_fn = VarianceBasedBinLoss()
opt_bin = optim.Adam(bin_learner.parameters(), lr=0.01)
scheduler_bin = optim.lr_scheduler.CosineAnnealingLR(opt_bin, T_max=BIN_LEARNER_EPOCHS)

# Track bin boundaries AND variance over epochs
boundary_history = []
variance_history = []  # NEW: Track variance evolution
boundary_checkpoints = list(range(0, BIN_LEARNER_EPOCHS + 1, 20))  # Every 20 epochs

print(f"{'Epoch':<6} | {'Bin Total':<10} | {'Intra':<10} | {'Inter':<10} | {'Intra Var':<12} | {'Inter Var':<12}")
print("-" * 80)

# Capture initial boundaries and variance (epoch 0)
with torch.no_grad():
    initial_boundaries = bin_learner.get_boundaries()
    boundary_history.append({
        'epoch': 0,
        'boundaries': initial_boundaries.cpu().clone()
    })
    
    # Compute initial variance on real training data
    batch_size_var = min(2000, X_train_tensor.size(0))
    indices = torch.randperm(X_train_tensor.size(0))[:batch_size_var]
    X_sample = X_train_tensor[indices]
    teacher_probs_sample = F.softmax(teacher(X_sample), dim=1)
    mship_train = bin_learner(X_sample)
    intra_var_0, inter_var_0, variance_map_0, inter_variance_per_feature_0 = calculate_hard_variance(
        mship_train, teacher_probs_sample
    )
    variance_history.append({
        'epoch': 0,
        'intra_var': intra_var_0,
        'inter_var': inter_var_0,
        'variance_map': variance_map_0.cpu().clone(),
        'inter_variance_per_feature': inter_variance_per_feature_0.cpu().clone()
    })

for epoch in range(1, BIN_LEARNER_EPOCHS + 1):
    # Anneal temperature: start hard, end soft
    bin_learner.temperature = 1.0 - (0.95 * (epoch / BIN_LEARNER_EPOCHS))  # 1.0 -> 0.1
    
    # Generate random data
    x_random = random_gen.generate(BATCH_SIZE)
    
    with torch.no_grad():
        t_probs = F.softmax(teacher(x_random), dim=1)
    
    opt_bin.zero_grad()
    mship = bin_learner(x_random)
    l_bin, l_intra, l_inter, intra_var, inter_var = loss_bin_fn(mship, t_probs)
    
    if not torch.isnan(l_bin):
        l_bin.backward()
        torch.nn.utils.clip_grad_norm_(bin_learner.parameters(), 1.0)
        opt_bin.step()
    
    scheduler_bin.step()
    
    # Capture boundaries AND variance at checkpoints
    if epoch in boundary_checkpoints:
        with torch.no_grad():
            # Boundaries
            boundaries = bin_learner.get_boundaries()
            boundary_history.append({
                'epoch': epoch,
                'boundaries': boundaries.cpu().clone()
            })
            
            # Variance on real training data
            batch_size_var = min(2000, X_train_tensor.size(0))
            indices = torch.randperm(X_train_tensor.size(0))[:batch_size_var]
            X_sample = X_train_tensor[indices]
            teacher_probs_sample = F.softmax(teacher(X_sample), dim=1)
            mship_train = bin_learner(X_sample)
            intra_var_ep, inter_var_ep, variance_map_ep, inter_variance_per_feature_ep = calculate_hard_variance(
                mship_train, teacher_probs_sample
            )
            variance_history.append({
                'epoch': epoch,
                'intra_var': intra_var_ep,
                'inter_var': inter_var_ep,
                'variance_map': variance_map_ep.cpu().clone(),
                'inter_variance_per_feature': inter_variance_per_feature_ep.cpu().clone()
            })
    
    if epoch % 20 == 0:
        print(f"{epoch:<6} | {l_bin.item():<10.5f} | {l_intra.item():<10.5f} | {l_inter.item():<10.5f} | {intra_var:<12.5f} | {inter_var:<12.5f}")

print(f"✅ Bin Learner Training Complete!")
print(f"   Boundaries are now frozen for distillation phase")

# ==========================================
# C.1 VISUALIZE BIN BOUNDARY EVOLUTION
# ==========================================
print("\n📊 Generating bin boundary evolution report...")

import os
os.makedirs('reports', exist_ok=True)

with open(f'reports/bin_boundaries_{DATASET_NAME}.txt', 'w') as f:
    f.write("=" * 100 + "\n")
    f.write("BIN BOUNDARY EVOLUTION REPORT\n")
    f.write("=" * 100 + "\n\n")
    f.write(f"Dataset: {DATASET_NAME}\n")
    f.write(f"Features: {DATA_DIM}\n")
    f.write(f"Bins per feature: {NUM_BINS}\n")
    f.write(f"Training epochs: {BIN_LEARNER_EPOCHS}\n")
    f.write(f"Checkpoints captured: {len(boundary_history)}\n\n")
    
    # For each feature, show boundary evolution
    for feature_idx in range(DATA_DIM):  # Show all features
        f.write("=" * 100 + "\n")
        f.write(f"FEATURE {feature_idx} - Boundary Evolution\n")
        f.write("=" * 100 + "\n\n")
        
        # Header: Epochs as columns
        f.write(f"{'Boundary':<12} | ")
        for entry in boundary_history:
            f.write(f"Epoch {entry['epoch']:<6} | ")
        f.write("\n")
        f.write("-" * 100 + "\n")
        
        # Each boundary as a row
        for bin_idx in range(NUM_BINS + 1):
            f.write(f"Bound {bin_idx:<6} | ")
            for entry in boundary_history:
                boundaries = entry['boundaries']
                value = boundaries[0, feature_idx, bin_idx].item()
                f.write(f"{value:>12.6f} | ")
            f.write("\n")
        
        # Calculate boundary movement (max change from initial)
        f.write("\n")
        f.write(f"{'Boundary':<12} | {'Initial':<12} | {'Final':<12} | {'Change':<12} | {'% Change':<12}\n")
        f.write("-" * 80 + "\n")
        
        initial_bounds = boundary_history[0]['boundaries'][0, feature_idx, :]
        final_bounds = boundary_history[-1]['boundaries'][0, feature_idx, :]
        
        for bin_idx in range(NUM_BINS + 1):
            initial = initial_bounds[bin_idx].item()
            final = final_bounds[bin_idx].item()
            change = final - initial
            pct_change = (change / (abs(initial) + 1e-8)) * 100
            
            f.write(f"Bound {bin_idx:<6} | {initial:>12.6f} | {final:>12.6f} | {change:>12.6f} | {pct_change:>11.2f}%\n")
        
        f.write("\n\n")
    
    # Summary statistics
    f.write("=" * 100 + "\n")
    f.write("OVERALL BOUNDARY STABILITY ANALYSIS\n")
    f.write("=" * 100 + "\n\n")
    
    total_movement = 0.0
    max_movement = 0.0
    
    for feature_idx in range(DATA_DIM):
        initial_bounds = boundary_history[0]['boundaries'][0, feature_idx, :]
        final_bounds = boundary_history[-1]['boundaries'][0, feature_idx, :]
        
        movement = torch.abs(final_bounds - initial_bounds).sum().item()
        max_move = torch.abs(final_bounds - initial_bounds).max().item()
        
        total_movement += movement
        max_movement = max(max_movement, max_move)
    
    avg_movement = total_movement / DATA_DIM
    
    f.write(f"Average boundary movement per feature: {avg_movement:.6f}\n")
    f.write(f"Maximum single boundary movement: {max_movement:.6f}\n")
    f.write(f"Total boundary updates across all features: {total_movement:.6f}\n")
    
    # Convergence check (compare last two checkpoints)
    if len(boundary_history) >= 2:
        penultimate_bounds = boundary_history[-2]['boundaries']
        final_bounds = boundary_history[-1]['boundaries']
        convergence = torch.abs(final_bounds - penultimate_bounds).mean().item()
        
        f.write(f"\nConvergence (change in last checkpoint): {convergence:.8f}\n")
        if convergence < 0.001:
            f.write("✅ Boundaries have CONVERGED (stable)\n")
        elif convergence < 0.01:
            f.write("⚠️  Boundaries are STABILIZING but not fully converged\n")
        else:
            f.write("❌ Boundaries are still MOVING significantly\n")

print(f"✅ Bin boundary report saved to: reports/bin_boundaries_{DATASET_NAME}.txt")

# ==========================================
# C.2 VARIANCE EVOLUTION REPORT
# ==========================================
print("\n📊 Generating variance evolution report...")

with open(f'reports/variance_evolution_{DATASET_NAME}.txt', 'w') as f:
    f.write("=" * 120 + "\n")
    f.write("BIN VARIANCE EVOLUTION OVER TRAINING\n")
    f.write("=" * 120 + "\n\n")
    f.write(f"Dataset: {DATASET_NAME}\n")
    f.write(f"Features: {DATA_DIM}\n")
    f.write(f"Bins per feature: {NUM_BINS}\n")
    f.write(f"Training epochs: {BIN_LEARNER_EPOCHS}\n")
    f.write(f"Checkpoints captured: {len(variance_history)}\n")
    f.write(f"Evaluated on: {batch_size_var} real training samples\n\n")
    
    # Overall variance evolution
    f.write("=" * 120 + "\n")
    f.write("OVERALL VARIANCE EVOLUTION\n")
    f.write("=" * 120 + "\n\n")
    
    f.write(f"{'Epoch':<8} | {'Overall Intra-Var':<18} | {'Overall Inter-Var':<18} | {'Quality Ratio':<15} | {'Trend':<15}\n")
    f.write("-" * 100 + "\n")
    
    for i, entry in enumerate(variance_history):
        epoch = entry['epoch']
        intra = entry['intra_var']
        inter = entry['inter_var']
        quality_ratio = inter / (intra + 1e-8)
        
        # Trend indicator
        if i > 0:
            prev_quality = variance_history[i-1]['inter_var'] / (variance_history[i-1]['intra_var'] + 1e-8)
            if quality_ratio > prev_quality * 1.05:
                trend = "↑ Improving"
            elif quality_ratio < prev_quality * 0.95:
                trend = "↓ Degrading"
            else:
                trend = "→ Stable"
        else:
            trend = "Initial"
        
        f.write(f"{epoch:<8} | {intra:<18.6f} | {inter:<18.6f} | {quality_ratio:<15.2f} | {trend:<15}\n")
    
    # Summary
    initial_quality = variance_history[0]['inter_var'] / (variance_history[0]['intra_var'] + 1e-8)
    final_quality = variance_history[-1]['inter_var'] / (variance_history[-1]['intra_var'] + 1e-8)
    improvement = ((final_quality - initial_quality) / initial_quality) * 100
    
    f.write("\n")
    f.write(f"Initial Quality Ratio: {initial_quality:.2f}\n")
    f.write(f"Final Quality Ratio:   {final_quality:.2f}\n")
    f.write(f"Improvement:           {improvement:+.1f}%\n\n")
    
    # Per-feature variance evolution
    for feature_idx in range(DATA_DIM):
        f.write("=" * 120 + "\n")
        f.write(f"FEATURE {feature_idx} - VARIANCE EVOLUTION\n")
        f.write("=" * 120 + "\n\n")
        
        # Inter-bin variance evolution for this feature
        f.write("Inter-Bin Variance (higher = better separation)\n")
        f.write(f"{'Epoch':<8} | ")
        for entry in variance_history:
            f.write(f"{entry['epoch']:<10} | ")
        f.write("\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Value':<8} | ")
        for entry in variance_history:
            inter_var_feat = entry['inter_variance_per_feature'][feature_idx].item()
            f.write(f"{inter_var_feat:<10.6f} | ")
        f.write("\n\n")
        
        # Intra-bin variance evolution for each bin
        f.write("Intra-Bin Variance per Bin (lower = purer bins)\n")
        f.write(f"{'Bin':<8} | ")
        for entry in variance_history:
            f.write(f"Epoch {entry['epoch']:<4} | ")
        f.write("\n")
        f.write("-" * 120 + "\n")
        
        for bin_idx in range(NUM_BINS):
            f.write(f"Bin {bin_idx:<4} | ")
            for entry in variance_history:
                intra_var_bin = entry['variance_map'][feature_idx, bin_idx].item()
                f.write(f"{intra_var_bin:<10.6f} | ")
            f.write("\n")
        
        f.write("\n")
        
        # Feature summary
        initial_inter = variance_history[0]['inter_variance_per_feature'][feature_idx].item()
        final_inter = variance_history[-1]['inter_variance_per_feature'][feature_idx].item()
        inter_change = final_inter - initial_inter
        
        initial_intra_avg = variance_history[0]['variance_map'][feature_idx, :].mean().item()
        final_intra_avg = variance_history[-1]['variance_map'][feature_idx, :].mean().item()
        intra_change = final_intra_avg - initial_intra_avg
        
        f.write(f"Feature {feature_idx} Summary:\n")
        f.write(f"  Inter-Bin Variance:  {initial_inter:.6f} → {final_inter:.6f} (change: {inter_change:+.6f})\n")
        f.write(f"  Avg Intra-Bin Var:   {initial_intra_avg:.6f} → {final_intra_avg:.6f} (change: {intra_change:+.6f})\n")
        
        if inter_change > 0.01:
            f.write(f"  ✅ Inter-bin separation IMPROVED\n")
        elif inter_change < -0.01:
            f.write(f"  ❌ Inter-bin separation DEGRADED\n")
        else:
            f.write(f"  → Inter-bin separation STABLE\n")
        
        if intra_change < -0.01:
            f.write(f"  ✅ Bin purity IMPROVED (variance decreased)\n")
        elif intra_change > 0.01:
            f.write(f"  ❌ Bin purity DEGRADED (variance increased)\n")
        else:
            f.write(f"  → Bin purity STABLE\n")
        
        f.write("\n\n")
    
    # Overall assessment
    f.write("=" * 120 + "\n")
    f.write("TRAINING PROGRESS ASSESSMENT\n")
    f.write("=" * 120 + "\n\n")
    
    improved_features = 0
    degraded_features = 0
    stable_features = 0
    
    for feature_idx in range(DATA_DIM):
        initial_quality_feat = variance_history[0]['inter_variance_per_feature'][feature_idx].item() / \
                               (variance_history[0]['variance_map'][feature_idx, :].mean().item() + 1e-8)
        final_quality_feat = variance_history[-1]['inter_variance_per_feature'][feature_idx].item() / \
                             (variance_history[-1]['variance_map'][feature_idx, :].mean().item() + 1e-8)
        
        if final_quality_feat > initial_quality_feat * 1.1:
            improved_features += 1
        elif final_quality_feat < initial_quality_feat * 0.9:
            degraded_features += 1
        else:
            stable_features += 1
    
    f.write(f"Features Improved:  {improved_features} / {DATA_DIM} ({improved_features/DATA_DIM*100:.1f}%)\n")
    f.write(f"Features Degraded:  {degraded_features} / {DATA_DIM} ({degraded_features/DATA_DIM*100:.1f}%)\n")
    f.write(f"Features Stable:    {stable_features} / {DATA_DIM} ({stable_features/DATA_DIM*100:.1f}%)\n\n")
    
    if improvement > 20:
        f.write("✅ VERDICT: Significant improvement in bin quality during training\n")
    elif improvement > 5:
        f.write("✓ VERDICT: Moderate improvement in bin quality\n")
    elif improvement > -5:
        f.write("→ VERDICT: Bin quality remained stable\n")
    else:
        f.write("❌ VERDICT: Bin quality degraded during training\n")

print(f"✅ Variance evolution report saved to: reports/variance_evolution_{DATASET_NAME}.txt")

# ==========================================
# D. PHASE 2: KNOWLEDGE DISTILLATION (Generator + Student)
# ==========================================
print("\n🎓 PHASE 2: Knowledge Distillation with Frozen Bins...")
print(f"   Generator: Hardness + Diversity Loss")
print(f"   Student: KL Divergence Loss")
print(f"   Bin Learner: FROZEN")

# Freeze bin learner
for param in bin_learner.parameters():
    param.requires_grad = False

# Set temperature for Phase 2 (generator training)
bin_learner.temperature = 0.2  # Hard boundaries for discrete coverage

# Losses & Optimizers
loss_div_fn = InteractionDiversityLoss(t_way=2)
loss_kl_fn = nn.KLDivLoss(reduction='batchmean')

opt_gen = optim.Adam(generator.parameters(), lr=0.001)
opt_stu = optim.Adam(student.parameters(), lr=0.001)

scheduler_gen = optim.lr_scheduler.CosineAnnealingLR(opt_gen, T_max=EPOCHS)
scheduler_stu = optim.lr_scheduler.CosineAnnealingLR(opt_stu, T_max=EPOCHS)

# Track variance on generator data during Phase 2
generator_variance_history = []
variance_checkpoints = list(range(0, EPOCHS + 1, 25))  # Every 25 epochs

# Initialize cumulative coverage tracker
print(f"   Initializing cumulative coverage tracker...")
num_feature_pairs = (DATA_DIM * (DATA_DIM - 1)) // 2
total_possible_pairs = num_feature_pairs * NUM_BINS * NUM_BINS
cumulative_coverage_tracker = torch.zeros(DATA_DIM, DATA_DIM, NUM_BINS, NUM_BINS, dtype=torch.bool, device=DEVICE)
print(f"   Total possible bin pairs to explore: {total_possible_pairs}")

def compute_coverage_bonus(membership, cumulative_tracker):
    """
    Reward generator for hitting unexplored bin pairs.
    Returns negative loss (reward) when visiting new bins.
    """
    N, F, K = membership.shape
    
    with torch.no_grad():
        hard_assignments = membership.argmax(dim=2)  # (N, F)
    
    novelty_score = 0.0
    total_samples = 0
    
    for n in range(N):
        sample_novelty = 0.0
        for i in range(F):
            for j in range(i + 1, F):
                bin_i = hard_assignments[n, i].item()
                bin_j = hard_assignments[n, j].item()
                
                # Reward if this bin pair hasn't been visited
                if not cumulative_tracker[i, j, bin_i, bin_j]:
                    sample_novelty += 1.0
        
        novelty_score += sample_novelty
        total_samples += 1
    
    # Average novelty per sample
    avg_novelty = novelty_score / (total_samples * ((F * (F - 1)) // 2))
    
    # Return negative (reward for novelty)
    return -avg_novelty

# History tracking
history = {
    'diversity': [],       # Diversity loss
    'hardness': [],        # Hardness loss
    'student': [],         # Student KD loss
    'test_acc': [],
    'agreement': [],
    'coverage_snapshot': [],      # Snapshot coverage (current 1000 samples)
    'coverage_cumulative': []     # Cumulative coverage (all samples seen)
}

print(f"\n{'Epoch':<6} | {'Div':<8} | {'Hard':<8} | {'Stu':<8} | {'Test Acc':<10} | {'Agreement':<10} | {'Cov (Snap)':<12} | {'Cov (Cumul)':<12}")
print("-" * 120)

best_student_state = None
best_agreement_score = 0.0

for epoch in range(1, EPOCHS + 1):
    z = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
    
    # ==========================
    # GENERATOR UPDATE (Diversity + Hardness + Coverage Bonus)
    # ==========================
    opt_gen.zero_grad()
    x_gen = generator(z)

    # 1. Interaction Diversity (Maximize Entropy)
    mship = bin_learner(x_gen)
    l_div = loss_div_fn(mship)

    # 2. Hardness (Maximize Student Error)
    with torch.no_grad():
        t_probs = F.softmax(teacher(x_gen), dim=1)
    s_log_probs = F.log_softmax(student(x_gen), dim=1)
    l_hard = -1.0 * F.kl_div(s_log_probs, t_probs, reduction='batchmean')

    # 3. Coverage Bonus (Reward unexplored bins)
    l_coverage = compute_coverage_bonus(mship, cumulative_coverage_tracker)

    # Combine losses
    l_gen = (LAMBDA_COV * l_div) + (LAMBDA_HARD * l_hard) + (8.0 * l_coverage)

    if not torch.isnan(l_gen):
        l_gen.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_gen.step()
    
    # ==========================
    # STUDENT UPDATE (KD + Replay Buffer)
    # ==========================
    opt_stu.zero_grad()
    
    # Generate new batch for student
    z_stu = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
    x_gen_stu = generator(z_stu).detach()
    
    # Mix with replay buffer samples
    replay_batch_size = int(BATCH_SIZE * REPLAY_RATIO)
    x_replay = replay_buffer.sample(replay_batch_size)
    
    if x_replay is not None:
        x_mixed = torch.cat([x_gen_stu, x_replay], dim=0)
    else:
        x_mixed = x_gen_stu
    
    with torch.no_grad():
        t_probs_stu = F.softmax(teacher(x_mixed), dim=1)
    s_log_probs_stu = F.log_softmax(student(x_mixed), dim=1)
    
    l_stu = loss_kl_fn(s_log_probs_stu, t_probs_stu)
    
    if not torch.isnan(l_stu):
        l_stu.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt_stu.step()
    
    # ==========================
    # UPDATE CUMULATIVE COVERAGE (Every Epoch)
    # ==========================
    with torch.no_grad():
        # Use the batch we just generated for student
        mship_batch = bin_learner(x_gen_stu)
        cumulative_cov = update_cumulative_coverage(mship_batch, cumulative_coverage_tracker)
    
    # ==========================
    # VARIANCE TRACKING ON GENERATOR DATA (Every 25 epochs)
    # ==========================
    if epoch in variance_checkpoints:
        with torch.no_grad():
            # Generate samples for variance calculation
            z_var = torch.randn(2000, LATENT_DIM).to(DEVICE)
            x_gen_var = generator(z_var)
            teacher_probs_var = F.softmax(teacher(x_gen_var), dim=1)
            mship_var = bin_learner(x_gen_var)
            
            intra_var_ep, inter_var_ep, variance_map_ep, inter_variance_per_feature_ep = calculate_hard_variance(
                mship_var, teacher_probs_var
            )
            
            generator_variance_history.append({
                'epoch': epoch,
                'intra_var': intra_var_ep,
                'inter_var': inter_var_ep,
                'variance_map': variance_map_ep.cpu().clone(),
                'inter_variance_per_feature': inter_variance_per_feature_ep.cpu().clone()
            })
    
    # Step schedulers
    scheduler_gen.step()
    scheduler_stu.step()
    
    # Store losses in history
    history['diversity'].append(l_div.item())
    history['hardness'].append(l_hard.item())
    history['student'].append(l_stu.item())
    history['coverage_cumulative'].append(cumulative_cov)
    
    if epoch % 20 == 0:
        student.eval()
        with torch.no_grad():
            s_test_preds = torch.argmax(student(X_test_tensor), dim=1)
            t_test_preds = torch.argmax(teacher(X_test_tensor), dim=1)
            test_acc = accuracy_score(y_test, s_test_preds.cpu().numpy())
            agreement = (t_test_preds == s_test_preds).float().mean().item()
            
            # Compute SNAPSHOT coverage (fresh 1000 samples)
            z_cov = torch.randn(1000, LATENT_DIM).to(DEVICE)
            x_cov = generator(z_cov)
            mship_cov = bin_learner(x_cov)
            snapshot_cov, _ = compute_coverage(mship_cov, threshold=1)
        
        history['test_acc'].append(test_acc)
        history['agreement'].append(agreement)
        history['coverage_snapshot'].append(snapshot_cov)
        
        student.train()
        
        print(f"{epoch:<6} | {l_div.item():<8.1f} | {l_hard.item():<8.3f} | {l_stu.item():<8.3f} | {test_acc*100:<10.1f}% | {agreement*100:<10.1f}% | {snapshot_cov*100:<12.1f}% | {cumulative_cov*100:<12.1f}%")
        
        # Track best student model
        if agreement > best_agreement_score:
            best_agreement_score = agreement
            best_student_state = {
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'test_acc': test_acc,
                'agreement': agreement,
                'coverage_snapshot': snapshot_cov,
                'coverage_cumulative': cumulative_cov
            }

print(f"\n✅ Phase 2 Complete!")
print(f"   Best Student (Epoch {best_student_state['epoch']}):")
print(f"   - Test Accuracy: {best_student_state['test_acc']*100:.2f}%")
print(f"   - Agreement with Teacher: {best_student_state['agreement']*100:.2f}%")
print(f"   - Snapshot Coverage: {best_student_state['coverage_snapshot']*100:.2f}%")
print(f"   - Cumulative Coverage: {best_student_state['coverage_cumulative']*100:.2f}%")
print(f"   - Total bin pairs explored: {int(best_student_state['coverage_cumulative'] * total_possible_pairs)} / {total_possible_pairs}")

# Save models and history
torch.save({
    'generator': generator.state_dict(),
    'student': student.state_dict(),
    'bin_learner': bin_learner.state_dict(),
    'best_student': best_student_state,
    'history': history
}, f'training_history_{DATASET_NAME}.pt')

print(f"\n💾 Models and history saved to: training_history_{DATASET_NAME}.pt")

