import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
import sys
import argparse
from datetime import datetime
import datasets  # Custom dataset loading module
import visualization  # Custom visualization module

# ==========================================
# 0. CONFIGURATION & HYPERPARAMETERS
# ==========================================

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Knowledge Distillation with Adversarial Training')
parser.add_argument('--teacher', type=str, default='neural', choices=['neural', 'xgboost', 'randomforest'],
                    help='Teacher model type (default: neural)')
parser.add_argument('--dataset', type=str, default='adult',
                    help='Dataset name (default: adult)')
args = parser.parse_args()

# Set configuration from arguments
TEACHER_TYPE = args.teacher
DATASET_NAME = args.dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Running on: {DEVICE}")
print(f"📚 Teacher Type: {TEACHER_TYPE.upper()}")
print(f"📊 Dataset: {DATASET_NAME}")

# Create output directory structure
BASE_REPORTS_DIR = 'reports'
TEACHER_REPORTS_DIR = os.path.join(BASE_REPORTS_DIR, TEACHER_TYPE)
os.makedirs(TEACHER_REPORTS_DIR, exist_ok=True)

# Set up logging to capture terminal output
class TeeLogger:
    """Captures print output to both terminal and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', buffering=1)
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

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
LAMBDA_COV = 2.0    # Weight for Interaction Diversity (Entropy)
LAMBDA_HARD = 12.0   # Weight for Adversarial Hardness

# ==========================================
# 1. DATA PREPARATION
# ==========================================
print("\n📊 Loading Data...")

X_train, X_test, y_train, y_test, X_min, X_max, col_names, scaler, ht, NUM_CLASSES = datasets.load_dataset(DATASET_NAME)

# Auto-detect dimensions
DATA_DIM = X_train.shape[1]
NUM_CLASSES = len(np.unique(y_train))

print(f"📋 Dataset Info:")
print(f"   Dataset: {DATASET_NAME}")
print(f"   Samples: {X_train.shape[0]}")
print(f"   Features (DATA_DIM): {DATA_DIM}")
print(f"   Classes (NUM_CLASSES): {NUM_CLASSES}")

# Start logging to file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(TEACHER_REPORTS_DIR, f'training_log_{DATASET_NAME}_{timestamp}.txt')
logger = TeeLogger(log_filename)
sys.stdout = logger

print("\n" + "="*100)
print(f"TRAINING SESSION: {DATASET_NAME.upper()} with {TEACHER_TYPE.upper()} Teacher")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)

# Convert to Tensors
X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
y_test_tensor = torch.LongTensor(y_test).to(DEVICE)

# ==========================================
# 1.5 RANDOM DATA GENERATION MODULE
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
        
        num_samples = samples.shape[0]
        current_size = len(self.buffer)
        
        if current_size < self.max_size:
            # Calculate how many samples can fit before buffer is full
            space_available = self.max_size - current_size
            num_to_add = min(num_samples, space_available)
            
            # Add samples that fit
            for i in range(num_to_add):
                self.buffer.append(samples[i:i+1])
            
            # If there are remaining samples, switch to overwrite mode
            if num_samples > num_to_add:
                remaining_samples = samples[num_to_add:]
                for i in range(remaining_samples.shape[0]):
                    self.buffer[self.position] = remaining_samples[i:i+1]
                    self.position = (self.position + 1) % self.max_size
        else:
            # Buffer is full, overwrite old samples
            for i in range(num_samples):
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
from models import TeacherNet, StudentNet, Generator, BinLearner

# ==========================================
# 3. LOSS FUNCTIONS
# ==========================================
from losses import (
    InteractionDiversityLoss,
    VarianceBasedBinLoss,
    calculate_hard_variance,
    compute_coverage,
    update_cumulative_coverage,
    compute_coverage_bonus
)

# ==========================================
# 3.5 FEATURE SPACE DIVERSITY LOSS
# ==========================================
class FeatureSpaceDiversityLoss(nn.Module):
    """
    Feature Space Diversity Loss: Encourages diverse samples in feature space.
    
    Maximizes variance across the batch for each feature dimension.
    This prevents mode collapse without requiring gradients through the teacher.
    
    L_feature_div = -Σ(Var(x[:, d]))  for d=0 to D (features)
    
    Lower loss = higher variance in generated samples = more diversity
    """
    def __init__(self):
        super(FeatureSpaceDiversityLoss, self).__init__()
    
    def forward(self, x_gen):
        """
        Args:
            x_gen: Generated samples, shape (N, D) where N=batch_size, D=features
        
        Returns:
            loss: Negative total variance (scalar)
        """
        # Compute variance per feature dimension
        # var(x[:, d]) = E[(x - μ)²]
        feature_variance = torch.var(x_gen, dim=0, unbiased=False)  # Shape: (D,)
        
        # Sum variance across all features
        total_variance = feature_variance.sum()
        
        # Return negative variance (we want to MAXIMIZE variance, so MINIMIZE negative variance)
        loss = -total_variance
        
        return loss

# Note: Loss classes and coverage utilities are now imported from losses.py

# ==========================================
# 4. TRAINING PIPELINE
# ==========================================

# A. Pre-train Teacher
print("\n🎓 Pre-training Teacher...")

if TEACHER_TYPE == 'neural':
    # Neural network teacher
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
    
    print(f"✅ Neural Teacher Trained.")
    print(f"   Train Accuracy: {teacher_train_acc*100:.2f}%")
    print(f"   Test Accuracy:  {teacher_test_acc*100:.2f}%")
    
    # Flag for whether teacher outputs logits or probabilities
    teacher_outputs_probs = False

elif TEACHER_TYPE == 'xgboost':
    # XGBoost teacher
    import xgboost as xgb
    from models import XGBoostTeacherWrapper
    
    print("   Training XGBoost classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic' if NUM_CLASSES == 2 else 'multi:softprob',
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train, y_train)
    
    # Wrap in PyTorch-compatible wrapper
    teacher = XGBoostTeacherWrapper(xgb_model, device=DEVICE)
    
    # Evaluate
    teacher_train_preds = xgb_model.predict(X_train)
    teacher_train_acc = accuracy_score(y_train, teacher_train_preds)
    
    teacher_test_preds = xgb_model.predict(X_test)
    teacher_test_acc = accuracy_score(y_test, teacher_test_preds)
    
    print(f"✅ XGBoost Teacher Trained.")
    print(f"   Train Accuracy: {teacher_train_acc*100:.2f}%")
    print(f"   Test Accuracy:  {teacher_test_acc*100:.2f}%")
    
    # XGBoost wrapper returns probabilities, not logits
    teacher_outputs_probs = True

elif TEACHER_TYPE == 'randomforest':
    # Random Forest teacher
    from sklearn.ensemble import RandomForestClassifier
    from models import RandomForestTeacherWrapper
    
    print("   Training Random Forest classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Wrap in PyTorch-compatible wrapper
    teacher = RandomForestTeacherWrapper(rf_model, device=DEVICE)
    
    # Evaluate
    teacher_train_preds = rf_model.predict(X_train)
    teacher_train_acc = accuracy_score(y_train, teacher_train_preds)
    
    teacher_test_preds = rf_model.predict(X_test)
    teacher_test_acc = accuracy_score(y_test, teacher_test_preds)
    
    print(f"✅ Random Forest Teacher Trained.")
    print(f"   Train Accuracy: {teacher_train_acc*100:.2f}%")
    print(f"   Test Accuracy:  {teacher_test_acc*100:.2f}%")
    
    # Random Forest wrapper returns probabilities, not logits
    teacher_outputs_probs = True

else:
    raise ValueError(f"Unknown TEACHER_TYPE: {TEACHER_TYPE}")

# Add after teacher training (around line 510)
def get_teacher_probs(x):
    """Get probabilities from teacher regardless of type"""
    output = teacher(x)
    if TEACHER_TYPE in ['xgboost', 'randomforest']:
        return output  # Already probabilities
    else:
        return F.softmax(output, dim=1)  # Convert logits to probs

# Teacher-adaptive temperature settings
if TEACHER_TYPE == 'xgboost':
    # XGBoost: Gradient boosting produces moderately sharp predictions
    PHASE1_TEMP_START = 1.5   # Start softer
    PHASE1_TEMP_END = 0.1      # End less hard
    PHASE2_TEMP = 0.4          # Softer for exploration
    DISTILL_TEMPERATURE = 2.0  # Soften for better knowledge transfer
    print(f"\n🌡️  Using SOFT temperature schedule (XGBoost teacher)")
elif TEACHER_TYPE == 'randomforest':
    # Random Forest: Majority voting produces very sharp predictions (close to 0/1)
    # Use harder boundaries to match the discrete nature of RF
    PHASE1_TEMP_START = 1.2   # Slightly softer than neural
    PHASE1_TEMP_END = 0.08    # Almost as hard as neural
    PHASE2_TEMP = 0.25         # Moderately hard boundaries
    DISTILL_TEMPERATURE = 1.5  # Less softening than XGBoost (RF already very sharp)
    print(f"\n🌡️  Using MODERATE temperature schedule (RandomForest teacher)")
else:
    # Neural teacher has softer predictions, can use harder boundaries
    PHASE1_TEMP_START = 1.0
    PHASE1_TEMP_END = 0.05
    PHASE2_TEMP = 0.2
    DISTILL_TEMPERATURE = 1.0
    print("\n🌡️  Using HARD temperature schedule (Neural teacher)")

print(f"   Phase 1 Temperature: {PHASE1_TEMP_START} → {PHASE1_TEMP_END}")
print(f"   Phase 2 Temperature: {PHASE2_TEMP}")
print(f"   Distillation Temperature: {DISTILL_TEMPERATURE}")

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
        teacher_probs = get_teacher_probs(x_random)
    
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
# C. PHASE 1: BIN LEARNER + GENERATOR TRAINING (Student Prediction Diversity)
# ==========================================

# History tracking
history = {
    "diversity": [],
    "hardness": [],
    "student": [],
    "test_acc": [],
    "agreement": [],
    "coverage_snapshot": [],
    "coverage_cumulative": [],
    "bin_total": [],
    "bin_intra": [],
    "bin_inter": [],
    "student_div": [],  # NEW: Track student diversity loss
}

print("\n📍 PHASE 1: Training Bin Learner + Generator (Student Prediction Diversity)...")
print(f"   Bin Learner: Learning stable boundaries")
print(f"   Generator: Training to maximize student prediction diversity")
print(f"   Student: Frozen during Phase 1 (using warmup state)")

BIN_LEARNER_EPOCHS = 100
loss_bin_fn = VarianceBasedBinLoss()

# ==========================================
# STUDENT PREDICTION DIVERSITY LOSS
# ==========================================
class StudentPredictionDiversityLoss(nn.Module):
    """
    Student Prediction Diversity Loss: Maximizes entropy of student predictions.
    
    This works because gradients flow through the student (differentiable NN),
    not through the teacher (which may be a non-differentiable tree model).
    
    L_student_div = -H(α) where α = E[student(x_gen)]
    
    Lower loss = higher prediction entropy = more diverse samples
    """
    def __init__(self):
        super(StudentPredictionDiversityLoss, self).__init__()
    
    def forward(self, student_logits):
        """
        Args:
            student_logits: Student predictions, shape (N, C) where C=num_classes
        
        Returns:
            loss: Negative entropy (scalar)
        """
        # Convert logits to probabilities
        probs = F.softmax(student_logits, dim=1)  # Shape: (N, C)
        
        # Compute average prediction across batch
        alpha = probs.mean(dim=0)  # Shape: (C,)
        
        # Compute entropy: H(α) = -Σ(α_j * log(α_j))
        entropy = -(alpha * torch.log(alpha + 1e-8)).sum()
        
        # Return negative entropy (we want to MAXIMIZE entropy, so MINIMIZE negative)
        loss = -entropy
        
        return loss

loss_student_div_fn = StudentPredictionDiversityLoss()

# Optimizers for Phase 1 - Bin learner AND generator
opt_bin = optim.Adam(bin_learner.parameters(), lr=0.01)
opt_gen_phase1 = optim.Adam(generator.parameters(), lr=0.001)

# Schedulers
scheduler_bin = optim.lr_scheduler.CosineAnnealingLR(opt_bin, T_max=BIN_LEARNER_EPOCHS)
scheduler_gen_phase1 = optim.lr_scheduler.CosineAnnealingLR(opt_gen_phase1, T_max=BIN_LEARNER_EPOCHS)

# Freeze student during Phase 1 (it will be trained in Phase 2)
for param in student.parameters():
    param.requires_grad = False

# Weight for student diversity loss
LAMBDA_STUDENT_DIV = 2.0

# Track bin boundaries AND variance over epochs
boundary_history = []
variance_history = []
boundary_checkpoints = list(range(0, BIN_LEARNER_EPOCHS + 1, 20))

print(f"\n{'Epoch':<6} | {'Bin Total':<10} | {'Intra':<10} | {'Inter':<10} | {'Stu Div':<10} | {'Intra Var':<12} | {'Inter Var':<12}")
print("-" * 110)

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
    teacher_probs_sample = get_teacher_probs(X_sample)
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
    # Anneal temperature
    bin_learner.temperature = PHASE1_TEMP_START - ((PHASE1_TEMP_START - PHASE1_TEMP_END) * (epoch / BIN_LEARNER_EPOCHS))
    
    # ==========================
    # GENERATOR UPDATE (Student Prediction Diversity)
    # ==========================
    opt_gen_phase1.zero_grad()
    
    z_gen = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
    x_gen = generator(z_gen)
    
    # Get student predictions (student is frozen, but gradients flow through it)
    student_logits = student(x_gen)
    l_student_div = loss_student_div_fn(student_logits)
    
    # Weighted loss
    l_gen_phase1 = LAMBDA_STUDENT_DIV * l_student_div
    
    if not torch.isnan(l_gen_phase1):
        l_gen_phase1.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_gen_phase1.step()
    
    # ==========================
    # BIN LEARNER UPDATE (Using generator samples)
    # ==========================
    opt_bin.zero_grad()
    
    # Generate new samples for bin learner
    z_bin = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
    x_gen_bin = generator(z_bin).detach()  # Detach to avoid backprop through generator again
    
    with torch.no_grad():
        t_probs = get_teacher_probs(x_gen_bin)
    
    mship = bin_learner(x_gen_bin)
    l_bin, l_intra, l_inter, intra_var, inter_var = loss_bin_fn(mship, t_probs)
    
    if not torch.isnan(l_bin):
        l_bin.backward()
        torch.nn.utils.clip_grad_norm_(bin_learner.parameters(), 1.0)
        opt_bin.step()
    
    # Step schedulers
    scheduler_bin.step()
    scheduler_gen_phase1.step()
    
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
            teacher_probs_sample = get_teacher_probs(X_sample)
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
        print(f"{epoch:<6} | {l_bin.item():<10.5f} | {l_intra.item():<10.5f} | {l_inter.item():<10.5f} | {l_student_div.item():<10.5f} | {intra_var:<12.5f} | {inter_var:<12.5f}")
        
        # Store in history
        history['bin_total'].append(l_bin.item())
        history['bin_intra'].append(l_intra.item())
        history['bin_inter'].append(l_inter.item())
        history['student_div'].append(l_student_div.item())

print(f"✅ Phase 1 Complete!")
print(f"   Bin boundaries are now frozen")
print(f"   Generator trained to maximize student prediction diversity")

# Unfreeze student for Phase 2
for param in student.parameters():
    param.requires_grad = True

# ==========================================
# C.1 VISUALIZE BIN BOUNDARY EVOLUTION
# ==========================================
print("\n📊 Generating bin boundary evolution report...")

import os
os.makedirs('reports', exist_ok=True)

with open(os.path.join(TEACHER_REPORTS_DIR, f'bin_boundaries_{DATASET_NAME}.txt'), 'w', encoding='utf-8') as f:
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

print(f"✅ Bin boundary report saved to: {os.path.join(TEACHER_REPORTS_DIR, f'bin_boundaries_{DATASET_NAME}.txt')}")

# ==========================================
# C.2 VARIANCE EVOLUTION REPORT
# ==========================================
print("\n📊 Generating variance evolution report...")

with open(os.path.join(TEACHER_REPORTS_DIR, f'variance_evolution_{DATASET_NAME}.txt'), 'w', encoding='utf-8') as f:
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

print(f"✅ Variance evolution report saved to: {os.path.join(TEACHER_REPORTS_DIR, f'variance_evolution_{DATASET_NAME}.txt')}")

# ==========================================
# D. PHASE 2: KNOWLEDGE DISTILLATION (Generator + Student)
# ==========================================

print("\n🔄 Re-initializing Generator for Phase 2...")
generator = Generator(LATENT_DIM, DATA_DIM, X_mean, X_min, X_max).to(DEVICE)
print("   Generator reset to fresh random weights")
print("   This avoids carrying Phase 1's class-diversity bias into Phase 2")

print("\n🎓 PHASE 2: Knowledge Distillation with Frozen Bins...")
print(f"   Generator: Fresh start - Hardness + Diversity Loss")
print(f"   Student: KL Divergence Loss")
print(f"   Bin Learner: FROZEN")

# Freeze bin learner
for param in bin_learner.parameters():
    param.requires_grad = False

# Set temperature for Phase 2 (generator training)
bin_learner.temperature = PHASE2_TEMP  # Teacher-adaptive

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

# History tracking

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
        t_probs = get_teacher_probs(x_gen)
    s_log_probs = F.log_softmax(student(x_gen), dim=1)
    l_hard = -1.0 * F.kl_div(s_log_probs, t_probs, reduction='batchmean')

    # 3. Coverage Bonus (Reward unexplored bins)
    l_coverage = compute_coverage_bonus(mship, cumulative_coverage_tracker)

    # Combine losses
    l_gen = (LAMBDA_COV * l_div) + (LAMBDA_HARD * l_hard) #+ (8.0 * l_coverage)

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
        # Apply distillation temperature for softer targets (especially for tree-based models)
        if DISTILL_TEMPERATURE != 1.0:
            if TEACHER_TYPE == 'neural':
                # For neural teacher, apply temperature to logits then softmax
                logits = teacher(x_mixed)
                t_probs_stu = F.softmax(logits / DISTILL_TEMPERATURE, dim=1)
            else:
                # For tree-based teachers (XGBoost/RandomForest), they return probabilities.
                # Use a power transform to soften/sharpen and renormalize.
                probs = teacher(x_mixed)
                t_probs_stu = probs.pow(1.0 / DISTILL_TEMPERATURE)
                t_probs_stu = t_probs_stu / t_probs_stu.sum(dim=1, keepdim=True)
        else:
            # No temperature adjustment; use teacher's probabilities consistently
            t_probs_stu = get_teacher_probs(x_mixed)
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
            teacher_probs_var = get_teacher_probs(x_gen_var)
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
    
    # Legacy interval logging removed: previous code referenced undefined
    # variables (report_interval, epochs, and file handle 'f') and has been
    # intentionally removed to avoid runtime errors.
    
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

visualization.generate_loss_report(
    history, 
    DATASET_NAME, 
    EPOCHS,
    save_dir=TEACHER_REPORTS_DIR,
    lambda_cov=LAMBDA_COV,
    lambda_hard=LAMBDA_HARD,
    batch_size=BATCH_SIZE,
    num_bins=NUM_BINS,
    num_features=DATA_DIM,
    num_classes=NUM_CLASSES,
    num_samples=X_train.shape[0]
)

print("\n" + "="*100)
print("✅ ALL TASKS COMPLETED SUCCESSFULLY!")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)

# Close logging and restore stdout
sys.stdout = logger.terminal
logger.close()
print(f"\n📄 Full training log saved to: {log_filename}")
print(f"📂 All reports saved to: {TEACHER_REPORTS_DIR}/")
