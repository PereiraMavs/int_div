import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

from bin_learner import BinLearner, BinDispersionLoss
from generator import Generator, generator_loss
from teacher import TeacherMLP
from student import StudentMLP

# Suppress warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# ============================================================================
# 1. LOAD DATA (RAW - NO SCALING)
# ============================================================================
print("="*70)
print("LOADING BREAST CANCER DATASET (RAW VALUES)")
print("="*70)

data = load_breast_cancer()
X = data.data  # RAW values (no scaling!)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Get raw feature ranges for bin initialization
feature_ranges = [
    (float(X_train[:, i].min()), float(X_train[:, i].max()))
    for i in range(X_train.shape[1])
]

print(f"Dataset: {X_train.shape[0]} train, {X_test.shape[0]} test")
print(f"Features: {X_train.shape[1]}")
print(f"Raw feature ranges (first 3):")
for i in range(3):
    print(f"  Feature {i}: [{feature_ranges[i][0]:.2f}, {feature_ranges[i][1]:.2f}]")

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

# ============================================================================
# 2. TRAIN TEACHER (HIGH-CAPACITY MLP)
# ============================================================================
print("\n" + "="*70)
print("TRAINING TEACHER MODEL (High-Capacity MLP)")
print("="*70)

teacher = TeacherMLP(input_dim=30, hidden_dims=[128, 64, 32], output_dim=2).to(device)
optimizer_teacher = optim.Adam(teacher.parameters(), lr=0.001)
ce_loss_fn = nn.CrossEntropyLoss()

# Train teacher for 100 epochs
teacher_epochs = 100
batch_size_teacher = 64

print(f"Training for {teacher_epochs} epochs...")

for epoch in range(teacher_epochs):
    teacher.train()
    perm = torch.randperm(X_train_t.size(0))
    
    total_loss = 0
    num_batches = 0
    
    for i in range(0, X_train_t.size(0), batch_size_teacher):
        idx = perm[i:i+batch_size_teacher]
        batch_x = X_train_t[idx]
        batch_y = y_train_t[idx]
        
        optimizer_teacher.zero_grad()
        logits = teacher(batch_x)
        loss = ce_loss_fn(logits, batch_y)
        loss.backward()
        optimizer_teacher.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    if epoch % 20 == 0 or epoch == teacher_epochs - 1:
        teacher.eval()
        with torch.no_grad():
            test_logits = teacher(X_test_t)
            test_preds = torch.argmax(test_logits, dim=1)
            test_acc = (test_preds == y_test_t).float().mean().item()
        
        print(f"Teacher Epoch {epoch:3d} | Loss: {total_loss/num_batches:.4f} | Test Acc: {test_acc:.4f}")

# Freeze teacher
for param in teacher.parameters():
    param.requires_grad = False
teacher.eval()

teacher_acc = test_acc
print(f"\n✓ Teacher trained! Final accuracy: {teacher_acc:.4f}")

# ============================================================================
# 3. INITIALIZE MODELS
# ============================================================================
print("\n" + "="*70)
print("INITIALIZING MODELS")
print("="*70)

# Generator
latent_dim = 100
generator = Generator(latent_dim=latent_dim, num_features=30, hidden_dims=[128, 64]).to(device)
print(f"Generator: {sum(p.numel() for p in generator.parameters())} params")

# Student (lightweight)
student = StudentMLP(input_dim=30, hidden_dim=32, output_dim=2).to(device)
print(f"Student: {sum(p.numel() for p in student.parameters())} params")

# Bin Learner
num_bins = 5
bin_learner = BinLearner(
    num_features=30,
    num_bins=num_bins,
    feature_ranges=feature_ranges,
    temperature=1.0
).to(device)
print(f"Bin Learner: {sum(p.numel() for p in bin_learner.parameters())} params")

# ============================================================================
# 4. INITIALIZE OPTIMIZERS AND LOSSES
# ============================================================================
print("\n" + "="*70)
print("INITIALIZING OPTIMIZERS")
print("="*70)

optimizer_gen = optim.Adam(generator.parameters(), lr=0.001)
optimizer_student = optim.Adam(student.parameters(), lr=0.001)
optimizer_bins = optim.Adam(bin_learner.parameters(), lr=0.01)

bin_loss_fn = BinDispersionLoss(lambda_repulsion=0.5, temperature=1.0).to(device)
kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

# Add gradient clipping to prevent explosion
max_grad_norm = 1.0

print("  Generator: lr=0.001")
print("  Student: lr=0.001")
print("  Bin Learner: lr=0.01")

# ============================================================================
# 5. TRAINING LOOP (SEQUENTIAL: MAP → SCENARIOS → LEARN)
# ============================================================================
print("\n" + "="*70)
print("TRAINING WITH SEQUENTIAL UPDATES")
print("="*70)

num_epochs = 200
batch_size = 128
bin_learner_warmup = 20

print(f"Bin learner warmup: {bin_learner_warmup} epochs")
print(f"Training for {num_epochs} epochs\n")

for epoch in range(num_epochs):
    
    # Temperature annealing for bin learner
    # Start at 1.0, anneal to 10.0 over training
    temp = 1.0 + (10.0 - 1.0) * (epoch / num_epochs)
    bin_learner.set_temperature(temp)
    bin_loss_fn.temperature = temp
    
    # ========================================================================
    # PHASE 1: MAP - Train Bin Learner (after warmup)
    # ========================================================================
    if epoch >= bin_learner_warmup:
        generator.eval()
        student.eval()
        bin_learner.train()
        
        optimizer_bins.zero_grad()
        
        # Generate samples
        z = torch.randn(batch_size, latent_dim, device=device)
        with torch.no_grad():
            x_gen = generator(z)
            teacher_probs = teacher.predict_proba(x_gen)
        
        # Compute bin memberships
        membership = bin_learner(x_gen)
        
        # Bin dispersion loss
        bin_loss, bin_metrics = bin_loss_fn(membership, teacher_probs)
        
        # Update bins
        bin_loss.backward()
        torch.nn.utils.clip_grad_norm_(bin_learner.parameters(), max_grad_norm)
        optimizer_bins.step()
    else:
        # Still compute for logging
        bin_learner.eval()
        with torch.no_grad():
            z = torch.randn(batch_size, latent_dim, device=device)
            x_gen = generator(z)
            teacher_probs = teacher.predict_proba(x_gen)
            membership = bin_learner(x_gen)
            bin_loss, bin_metrics = bin_loss_fn(membership, teacher_probs)
    
    # ========================================================================
    # PHASE 2: SCENARIOS - Train Generator
    # ========================================================================
    generator.train()
    student.eval()
    bin_learner.eval()
    
    optimizer_gen.zero_grad()
    
    # Generate samples
    z = torch.randn(batch_size, latent_dim, device=device)
    x_gen = generator(z)
    
    # Get predictions (with gradients for generator)
    teacher_probs = teacher.predict_proba(x_gen)
    student_probs = student.predict_proba(x_gen)
    membership = bin_learner(x_gen)
    
    # Generator loss
    gen_loss, loss_adv, loss_div = generator_loss(student_probs, teacher_probs, membership)
    
    # Update generator
    gen_loss.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)
    optimizer_gen.step()
    
    # ========================================================================
    # PHASE 3: LEARN - Train Student
    # ========================================================================
    generator.eval()
    student.train()
    bin_learner.eval()
    
    optimizer_student.zero_grad()
    
    # Generate NEW samples
    z = torch.randn(batch_size, latent_dim, device=device)
    with torch.no_grad():
        x_gen = generator(z)
        teacher_probs = teacher.predict_proba(x_gen)
    
    # Student forward
    student_logits = student(x_gen)
    
    # KL divergence loss
    temperature_kd = 3.0
    log_probs_student = torch.log_softmax(student_logits / temperature_kd, dim=1)
    teacher_probs_soft = torch.softmax(
        torch.log(teacher_probs + 1e-8) * temperature_kd, dim=1
    )
    kl_loss = kl_loss_fn(log_probs_student, teacher_probs_soft) * (temperature_kd ** 2)
    
    # Update student
    kl_loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
    optimizer_student.step()
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    if epoch % 20 == 0 or epoch == num_epochs - 1:
        # Evaluate student on real test data
        student.eval()
        with torch.no_grad():
            test_logits = student(X_test_t)
            test_preds = torch.argmax(test_logits, dim=1)
            student_acc = (test_preds == y_test_t).float().mean().item()
        
        bin_status = "FROZEN" if epoch < bin_learner_warmup else "TRAINING"
        print(f"Epoch {epoch:3d} | Temp: {temp:.2f} | "
              f"Gen: {gen_loss.item():.4f} (Adv: {loss_adv.item():.4f}, Div: {loss_div.item():.4f}) | "
              f"Bin: {bin_loss.item():.4f} [{bin_status}] | "
              f"Student KL: {kl_loss.item():.4f} | "
              f"Student Acc: {student_acc:.4f}")

    # Check for NaN
    if torch.isnan(gen_loss) or torch.isnan(bin_loss) or torch.isnan(kl_loss):
        print(f"\n!!! NaN detected at epoch {epoch}! Stopping training.")
        print(f"  Gen loss: {gen_loss.item()}")
        print(f"  Bin loss: {bin_loss.item()}")
        print(f"  KL loss: {kl_loss.item()}")
        break

# ============================================================================
# 6. FINAL EVALUATION
# ============================================================================
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

student.eval()
with torch.no_grad():
    test_logits = student(X_test_t)
    test_preds = torch.argmax(test_logits, dim=1)
    final_student_acc = (test_preds == y_test_t).float().mean().item()

print(f"Teacher (MLP) Test Accuracy:  {teacher_acc:.4f}")
print(f"Student (MLP) Test Accuracy:  {final_student_acc:.4f}")
print(f"Accuracy Gap:                 {abs(teacher_acc - final_student_acc):.4f}")

# Save models
torch.save({
    'teacher': teacher.state_dict(),
    'generator': generator.state_dict(),
    'bin_learner': bin_learner.state_dict(),
    'student': student.state_dict(),
    'boundaries': bin_learner.get_boundaries().cpu(),
    'results': {
        'teacher_acc': teacher_acc,
        'student_acc': final_student_acc
    }
}, 'experiment_results.pt')

print("\n✓ Experiment complete! Results saved to 'experiment_results.pt'")