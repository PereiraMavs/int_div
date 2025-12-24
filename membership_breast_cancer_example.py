"""
Concrete example showing membership calculation with actual breast cancer data.
"""

import torch
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from bin_learner import BinLearner

print("="*80)
print("MEMBERSHIP CALCULATION - BREAST CANCER DATASET EXAMPLE")
print("="*80)

# Load data
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Focus on one feature for clarity
feature_idx = 0  # mean radius
feature_name = feature_names[feature_idx]

print(f"\n📊 Looking at Feature {feature_idx}: '{feature_name}'")
print(f"   Range: [{X_train[:, feature_idx].min():.2f}, {X_train[:, feature_idx].max():.2f}]")

# Get feature ranges
feature_ranges = [
    (float(X_train[:, i].min()), float(X_train[:, i].max()))
    for i in range(X_train.shape[1])
]

# Initialize BinLearner
num_bins = 5
bin_learner = BinLearner(
    num_features=30,
    num_bins=num_bins,
    feature_ranges=feature_ranges,
    temperature=1.0
)

# Get boundaries for our feature
boundaries = bin_learner.get_boundaries()
feature_boundaries = boundaries[feature_idx].detach().numpy()

print(f"\n🔹 Bin Boundaries for '{feature_name}':")
for i in range(num_bins):
    print(f"   Bin {i}: [{feature_boundaries[i]:.2f}, {feature_boundaries[i+1]:.2f})")

# ============================================================================
# EXAMPLE 1: Single sample
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 1: Calculate Membership for a Single Sample")
print("="*80)

# Take first sample from training set
sample_idx = 0
sample_value = X_train[sample_idx, feature_idx]

print(f"\nSample {sample_idx}:")
print(f"   Feature value: {sample_value:.2f}")
print(f"   True label: {['Malignant', 'Benign'][y_train[sample_idx]]}")

# Convert to tensor
sample_tensor = torch.tensor(X_train[sample_idx:sample_idx+1], dtype=torch.float32)

# Get membership
bin_learner.eval()
with torch.no_grad():
    membership = bin_learner(sample_tensor)

# Extract membership for our feature
feature_membership = membership[0, feature_idx].numpy()

print(f"\n🎯 Soft Bin Memberships:")
for i in range(num_bins):
    bar = "█" * int(feature_membership[i] * 50)
    print(f"   Bin {i}: {feature_membership[i]:.4f} ({feature_membership[i]*100:5.1f}%) {bar}")

print(f"\nSum of memberships: {feature_membership.sum():.6f} ✓")

# Hard assignment
hard_bin = np.argmax(feature_membership)
print(f"\nHard bin assignment (argmax): Bin {hard_bin}")

# ============================================================================
# EXAMPLE 2: Multiple samples showing variety
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 2: Membership for Multiple Samples")
print("="*80)

# Select 5 diverse samples
selected_indices = [0, 50, 150, 250, 350]
selected_values = X_train[selected_indices, feature_idx]

print(f"\nShowing 5 samples with different feature values:\n")
print(f"{'Sample':>6} | {'Value':>8} | {'Label':>10} | Bin Memberships")
print("-" * 85)

# Convert to tensor
samples_tensor = torch.tensor(X_train[selected_indices], dtype=torch.float32)

with torch.no_grad():
    memberships = bin_learner(samples_tensor)

for i, idx in enumerate(selected_indices):
    value = selected_values[i]
    label = ['Malignant', 'Benign'][y_train[idx]]
    mems = memberships[i, feature_idx].numpy()
    hard_bin = np.argmax(mems)
    
    mem_str = " | ".join([f"{m*100:4.1f}%" for m in mems])
    print(f"{idx:6d} | {value:8.2f} | {label:>10} | {mem_str} → Bin {hard_bin}")

# ============================================================================
# VISUALIZATION: Show actual data distribution with memberships
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION: Feature Distribution with Soft Memberships")
print("="*80)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Subplot 1: Histogram with boundaries
ax1 = axes[0]
ax1.hist(X_train[:, feature_idx], bins=50, alpha=0.6, color='skyblue', edgecolor='black')

for i, boundary in enumerate(feature_boundaries):
    if i == 0:
        ax1.axvline(boundary, color='red', linestyle='--', linewidth=2, label='Bin Boundaries')
    else:
        ax1.axvline(boundary, color='red', linestyle='--', linewidth=2)

# Mark selected samples
for i, idx in enumerate(selected_indices):
    value = selected_values[i]
    ax1.scatter([value], [ax1.get_ylim()[1] * 0.85], 
                s=100, marker='v', color='darkgreen', zorder=5)
    ax1.text(value, ax1.get_ylim()[1] * 0.9, f'#{idx}', 
             ha='center', fontsize=9, color='darkgreen', fontweight='bold')

ax1.set_xlabel(f'Feature Value: {feature_name}', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title(f'Distribution of "{feature_name}" with Learned Bin Boundaries', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Subplot 2: Membership curves
ax2 = axes[1]

# Generate continuous range
x_range = np.linspace(X_train[:, feature_idx].min(), X_train[:, feature_idx].max(), 200)
x_tensor = torch.zeros(200, 30)
x_tensor[:, feature_idx] = torch.tensor(x_range, dtype=torch.float32)

with torch.no_grad():
    memberships_continuous = bin_learner(x_tensor)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i in range(num_bins):
    mems = memberships_continuous[:, feature_idx, i].numpy()
    ax2.plot(x_range, mems, label=f'Bin {i}', linewidth=2.5, color=colors[i])
    ax2.fill_between(x_range, 0, mems, alpha=0.2, color=colors[i])

# Mark boundaries
for boundary in feature_boundaries:
    ax2.axvline(boundary, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

# Mark selected samples
for i, idx in enumerate(selected_indices):
    value = selected_values[i]
    mems = memberships[i, feature_idx].numpy()
    hard_bin = np.argmax(mems)
    ax2.scatter([value], [mems[hard_bin]], s=150, marker='o', 
                color=colors[hard_bin], edgecolors='black', linewidths=2, zorder=5)
    ax2.text(value, mems[hard_bin] + 0.05, f'#{idx}', 
             ha='center', fontsize=9, fontweight='bold')

ax2.set_xlabel(f'Feature Value: {feature_name}', fontsize=12)
ax2.set_ylabel('Bin Membership (probability)', fontsize=12)
ax2.set_title('Soft Bin Membership Functions', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('breast_cancer_membership_example.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved to 'breast_cancer_membership_example.png'")

# ============================================================================
# SHOW TEMPERATURE EFFECT ON ACTUAL DATA
# ============================================================================
print("\n" + "="*80)
print("TEMPERATURE EFFECT ON REAL DATA")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
temperatures = [0.5, 1.0, 10.0]

for idx, temp in enumerate(temperatures):
    ax = axes[idx]
    
    # Create bin learner with different temperature
    bin_learner_temp = BinLearner(
        num_features=30,
        num_bins=num_bins,
        feature_ranges=feature_ranges,
        temperature=temp
    )
    
    # Copy boundaries from original
    with torch.no_grad():
        bin_learner_temp.internal_boundaries.copy_(bin_learner.internal_boundaries)
    
    bin_learner_temp.eval()
    
    # Get memberships
    with torch.no_grad():
        memberships_temp = bin_learner_temp(x_tensor)
    
    # Plot
    for i in range(num_bins):
        mems = memberships_temp[:, feature_idx, i].numpy()
        ax.plot(x_range, mems, label=f'Bin {i}', linewidth=2, color=colors[i])
    
    # Mark boundaries
    for boundary in feature_boundaries:
        ax.axvline(boundary, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel(f'{feature_name}', fontsize=11)
    ax.set_ylabel('Membership', fontsize=11)
    ax.set_title(f'Temperature = {temp}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('breast_cancer_temperature_effect.png', dpi=300, bbox_inches='tight')
print("\n✓ Temperature effect visualization saved to 'breast_cancer_temperature_effect.png'")

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. **Soft Membership**: Each sample belongs to ALL bins with different probabilities
   - Sample near bin center: High membership in that bin
   - Sample near boundary: Split membership between adjacent bins
   - Sample far from bin: Near-zero membership

2. **Sum to 1**: All memberships for a feature always sum to 1.0

3. **Differentiable**: Gradients can flow through sigmoid operations
   - Allows learning optimal bin boundaries via backpropagation

4. **Temperature Control**: Higher temperature → sharper transitions
   - Low temp: Fuzzy, overlapping bins (more soft assignments)
   - High temp: Crisp boundaries (approaches hard binning)

5. **Per-Feature Bins**: Each feature has its own set of bins
   - Shape: (N_samples, N_features, N_bins)
   - Bins for feature 0 are independent of bins for feature 1
""")

print("="*80)
