"""
Visual explanation of how soft bin membership is calculated in BinLearner.

This script demonstrates the sigmoid-based soft binning with a concrete example.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("SOFT BIN MEMBERSHIP CALCULATION - STEP BY STEP")
print("="*80)

# ============================================================================
# STEP 1: Setup boundaries
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Define Bin Boundaries")
print("="*80)

# Example: 1 feature, 3 bins
# Feature range: [0, 10]
# Boundaries: [0, 3, 7, 10] creates 3 bins:
#   Bin 0: [0, 3)
#   Bin 1: [3, 7)
#   Bin 2: [7, 10]

boundaries = torch.tensor([[0.0, 3.0, 7.0, 10.0]])  # Shape: (1 feature, 4 boundaries)
temperature = 1.0

print(f"Boundaries: {boundaries[0].tolist()}")
print(f"Temperature: {temperature}")
print(f"\nThis creates 3 bins:")
print(f"  Bin 0: [{boundaries[0,0]:.1f}, {boundaries[0,1]:.1f})")
print(f"  Bin 1: [{boundaries[0,1]:.1f}, {boundaries[0,2]:.1f})")
print(f"  Bin 2: [{boundaries[0,2]:.1f}, {boundaries[0,3]:.1f}]")

# ============================================================================
# STEP 2: Sample data point
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Take a Sample Data Point")
print("="*80)

# Example: sample with value 5.0
x = torch.tensor([[5.0]])  # Shape: (1 sample, 1 feature)

print(f"Sample value: x = {x[0,0].item():.1f}")
print(f"This value falls in Bin 1 (between 3.0 and 7.0)")

# ============================================================================
# STEP 3: Compute differences from boundaries
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Compute (x - boundary) for Each Boundary")
print("="*80)

# Expand for broadcasting
x_expanded = x.unsqueeze(2)  # (1, 1, 1)
boundaries_expanded = boundaries.unsqueeze(0)  # (1, 1, 4)

# Compute differences
diff = x_expanded - boundaries_expanded  # (1, 1, 4)

print(f"x = {x[0,0].item():.1f}")
print(f"\nDifferences (x - boundary):")
for i, boundary in enumerate(boundaries[0]):
    print(f"  x - boundary[{i}] = {x[0,0].item():.1f} - {boundary.item():.1f} = {diff[0,0,i].item():6.1f}")

# ============================================================================
# STEP 4: Apply sigmoid function
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Apply Sigmoid to Each Difference")
print("="*80)

print(f"\nSigmoid formula: σ(z) = 1 / (1 + e^(-z))")
print(f"We compute: σ(temperature × (x - boundary))")
print(f"\nWith temperature = {temperature}:")

sigmoids = torch.sigmoid(temperature * diff)  # (1, 1, 4)

print(f"\nSigmoid values:")
for i, (boundary, sig) in enumerate(zip(boundaries[0], sigmoids[0,0])):
    print(f"  σ({temperature} × {diff[0,0,i].item():6.1f}) = {sig.item():.6f}  "
          f"[boundary = {boundary.item():.1f}]")

print(f"\n📊 Interpretation:")
print(f"  - If x > boundary: sigmoid ≈ 1 (sample is 'above' this boundary)")
print(f"  - If x < boundary: sigmoid ≈ 0 (sample is 'below' this boundary)")
print(f"  - If x ≈ boundary: sigmoid ≈ 0.5 (sample is 'on' this boundary)")

# ============================================================================
# STEP 5: Compute bin memberships
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Compute Bin Memberships")
print("="*80)

print(f"\nMembership for each bin = sigmoid[i] - sigmoid[i+1]")
print(f"\nThis measures 'how much the sample is between boundary i and i+1'")

membership_raw = sigmoids[:, :, :-1] - sigmoids[:, :, 1:]  # (1, 1, 3)

print(f"\nRaw memberships (before normalization):")
for i in range(3):
    sig_left = sigmoids[0, 0, i].item()
    sig_right = sigmoids[0, 0, i+1].item()
    mem = membership_raw[0, 0, i].item()
    print(f"  Bin {i}: σ[{i}] - σ[{i+1}] = {sig_left:.6f} - {sig_right:.6f} = {mem:.6f}")

# ============================================================================
# STEP 6: Normalize to sum to 1
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Normalize So Memberships Sum to 1")
print("="*80)

membership_sum = membership_raw.sum(dim=2, keepdim=True)
membership = membership_raw / (membership_sum + 1e-8)

print(f"\nSum of raw memberships: {membership_sum[0,0,0].item():.6f}")
print(f"\nFinal normalized memberships:")
for i in range(3):
    mem = membership[0, 0, i].item()
    percentage = mem * 100
    print(f"  Bin {i}: {mem:.6f} ({percentage:.2f}%)")

print(f"\nSum of normalized memberships: {membership.sum().item():.6f} ✓")

# ============================================================================
# STEP 7: Visualize with multiple samples
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Visualize Membership for Multiple Sample Values")
print("="*80)

# Create range of x values
x_values = torch.linspace(0, 10, 100).unsqueeze(1)  # (100, 1)

# Compute memberships for all x values
x_expanded = x_values.unsqueeze(2)  # (100, 1, 1)
diff_all = x_expanded - boundaries_expanded  # (100, 1, 4)
sigmoids_all = torch.sigmoid(temperature * diff_all)
membership_all = sigmoids_all[:, :, :-1] - sigmoids_all[:, :, 1:]
membership_all = membership_all / (membership_all.sum(dim=2, keepdim=True) + 1e-8)

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Sigmoid curves
ax1 = axes[0]
for i, boundary in enumerate(boundaries[0]):
    ax1.plot(x_values.numpy(), sigmoids_all[:, 0, i].numpy(), 
             label=f'σ(x - {boundary.item():.1f})', linewidth=2)
    ax1.axvline(boundary.item(), color='gray', linestyle='--', alpha=0.5)

ax1.set_xlabel('Sample Value (x)', fontsize=12)
ax1.set_ylabel('Sigmoid Value', fontsize=12)
ax1.set_title('Step 4: Sigmoid Functions for Each Boundary', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([-0.1, 1.1])

# Plot 2: Membership curves
ax2 = axes[1]
colors = ['blue', 'green', 'red']
for i in range(3):
    ax2.plot(x_values.numpy(), membership_all[:, 0, i].numpy(), 
             label=f'Bin {i} Membership', linewidth=2, color=colors[i])
    # Shade the region
    ax2.fill_between(x_values.numpy().flatten(), 
                      0, membership_all[:, 0, i].numpy().flatten(), 
                      alpha=0.2, color=colors[i])

# Mark boundaries
for boundary in boundaries[0]:
    ax2.axvline(boundary.item(), color='black', linestyle='--', linewidth=1.5, alpha=0.7)

# Mark our example point
ax2.plot(5.0, membership[0, 0, 1].item(), 'ko', markersize=10, 
         label=f'Our example (x=5.0)', zorder=5)

ax2.set_xlabel('Sample Value (x)', fontsize=12)
ax2.set_ylabel('Bin Membership (probability)', fontsize=12)
ax2.set_title('Step 5-6: Final Soft Bin Memberships (Normalized)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([-0.1, 1.1])

plt.tight_layout()
plt.savefig('membership_calculation_visualization.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to 'membership_calculation_visualization.png'")

# ============================================================================
# STEP 8: Show effect of temperature
# ============================================================================
print("\n" + "="*80)
print("STEP 8: Effect of Temperature on Membership Sharpness")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
temperatures = [0.5, 1.0, 5.0]

for idx, temp in enumerate(temperatures):
    ax = axes[idx]
    
    # Compute memberships with this temperature
    sigmoids_temp = torch.sigmoid(temp * diff_all)
    membership_temp = sigmoids_temp[:, :, :-1] - sigmoids_temp[:, :, 1:]
    membership_temp = membership_temp / (membership_temp.sum(dim=2, keepdim=True) + 1e-8)
    
    # Plot
    for i in range(3):
        ax.plot(x_values.numpy(), membership_temp[:, 0, i].numpy(), 
                label=f'Bin {i}', linewidth=2, color=colors[i])
    
    # Mark boundaries
    for boundary in boundaries[0]:
        ax.axvline(boundary.item(), color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Sample Value (x)', fontsize=11)
    ax.set_ylabel('Membership', fontsize=11)
    ax.set_title(f'Temperature = {temp}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.1])

plt.tight_layout()
plt.savefig('temperature_effect_visualization.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Temperature effect visualization saved to 'temperature_effect_visualization.png'")

print(f"\n📊 Temperature Effect:")
print(f"  - Low temp (0.5): Smooth, gradual transitions (samples belong to multiple bins)")
print(f"  - Medium temp (1.0): Balanced (some overlap)")
print(f"  - High temp (5.0): Sharp, hard boundaries (samples mostly in one bin)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
The membership calculation works as follows:

1. **Start with boundaries**: [0, 3, 7, 10] creates 3 bins
2. **For each sample x**: Compute how "far above" each boundary it is
3. **Apply sigmoid**: Convert differences to smooth 0-1 values
4. **Compute membership**: Bin_i = σ(x - boundary_i) - σ(x - boundary_{i+1})
5. **Normalize**: Ensure all memberships sum to 1

Key insight: 
- If x is BETWEEN two boundaries, the difference of their sigmoids is high
- If x is FAR from a bin's range, that bin's membership approaches 0
- This creates "soft" bin assignments where samples can partially belong to multiple bins

This is differentiable, so gradients can flow back to adjust the boundaries!
""")

print("="*80)
