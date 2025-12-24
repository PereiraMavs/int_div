import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from bin_learner import BinLearner
from teacher import TeacherMLP


def load_and_prepare_dataset():
    """
    Load and prepare the breast cancer dataset.
    
    Returns:
        dict: Contains X_train, X_test, y_train, y_test, feature_names, feature_ranges
    """
    print("="*70)
    print("LOADING BREAST CANCER DATASET")
    print("="*70)
    
    data = load_breast_cancer()
    X = data.data  # RAW values, no scaling
    y = data.target
    feature_names = data.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples:  {X_test.shape[0]}")
    print(f"Features:      {X_train.shape[1]}")
    
    # Get feature ranges
    print("\n" + "="*70)
    print("FEATURE RANGES (RAW VALUES)")
    print("="*70)
    
    feature_ranges = [
        (float(X_train[:, i].min()), float(X_train[:, i].max()))
        for i in range(X_train.shape[1])
    ]
    
    print(f"\nShowing first 10 features:")
    for i in range(10):
        f_min, f_max = feature_ranges[i]
        print(f"  {i:2d}. {feature_names[i]:30s} [{f_min:10.2f}, {f_max:10.2f}]")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'feature_ranges': feature_ranges
    }


def train_teacher_model(dataset, device='cpu', epochs=100):
    """
    Train the teacher model on the real dataset.
    
    Args:
        dataset (dict): Dataset dictionary from load_and_prepare_dataset()
        device (str): Device to run on
        epochs (int): Number of training epochs
    
    Returns:
        TeacherMLP: Trained teacher model
    """
    print("\n" + "="*70)
    print("TRAINING TEACHER MODEL")
    print("="*70)
    
    X_train = dataset['X_train']
    X_test = dataset['X_test']
    y_train = dataset['y_train']
    y_test = dataset['y_test']
    
    # Normalize data for better training
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)
    
    # Initialize teacher
    teacher = TeacherMLP(input_dim=30, hidden_dims=[128, 64, 32], output_dim=2).to(device)
    
    # Initialize weights with Xavier initialization
    for module in teacher.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    # Loss and optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    print(f"Training for {epochs} epochs with normalization and learning rate scheduling...")
    
    best_test_acc = 0.0
    patience_counter = 0
    
    teacher.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        logits = teacher(X_train_t)
        loss = criterion(logits, y_train_t)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            # Evaluate
            teacher.eval()
            with torch.no_grad():
                train_logits = teacher(X_train_t)
                train_preds = torch.argmax(train_logits, dim=1)
                train_acc = (train_preds == y_train_t).float().mean().item()
                
                test_logits = teacher(X_test_t)
                test_preds = torch.argmax(test_logits, dim=1)
                test_acc = (test_preds == y_test_t).float().mean().item()
            
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
            
            # Learning rate scheduling
            scheduler.step(test_acc)
            
            # Track best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            teacher.train()
    
    # Final evaluation
    teacher.eval()
    with torch.no_grad():
        test_logits = teacher(X_test_t)
        test_preds = torch.argmax(test_logits, dim=1)
        test_acc = (test_preds == y_test_t).float().mean().item()
    
    print(f"\n✓ Teacher training complete! Final test accuracy: {test_acc:.4f}")
    print(f"  Best test accuracy during training: {best_test_acc:.4f}")
    
    # Store normalization parameters for later use
    teacher.X_mean = torch.tensor(X_mean, dtype=torch.float32, device=device)
    teacher.X_std = torch.tensor(X_std, dtype=torch.float32, device=device)
    
    return teacher


def train_bin_learner_with_teacher(dataset, teacher, num_bins=5, temperature=1.0, 
                                   device='cpu', epochs=100, lr=0.01):
    """
    Train the bin learner using signals from the teacher model.
    
    Args:
        dataset (dict): Dataset dictionary from load_and_prepare_dataset()
        teacher (TeacherMLP): Trained teacher model
        num_bins (int): Number of bins per feature
        temperature (float): Initial temperature for bin learner
        device (str): Device to run on
        epochs (int): Number of training epochs
        lr (float): Learning rate
    
    Returns:
        dict: Results containing initial and final boundaries, bin_learner
    """
    print("\n" + "="*70)
    print("TRAINING BIN LEARNER WITH TEACHER SIGNALS")
    print("="*70)
    
    X_train = dataset['X_train']
    feature_names = dataset['feature_names']
    feature_ranges = dataset['feature_ranges']
    y_train = dataset['y_train']
    
    # Normalize data using teacher's normalization parameters
    X_train_norm = (X_train - teacher.X_mean.cpu().numpy()) / teacher.X_std.cpu().numpy()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_train_norm_t = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    
    # Initialize bin learner
    bin_learner = BinLearner(
        num_features=30,
        num_bins=num_bins,
        feature_ranges=feature_ranges,
        temperature=temperature
    ).to(device)
    
    # Save initial boundaries
    bin_learner.eval()
    with torch.no_grad():
        initial_boundaries = bin_learner.get_boundaries().clone()
    
    print(f"Number of bins per feature: {num_bins}")
    print(f"Initial temperature: {temperature}")
    print(f"Training for {epochs} epochs with lr={lr}")
    
    # Optimizer for bin learner
    optimizer = optim.Adam(bin_learner.parameters(), lr=lr)
    
    # Freeze teacher
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    bin_learner.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Get soft bin memberships
        membership = bin_learner(X_train_t)  # (N, 30, K)
        
        # Get teacher predictions (soft labels) - use normalized data
        with torch.no_grad():
            teacher_logits = teacher(X_train_norm_t)  # (N, 2)
            teacher_probs = torch.softmax(teacher_logits, dim=1)  # (N, 2)
        
        # Compute bin purity: for each bin, how consistent are the teacher predictions?
        # Goal: adjust bins to maximize purity (samples in same bin should have similar predictions)
        
        # For each feature and bin, compute weighted average of teacher predictions
        # membership: (N, F, K)
        # teacher_probs: (N, 2)
        
        # Reshape for broadcasting
        teacher_probs_expanded = teacher_probs.unsqueeze(1).unsqueeze(2)  # (N, 1, 1, 2)
        membership_expanded = membership.unsqueeze(-1)  # (N, F, K, 1)
        
        # Weighted average of teacher predictions per bin
        # (N, F, K, 2)
        weighted_probs = membership_expanded * teacher_probs_expanded
        
        # Sum over samples to get bin-wise predictions
        bin_probs = weighted_probs.sum(dim=0)  # (F, K, 2)
        bin_counts = membership.sum(dim=0).unsqueeze(-1) + 1e-8  # (F, K, 1)
        
        # Average prediction per bin
        avg_bin_probs = bin_probs / bin_counts  # (F, K, 2)
        
        # Compute entropy of each bin (lower entropy = more pure)
        eps = 1e-8
        avg_bin_probs_safe = torch.clamp(avg_bin_probs, eps, 1.0)
        bin_entropy = -(avg_bin_probs_safe * torch.log(avg_bin_probs_safe + eps)).sum(dim=-1)  # (F, K)
        
        # Loss: maximize purity (minimize entropy)
        # Weight by bin size to focus on populated bins
        bin_weights = membership.sum(dim=0) / X_train_t.shape[0]  # (F, K)
        purity_loss = (bin_entropy * bin_weights).sum()
        
        # Add within-bin entropy minimization: samples in same bin should have similar predictions
        # For each sample, compute entropy of its prediction across all bins it belongs to
        
        # membership: (N, F, K)
        # teacher_probs: (N, 2)
        
        # For each feature and bin, compute the variance of teacher predictions
        # This encourages samples in the same bin to have similar teacher predictions
        
        # Expand teacher_probs for broadcasting: (N, 1, 1, 2)
        teacher_probs_exp = teacher_probs.unsqueeze(1).unsqueeze(2)
        
        # For each bin, compute weighted variance of teacher predictions
        # membership: (N, F, K), teacher_probs: (N, 2)
        
        # Compute mean prediction per bin (already computed above)
        # avg_bin_probs: (F, K, 2)
        
        # For each sample, compute squared difference from bin mean
        # (N, F, K, 2) = (N, 1, 1, 2) - (1, F, K, 2)
        pred_diff = teacher_probs_exp - avg_bin_probs.unsqueeze(0)  # (N, F, K, 2)
        pred_variance = (pred_diff ** 2).sum(dim=-1)  # (N, F, K)
        
        # Weight by membership (samples contribute more to bins they belong to)
        weighted_variance = pred_variance * membership  # (N, F, K)
        
        # Sum over samples and normalize by bin counts
        bin_variance = weighted_variance.sum(dim=0) / (membership.sum(dim=0) + eps)  # (F, K)
        
        # Within-bin entropy loss: minimize variance (encourage similarity)
        within_bin_entropy_loss = (bin_variance * bin_weights).sum()
        
        # Total loss
        total_loss = purity_loss + 1.0 * within_bin_entropy_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(bin_learner.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Purity Loss: {purity_loss.item():.4f} | "
                  f"Within-Bin Entropy Loss: {within_bin_entropy_loss.item():.4f} | "
                  f"Total Loss: {total_loss.item():.4f}")
    
    # Get final boundaries
    bin_learner.eval()
    with torch.no_grad():
        final_boundaries = bin_learner.get_boundaries()
    
    print("\n✓ Bin learner training complete!")
    
    return {
        'bin_learner': bin_learner,
        'initial_boundaries': initial_boundaries,
        'final_boundaries': final_boundaries
    }


def run_bin_analysis_experiment(dataset, num_bins=5, temperature=1.0, device='cpu'):
    """
    Run the bin analysis experiment.
    
    Args:
        dataset (dict): Dataset dictionary from load_and_prepare_dataset()
        num_bins (int): Number of bins per feature
        temperature (float): Initial temperature for bin learner
        device (str): Device to run on ('cpu' or 'cuda')
    
    Returns:
        dict: Results containing bin_learner, boundaries, membership, bin_counts
    """
    X_train = dataset['X_train']
    feature_names = dataset['feature_names']
    feature_ranges = dataset['feature_ranges']
    
    # ============================================================================
    # INITIALIZE BIN LEARNER
    # ============================================================================
    print("\n" + "="*70)
    print("INITIALIZING BIN LEARNER")
    print("="*70)
    
    bin_learner = BinLearner(
        num_features=30,
        num_bins=num_bins,
        feature_ranges=feature_ranges,
        temperature=temperature
    ).to(device)
    
    print(f"Number of bins per feature: {num_bins}")
    print(f"Temperature: {temperature}")
    
    # ============================================================================
    # GET INITIAL BOUNDARIES
    # ============================================================================
    print("\n" + "="*70)
    print("INITIAL BIN BOUNDARIES (BEFORE TRAINING)")
    print("="*70)
    
    bin_learner.eval()
    with torch.no_grad():
        boundaries = bin_learner.get_boundaries()  # (30, 6)
    
    print(f"Boundaries shape: {boundaries.shape}")
    print(f"\nShowing first 5 features:")
    for i in range(5):
        bounds = boundaries[i].cpu().numpy()
        print(f"\nFeature {i}: {feature_names[i]}")
        print(f"  Boundaries: {[f'{b:.2f}' for b in bounds]}")
        print(f"  Bins:")
        for j in range(num_bins):
            print(f"    Bin {j}: [{bounds[j]:.2f}, {bounds[j+1]:.2f})")
    
    # ============================================================================
    # COMPUTE BIN MEMBERSHIPS FOR TRAIN DATA
    # ============================================================================
    print("\n" + "="*70)
    print("COMPUTING BIN MEMBERSHIPS FOR TRAIN DATA")
    print("="*70)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    
    bin_learner.eval()
    with torch.no_grad():
        membership = bin_learner(X_train_t)  # (N, 30, 5)
    
    print(f"Membership shape: {membership.shape}")
    print(f"  N (samples): {membership.shape[0]}")
    print(f"  F (features): {membership.shape[1]}")
    print(f"  K (bins): {membership.shape[2]}")
    
    # ============================================================================
    # ANALYZE BIN DISTRIBUTIONS
    # ============================================================================
    print("\n" + "="*70)
    print("BIN DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Sum membership across all samples to get bin counts
    bin_counts = membership.sum(dim=0)  # (30, 5)
    
    print(f"\nShowing first 5 features:")
    for i in range(5):
        counts = bin_counts[i].cpu().numpy()
        total = counts.sum()
        percentages = (counts / total) * 100
        
        print(f"\nFeature {i}: {feature_names[i]}")
        print(f"  Total samples: {total:.1f}")
        for j in range(num_bins):
            print(f"    Bin {j}: {counts[j]:6.1f} samples ({percentages[j]:5.1f}%)")
    
    # ============================================================================
    # VISUALIZE BIN SPLITS FOR FIRST 3 FEATURES
    # ============================================================================
    print("\n" + "="*70)
    print("VISUALIZING BIN SPLITS")
    print("="*70)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for i in range(3):
        ax = axes[i]
        
        # Get feature values
        feature_values = X_train[:, i]
        
        # Get boundaries
        bounds = boundaries[i].cpu().numpy()
        
        # Plot histogram
        ax.hist(feature_values, bins=50, alpha=0.6, color='skyblue', edgecolor='black')
        
        # Plot bin boundaries
        for j, bound in enumerate(bounds):
            if j == 0:
                ax.axvline(bound, color='red', linestyle='--', linewidth=2, label='Bin Boundaries')
            else:
                ax.axvline(bound, color='red', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Feature Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Feature {i}: {feature_names[i]}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add bin labels
        for j in range(num_bins):
            mid_point = (bounds[j] + bounds[j+1]) / 2
            count = bin_counts[i, j].item()
            ax.text(mid_point, ax.get_ylim()[1] * 0.9, 
                    f'Bin {j}\n{count:.0f}', 
                    ha='center', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('bin_splits_visualization.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved to 'bin_splits_visualization.png'")
    
    # ============================================================================
    # COMPUTE SAMPLE ASSIGNMENTS (HARD BINS)
    # ============================================================================
    print("\n" + "="*70)
    print("SAMPLE BIN ASSIGNMENTS (HARD BINS)")
    print("="*70)
    
    # Get hard bin assignments (argmax of membership)
    hard_bins = torch.argmax(membership, dim=2)  # (N, 30)
    
    print(f"\nShowing first 10 samples for first 3 features:")
    print(f"{'Sample':>8} | {'Feature 0':>10} | {'Feature 1':>10} | {'Feature 2':>10}")
    print("-" * 60)
    
    for i in range(10):
        f0_bin = hard_bins[i, 0].item()
        f1_bin = hard_bins[i, 1].item()
        f2_bin = hard_bins[i, 2].item()
        
        print(f"{i:8d} | Bin {f0_bin:1d} ({X_train[i, 0]:6.2f}) | "
              f"Bin {f1_bin:1d} ({X_train[i, 1]:6.2f}) | "
              f"Bin {f2_bin:1d} ({X_train[i, 2]:6.2f})")
    
    # ============================================================================
    # SUMMARY STATISTICS
    # ============================================================================
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nBin Statistics:")
    print(f"  Minimum samples in any bin: {bin_counts.min().item():.1f}")
    print(f"  Maximum samples in any bin: {bin_counts.max().item():.1f}")
    print(f"  Average samples per bin:     {bin_counts.mean().item():.1f}")
    print(f"  Std dev of samples per bin:  {bin_counts.std().item():.1f}")
    
    # Check for empty bins
    empty_bins = (bin_counts < 0.1).sum().item()
    print(f"\n  Empty bins (count < 0.1): {empty_bins} / {30 * num_bins}")
    
    # Check uniformity (coefficient of variation)
    cv_per_feature = bin_counts.std(dim=1) / (bin_counts.mean(dim=1) + 1e-8)
    avg_cv = cv_per_feature.mean().item()
    print(f"\n  Average CV (lower = more uniform): {avg_cv:.4f}")
    
    print("\n" + "="*70)
    print("✓ BIN ANALYSIS COMPLETE!")
    print("="*70)
    
    return {
        'bin_learner': bin_learner,
        'boundaries': boundaries,
        'membership': membership,
        'bin_counts': bin_counts,
        'hard_bins': hard_bins
    }


def compare_boundaries(initial_boundaries, final_boundaries, feature_names, num_features_to_show=5):
    """
    Compare initial and final bin boundaries.
    
    Args:
        initial_boundaries (torch.Tensor): Initial boundaries (F, K+1)
        final_boundaries (torch.Tensor): Final boundaries (F, K+1)
        feature_names (list): List of feature names
        num_features_to_show (int): Number of features to display
    """
    print("\n" + "="*70)
    print("BOUNDARY COMPARISON: INITIAL vs FINAL")
    print("="*70)
    
    initial_np = initial_boundaries.cpu().numpy()
    final_np = final_boundaries.cpu().numpy()
    
    # Compute changes
    boundary_changes = np.abs(final_np - initial_np)
    max_change_per_feature = boundary_changes.max(axis=1)
    avg_change_per_feature = boundary_changes.mean(axis=1)
    
    print(f"\nShowing first {num_features_to_show} features:\n")
    
    for i in range(num_features_to_show):
        print(f"Feature {i}: {feature_names[i]}")
        print(f"  Initial: {[f'{b:.2f}' for b in initial_np[i]]}")
        print(f"  Final:   {[f'{b:.2f}' for b in final_np[i]]}")
        print(f"  Max change: {max_change_per_feature[i]:.4f}")
        print(f"  Avg change: {avg_change_per_feature[i]:.4f}\n")
    
    # Overall statistics
    print(f"Overall Statistics:")
    print(f"  Max boundary change across all features: {boundary_changes.max():.4f}")
    print(f"  Average boundary change: {boundary_changes.mean():.4f}")
    print(f"  Std dev of changes: {boundary_changes.std():.4f}")
    
    # Visualize changes for first 3 features
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for i in range(3):
        ax = axes[i]
        
        num_boundaries = initial_np.shape[1]
        x_pos = np.arange(num_boundaries)
        width = 0.35
        
        ax.bar(x_pos - width/2, initial_np[i], width, label='Initial', alpha=0.7, color='skyblue')
        ax.bar(x_pos + width/2, final_np[i], width, label='Final', alpha=0.7, color='orange')
        
        ax.set_xlabel('Boundary Index', fontsize=12)
        ax.set_ylabel('Boundary Value', fontsize=12)
        ax.set_title(f'Feature {i}: {feature_names[i]}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('boundary_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Boundary comparison visualization saved to 'boundary_comparison.png'")


def main():
    """
    Main function to run the bin analysis experiment.
    """
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load and prepare dataset
    dataset = load_and_prepare_dataset()
    
    # Train teacher model
    teacher = train_teacher_model(dataset, device=device, epochs=100)
    
    # Train bin learner with teacher signals
    bin_results = train_bin_learner_with_teacher(
        dataset=dataset,
        teacher=teacher,
        num_bins=5,
        temperature=1.0,
        device=device,
        epochs=100,
        lr=0.01
    )
    
    # Compare initial and final boundaries
    compare_boundaries(
        initial_boundaries=bin_results['initial_boundaries'],
        final_boundaries=bin_results['final_boundaries'],
        feature_names=dataset['feature_names'],
        num_features_to_show=5
    )
    
    # Run final bin analysis with trained bin learner
    print("\n" + "="*70)
    print("FINAL BIN ANALYSIS WITH TRAINED BOUNDARIES")
    print("="*70)
    
    X_train = dataset['X_train']
    feature_names = dataset['feature_names']
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    
    bin_learner = bin_results['bin_learner']
    bin_learner.eval()
    
    with torch.no_grad():
        membership = bin_learner(X_train_t)
        boundaries = bin_results['final_boundaries']
    
    # Analyze final bin distributions
    bin_counts = membership.sum(dim=0)
    
    print(f"\nShowing first 5 features:")
    for i in range(5):
        counts = bin_counts[i].cpu().numpy()
        total = counts.sum()
        percentages = (counts / total) * 100
        
        print(f"\nFeature {i}: {feature_names[i]}")
        print(f"  Total samples: {total:.1f}")
        for j in range(5):
            print(f"    Bin {j}: {counts[j]:6.1f} samples ({percentages[j]:5.1f}%)")
    
    # Visualize final bins
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for i in range(3):
        ax = axes[i]
        
        feature_values = X_train[:, i]
        bounds = boundaries[i].cpu().numpy()
        
        ax.hist(feature_values, bins=50, alpha=0.6, color='skyblue', edgecolor='black')
        
        for j, bound in enumerate(bounds):
            if j == 0:
                ax.axvline(bound, color='red', linestyle='--', linewidth=2, label='Learned Boundaries')
            else:
                ax.axvline(bound, color='red', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Feature Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Feature {i}: {feature_names[i]} (After Training)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        for j in range(5):
            mid_point = (bounds[j] + bounds[j+1]) / 2
            count = bin_counts[i, j].item()
            ax.text(mid_point, ax.get_ylim()[1] * 0.9, 
                    f'Bin {j}\n{count:.0f}', 
                    ha='center', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('final_bin_splits_visualization.png', dpi=300, bbox_inches='tight')
    print("\n✓ Final visualization saved to 'final_bin_splits_visualization.png'")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*70)
    
    return {
        'teacher': teacher,
        'bin_learner': bin_learner,
        'initial_boundaries': bin_results['initial_boundaries'],
        'final_boundaries': bin_results['final_boundaries'],
        'membership': membership,
        'bin_counts': bin_counts
    }


if __name__ == "__main__":
    main()