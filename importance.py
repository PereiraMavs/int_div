import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import torch
import torch.nn as nn

# ==========================================
# 1. LOAD DATA AND TEACHER MODEL
# ==========================================
print("Loading data and teacher model...")

# Load dataset
data = load_breast_cancer()
X_raw = data.data
y_raw = data.target

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

# Convert to tensors
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)

DATA_DIM = X_raw.shape[1]
NUM_CLASSES = len(np.unique(y_raw))

# Define teacher architecture (must match saved model)
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

# Load teacher model
teacher = TeacherNet(DATA_DIM, NUM_CLASSES).to(DEVICE)
checkpoint = torch.load(f'teacher_model_{DATA_DIM}features_{NUM_CLASSES}classes.pt')
teacher.load_state_dict(checkpoint['model_state_dict'])
teacher.eval()

print(f"✓ Teacher loaded - Test Accuracy: {checkpoint['test_acc']*100:.2f}%")
print(f"✓ Dataset: {DATA_DIM} features, {NUM_CLASSES} classes")
print(f"✓ Feature names: {data.feature_names}\n")


# ==========================================
# 2. METHOD 1: GRADIENT-BASED IMPORTANCE
# ==========================================
print("=" * 70)
print("METHOD 1: GRADIENT-BASED FEATURE IMPORTANCE")
print("=" * 70)

def compute_gradient_importance(model, X, y):
    """Compute importance as mean absolute gradient w.r.t. inputs"""
    model.eval()
    X_var = X.clone().requires_grad_(True)
    
    output = model(X_var)
    # For correct class predictions
    loss = F.cross_entropy(output, torch.LongTensor(y).to(DEVICE))
    loss.backward()
    
    # Mean absolute gradient per feature
    gradients = X_var.grad.abs().mean(dim=0).cpu().numpy()
    return gradients

grad_importance = compute_gradient_importance(teacher, X_test_tensor, y_test)
grad_importance_norm = grad_importance / grad_importance.sum()

print("\nTop 10 Most Important Features (Gradient-based):")
top_indices = np.argsort(grad_importance_norm)[::-1][:10]
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank:2d}. Feature {idx:2d} ({data.feature_names[idx].item():30s}): {grad_importance_norm[idx]:.4f}")


# ==========================================
# 3. METHOD 2: PERMUTATION IMPORTANCE
# ==========================================
print("\n" + "=" * 70)
print("METHOD 2: PERMUTATION FEATURE IMPORTANCE")
print("=" * 70)

def compute_permutation_importance(model, X, y, n_repeats=10):
    """Compute importance by measuring accuracy drop when feature is shuffled"""
    from sklearn.metrics import accuracy_score
    
    model.eval()
    with torch.no_grad():
        baseline_preds = torch.argmax(model(X), dim=1).cpu().numpy()
        baseline_acc = accuracy_score(y, baseline_preds)
    
    importances = np.zeros(X.shape[1])
    
    for feat_idx in range(X.shape[1]):
        acc_drops = []
        for _ in range(n_repeats):
            X_permuted = X.clone()
            # Shuffle this feature
            perm_idx = torch.randperm(X.shape[0])
            X_permuted[:, feat_idx] = X[perm_idx, feat_idx]
            
            with torch.no_grad():
                preds = torch.argmax(model(X_permuted), dim=1).cpu().numpy()
                acc = accuracy_score(y, preds)
                acc_drops.append(baseline_acc - acc)
        
        importances[feat_idx] = np.mean(acc_drops)
    
    return importances, baseline_acc

print("Computing permutation importance (may take a moment)...")
perm_importance, baseline_acc = compute_permutation_importance(
    teacher, X_test_tensor, y_test, n_repeats=10
)
perm_importance_norm = perm_importance / perm_importance.sum()

print(f"\nBaseline Accuracy: {baseline_acc*100:.2f}%")
print("\nTop 10 Most Important Features (Permutation-based):")
top_indices = np.argsort(perm_importance)[::-1][:10]
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank:2d}. Feature {idx:2d} ({data.feature_names[idx].item():30s}): "
          f"{perm_importance[idx]:.4f} (drop: {perm_importance[idx]*100:.2f}%)")


# ==========================================
# 4. METHOD 3: SHAP VALUES
# ==========================================
print("\n" + "=" * 70)
print("METHOD 3: SHAP (SHapley Additive exPlanations)")
print("=" * 70)

# Wrapper for SHAP (expects numpy output)
def model_predict(x):
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x).to(DEVICE)
        logits = teacher(x_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs

print("Computing SHAP values (may take a moment)...")
# Use 100 background samples for faster computation
background = shap.sample(X_train, 100)
explainer = shap.KernelExplainer(model_predict, background)

# Compute SHAP for test set (use subset for speed)
shap_values = explainer.shap_values(X_test[:100])

# For binary classification, SHAP returns values for each class
# Use absolute mean SHAP values across samples and classes
if isinstance(shap_values, list):
    # Binary/multi-class: average absolute SHAP across classes and samples
    shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
else:
    shap_importance = np.abs(shap_values).mean(axis=0)

shap_importance_norm = shap_importance / shap_importance.sum()

print("\nTop 10 Most Important Features (SHAP-based):")
top_indices = np.argsort(shap_importance)[::-1][:10]
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank:2d}. Feature {idx:2d} ({data.feature_names.tolist()[idx]:30s}): {shap_importance_norm[idx]:.4f}")


# ==========================================
# 5. VISUALIZE ALL METHODS
# ==========================================
print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Gradient Importance
ax = axes[0, 0]
sorted_idx = np.argsort(grad_importance_norm)
ax.barh(range(DATA_DIM), grad_importance_norm[sorted_idx])
ax.set_yticks(range(DATA_DIM))
ax.set_yticklabels([data.feature_names[i] for i in sorted_idx], fontsize=8)
ax.set_xlabel('Normalized Importance')
ax.set_title('Gradient-Based Feature Importance')
ax.grid(axis='x', alpha=0.3)

# Plot 2: Permutation Importance
ax = axes[0, 1]
sorted_idx = np.argsort(perm_importance)
ax.barh(range(DATA_DIM), perm_importance[sorted_idx])
ax.set_yticks(range(DATA_DIM))
ax.set_yticklabels([data.feature_names[i] for i in sorted_idx], fontsize=8)
ax.set_xlabel('Accuracy Drop')
ax.set_title('Permutation Feature Importance')
ax.grid(axis='x', alpha=0.3)

# Plot 3: SHAP Importance
ax = axes[1, 0]
sorted_idx = np.argsort(shap_importance)
ax.barh(range(DATA_DIM), shap_importance[sorted_idx])
ax.set_yticks(range(DATA_DIM))
ax.set_yticklabels([data.feature_names[i] for i in sorted_idx], fontsize=8)
ax.set_xlabel('Mean |SHAP Value|')
ax.set_title('SHAP Feature Importance')
ax.grid(axis='x', alpha=0.3)

# Plot 4: Comparison of Top 10 from each method
ax = axes[1, 1]
top_grad = np.argsort(grad_importance_norm)[::-1][:10]
top_perm = np.argsort(perm_importance)[::-1][:10]
top_shap = np.argsort(shap_importance)[::-1][:10]

# Create union of top features
all_top = np.unique(np.concatenate([top_grad, top_perm, top_shap]))
x = np.arange(len(all_top))
width = 0.25

ax.bar(x - width, [grad_importance_norm[i] for i in all_top], width, label='Gradient')
ax.bar(x, [perm_importance_norm[i] for i in all_top], width, label='Permutation')
ax.bar(x + width, [shap_importance_norm[i] for i in all_top], width, label='SHAP')

ax.set_xlabel('Feature Index')
ax.set_ylabel('Normalized Importance')
ax.set_title('Top Features Comparison')
ax.set_xticks(x)
ax.set_xticklabels(all_top)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'teacher_feature_importance_{DATA_DIM}features_{NUM_CLASSES}classes.png', dpi=300)
print(f"\n✓ Plot saved: teacher_feature_importance_{DATA_DIM}features_{NUM_CLASSES}classes.png")


# ==========================================
# 6. SAVE RESULTS
# ==========================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results = {
    'feature_names': data.feature_names,
    'gradient_importance': grad_importance_norm,
    'permutation_importance': perm_importance_norm,
    'shap_importance': shap_importance_norm,
    'baseline_accuracy': baseline_acc
}

np.save(f'teacher_importance_{DATA_DIM}features_{NUM_CLASSES}classes.npy', results)
print(f"✓ Results saved: teacher_importance_{DATA_DIM}features_{NUM_CLASSES}classes.npy")

print("\n✅ Feature importance analysis complete!")