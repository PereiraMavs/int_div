import torch
import torch.nn as nn
import numpy as np

class TeacherNet(nn.Module):
    """
    Teacher network with higher capacity.
    Architecture: input -> 128 -> 64 -> num_classes
    """
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
    """
    Lightweight student network for efficient deployment.
    Architecture: input -> 32 -> num_classes
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    """
    Generator network for adversarial data synthesis.
    Outputs are clamped to valid feature ranges.
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
    """
    Learnable bin boundaries with soft membership via temperature-annealed sigmoid.
    """
    def __init__(self, num_features, num_bins, min_vals, max_vals):
        super().__init__()
        self.num_bins = num_bins
        self.temperature = 1.0  # Annealed during training
        
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
        widths = torch.nn.functional.softplus(self.raw_widths)
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


class XGBoostTeacherWrapper(nn.Module):
    """
    Wrapper to make XGBoost compatible with PyTorch training pipeline.
    Returns probabilities (not logits) since XGBoost uses predict_proba().
    """
    def __init__(self, xgb_model, device='cpu'):
        super().__init__()
        self.model = xgb_model
        self.device = device
        self._is_trained = False
        
    def forward(self, x):
        """
        Convert PyTorch tensor -> NumPy -> XGBoost prediction -> PyTorch tensor
        Returns: (N, num_classes) probability tensor
        """
        # Convert to numpy on CPU
        if isinstance(x, torch.Tensor):
            x_numpy = x.detach().cpu().numpy()
        else:
            x_numpy = x
        
        # Get probabilities from XGBoost
        probs = self.model.predict_proba(x_numpy)
        
        # Convert back to tensor on correct device
        return torch.tensor(probs, dtype=torch.float32, device=self.device)
    
    def train(self, mode=True):
        """Override train() - XGBoost doesn't have training mode"""
        # Keep PyTorch happy but don't do anything
        return self
    
    def eval(self):
        """Override eval() - XGBoost doesn't have eval mode"""
        # Keep PyTorch happy but don't do anything
        return self


class RandomForestTeacherWrapper(nn.Module):
    """
    Wrapper to make Random Forest compatible with PyTorch training pipeline.
    Returns probabilities (not logits) since sklearn uses predict_proba().
    """
    def __init__(self, rf_model, device='cpu'):
        super().__init__()
        self.model = rf_model
        self.device = device
        
    def forward(self, x):
        """
        Convert PyTorch tensor -> NumPy -> Random Forest prediction -> PyTorch tensor
        Returns: (N, num_classes) probability tensor
        """
        # Convert to numpy on CPU
        if isinstance(x, torch.Tensor):
            x_numpy = x.detach().cpu().numpy()
        else:
            x_numpy = x
        
        # Get probabilities from Random Forest
        probs = self.model.predict_proba(x_numpy)
        
        # Convert back to tensor on correct device
        return torch.tensor(probs, dtype=torch.float32, device=self.device)
    
    def train(self, mode=True):
        """Override train() - Random Forest doesn't have training mode"""
        # Keep PyTorch happy but don't do anything
        return self
    
    def eval(self):
        """Override eval() - Random Forest doesn't have eval mode"""
        # Keep PyTorch happy but don't do anything
        return self