import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(nn.Module):
    """
    Generates synthetic samples for training.
    
    Key Design:
    - No scaling (outputs raw values)
    - No reality anchoring
    - Purely adversarial + diversity objectives
    """
    def __init__(self, latent_dim=100, num_features=30, hidden_dims=[128, 64]):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_features = num_features
        
        # Build network
        layers = []
        input_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))  # Add normalization
            input_dim = hidden_dim
        
        # Output layer (raw values, no activation)
        layers.append(nn.Linear(input_dim, num_features))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, z):
        """
        Args:
            z (Tensor): Noise, shape (Batch, latent_dim)
        
        Returns:
            x (Tensor): Raw features, shape (Batch, num_features)
        """
        return self.network(z)


def pairwise_entropy_loss(membership, num_bins=5, epsilon=1e-8):
    """
    Compute 2-way interaction diversity using pairwise entropy.
    
    Goal: Maximize entropy = encourage uniform distribution across K×K bin grid
    
    Args:
        membership: (Batch, Features, Bins)
        num_bins: K
        epsilon: Small constant for numerical stability
    
    Returns:
        loss: Scalar (lower = better diversity)
    """
    batch_size, num_features, _ = membership.shape
    
    # Maximum entropy for uniform distribution over K×K grid
    max_entropy = np.log(num_bins * num_bins)
    
    total_entropy = 0.0
    num_pairs = 0
    
    for i in range(num_features):
        for j in range(i + 1, num_features):
            # Get memberships for features i and j
            m_i = membership[:, i, :]  # (Batch, K)
            m_j = membership[:, j, :]  # (Batch, K)
            
            # Joint distribution via outer product
            joint = m_i.unsqueeze(2) * m_j.unsqueeze(1)  # (Batch, K, K)
            
            # Marginalize over batch
            joint_dist = joint.sum(dim=0)  # (K, K)
            
            # Normalize to probability with epsilon
            joint_sum = joint_dist.sum()
            if joint_sum < epsilon:
                continue  # Skip if empty
            
            joint_prob = joint_dist / (joint_sum + epsilon)
            
            # Clamp probabilities to avoid log(0)
            joint_prob = torch.clamp(joint_prob, min=epsilon, max=1.0)
            
            # Compute entropy
            entropy = -(joint_prob * torch.log(joint_prob)).sum()
            
            total_entropy += entropy
            num_pairs += 1
    
    if num_pairs == 0:
        return torch.tensor(0.0, device=membership.device)
    
    # Average entropy
    avg_entropy = total_entropy / num_pairs
    
    # Loss: penalize deviation from max entropy
    diversity_loss = max_entropy - avg_entropy
    
    return diversity_loss


def generator_loss(student_probs, teacher_probs, membership, epsilon=1e-8):
    """
    Generator objective: Adversarial Hardness + Interaction Diversity
    
    Args:
        student_probs: (Batch, Classes)
        teacher_probs: (Batch, Classes)
        membership: (Batch, Features, Bins)
        epsilon: Small constant for numerical stability
    
    Returns:
        total_loss: Scalar
        loss_adv: Adversarial component
        loss_div: Diversity component
    """
    # Clamp probabilities to avoid numerical issues
    teacher_probs = torch.clamp(teacher_probs, min=epsilon, max=1.0)
    student_probs = torch.clamp(student_probs, min=epsilon, max=1.0)
    
    # Loss 1: Adversarial Hardness
    # Maximize KL(Teacher || Student) = make student fail
    kl_div = (teacher_probs * torch.log(teacher_probs / student_probs)).sum(dim=1).mean()
    
    # Clamp to avoid NaN
    kl_div = torch.clamp(kl_div, min=-10.0, max=10.0)
    
    loss_adversarial = -kl_div  # Negative because we want to maximize
    
    # Loss 2: Interaction Diversity
    loss_diversity = pairwise_entropy_loss(membership)
    
    # Combined loss (equal weights initially)
    total_loss = loss_adversarial + loss_diversity
    
    # Check for NaN
    if torch.isnan(total_loss):
        print("WARNING: NaN detected in generator loss!")
        total_loss = torch.tensor(0.0, device=total_loss.device, requires_grad=True)
    
    return total_loss, loss_adversarial, loss_diversity