"""
Loss functions and coverage utilities for knowledge distillation.

This module contains:
- Variance-based bin loss
- Interaction diversity loss
- Coverage computation and tracking utilities
- Hard variance calculation for interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            # Note: Using population variance (divides by N) rather than sample variance (N-1)
            # This is acceptable for large K and provides a stable metric
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


def compute_coverage_bonus(membership, cumulative_tracker):
    """
    Reward generator for hitting unexplored bin pairs.
    Uses SOFT assignments for differentiability (allows gradients to flow).
    Returns negative loss (reward) when visiting new bins.
    
    FIXED: Removed torch.no_grad() and uses soft membership probabilities
    instead of hard assignments, enabling gradient flow to generator.
    """
    N, F, K = membership.shape
    
    # Convert boolean tracker to float for differentiable operations
    # Unexplored bins (False) become 1.0, explored bins (True) become 0.0
    unexplored_mask = (~cumulative_tracker).float()  # (F, F, K, K)
    
    # Compute joint probabilities for all samples and feature pairs
    # membership: (N, F, K)
    total_novelty = 0.0
    num_pairs = (F * (F - 1)) // 2
    
    for i in range(F):
        for j in range(i + 1, F):
            # Get membership for features i and j: (N, K) each
            mem_i = membership[:, i, :]  # (N, K)
            mem_j = membership[:, j, :]  # (N, K)
            
            # Compute joint probabilities: (N, K, K)
            joint_probs = mem_i.unsqueeze(2) * mem_j.unsqueeze(1)  # (N, K, K)
            
            # Apply unexplored mask: (K, K) -> broadcast to (N, K, K)
            mask_ij = unexplored_mask[i, j, :, :].unsqueeze(0)  # (1, K, K)
            novelty_ij = (joint_probs * mask_ij).sum(dim=(1, 2))  # (N,)
            
            total_novelty += novelty_ij.sum()
    
    # Average novelty across samples
    avg_novelty = total_novelty / (N * num_pairs)
    
    # Return negative (reward for novelty) - this is differentiable!
    return -avg_novelty
