import torch
import torch.nn as nn

class DispersionLoss(nn.Module):
    def __init__(self, lambda_entropy=0.1, lambda_repulsion=0.5, lambda_inter_dispersion=0.3, temperature=1.0):
        super().__init__()
        self.lambda_entropy = lambda_entropy
        self.lambda_repulsion = lambda_repulsion
        self.lambda_inter_dispersion = lambda_inter_dispersion
        self.temperature = temperature  # Temperature for exponential penalties

    def forward(self, membership, teacher_preds):
        """
        Args:
            membership: (Batch, Features, Bins) - From BinLearner
            teacher_preds: (Batch, Classes) - From Teacher
        """
        # Dimensions
        N, F, K = membership.shape
        _, C = teacher_preds.shape
        epsilon = 1e-8

        # --- 1. PREPARE DATA FOR BROADCASTING ---
        # We want to analyze every Feature independently against the Teacher.
        # Membership: (N, F, K, 1)
        m = membership.unsqueeze(3)
        
        # Teacher: (N, 1, 1, C) - Broadcast across Features and Bins
        y = teacher_preds.unsqueeze(1).unsqueeze(1)
        
        # --- 2. CALCULATE BIN MASS ---
        # Sum over batch (N)
        # Shape: (F, K, 1)
        bin_mass = m.sum(dim=0) + epsilon

        # --- 3. CALCULATE CENTROIDS (Mu) ---
        # Weighted sum of teacher preds
        # (N, F, K, 1) * (N, 1, 1, C) -> (N, F, K, C)
        weighted_y = m * y
        
        # Sum over batch -> (F, K, C)
        sum_weighted_y = weighted_y.sum(dim=0)
        
        # Divide by mass -> (F, K, C)
        centroids = sum_weighted_y / bin_mass
        
        # --- 4. CALCULATE DISPERSION (Intra-Bin Variance) ---
        # Distance: y - centroid
        # (N, 1, 1, C) - (1, F, K, C) -> (N, F, K, C)
        diff = y - centroids.unsqueeze(0)
        
        # Squared Euclidean Distance summed over Classes
        # -> (N, F, K)
        dist_sq = diff.pow(2).sum(dim=3)
        
        # Weighted Variance: m * dist / mass
        # We perform the weighting using the original membership matrix (N, F, K)
        # Sum over N, then divide by mass (F, K)
        # Result -> (F, K)
        weighted_variance = (membership * dist_sq).sum(dim=0) / bin_mass.squeeze(2)
        
        # Sum over all Features and Bins to get scalar loss
        loss_dispersion = weighted_variance.sum()

        # --- 5. ENTROPY REGULARIZATION (Balance) ---
        # Normalize mass to get probabilities: p = mass / N
        p_bins = bin_mass.squeeze(2) / N
        loss_entropy = (p_bins * torch.log(p_bins + epsilon)).sum()
        
        # --- 6. INTER-BIN REPULSION (Neighbor Boundaries) ---
        # With temperature: exp(-dist / T)
        # Higher T → softer penalty (less sensitive to small distances)
        # Lower T → sharper penalty (more sensitive)
        curr_bins = centroids[:, :-1, :]
        next_bins = centroids[:, 1:, :]
        
        # Squared dist between neighbors -> (F, K-1)
        neighbor_dist = (curr_bins - next_bins).pow(2).sum(dim=2)
        loss_repulsion = torch.exp(-neighbor_dist / self.temperature).sum()

        # --- 7. INTER-BIN DISPERSION (Maximize All Pairwise Distances) ---
        loss_inter_dispersion = 0.0
        
        for f in range(F):
            feat_centroids = centroids[f, :, :]  # (K, C)
            
            # Compute pairwise squared distances
            sq_norms = (feat_centroids ** 2).sum(dim=1)
            dot_products = torch.mm(feat_centroids, feat_centroids.t())
            pairwise_dist = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2 * dot_products
            
            # Upper triangle mask (avoid diagonal and double counting)
            mask = torch.triu(torch.ones(K, K, device=pairwise_dist.device), diagonal=1)
            
            # Penalty with temperature: exp(-dist / T)
            # Higher T → softer penalty (allows closer bins)
            # Lower T → sharper penalty (forces bins further apart)
            penalty = torch.exp(-pairwise_dist / self.temperature) * mask
            loss_inter_dispersion += penalty.sum()
        
        loss_inter_dispersion = loss_inter_dispersion / F

        # --- FINAL SUM ---
        total_loss = loss_dispersion + \
                     (self.lambda_entropy * loss_entropy) + \
                     (self.lambda_repulsion * loss_repulsion) + \
                     (self.lambda_inter_dispersion * loss_inter_dispersion)
                     
        return total_loss, {
            "dispersion": loss_dispersion.item(),
            "entropy": loss_entropy.item(),
            "repulsion": loss_repulsion.item(),
            "inter_dispersion": loss_inter_dispersion.item()
        }