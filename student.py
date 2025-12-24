import torch
import torch.nn as nn

class StudentMLP(nn.Module):
    """
    Lightweight student model for deployment.
    
    Key Design:
    - Small capacity (1-2 hidden layers)
    - Trains only on synthetic data
    """
    def __init__(self, input_dim=30, hidden_dim=32, output_dim=2):
        super(StudentMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: (Batch, Features)
        
        Returns:
            logits: (Batch, Classes)
        """
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits
    
    def predict_proba(self, x):
        """
        Returns probability distribution.
        
        Args:
            x: (Batch, Features)
        
        Returns:
            probs: (Batch, Classes)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)