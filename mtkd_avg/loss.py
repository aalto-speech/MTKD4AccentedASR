import torch
import torch.nn.functional as F
from torch import nn


class LossWithLearnableWeights(nn.Module):
    """
    Custom loss function for Multi-Task or Knowledge Distillation-based training with learnable weights.

    This class:
    - Combines CTC loss and KL divergence loss (between student and teacher logits).
    - Learns two weights (α and β) to dynamically balance these two losses during training.
    - Applies constraints to prevent either weight from collapsing below a minimum threshold (e.g., 0.1).
    - Ensures that the weights α and β sum to 1 using softmax over learnable logits.

    Attributes:
        logits (nn.Parameter): A 2-element parameter tensor used to compute softmax weights α and β.
    """
    
    
    
    def __init__(self):
        """
        Initializes the learnable logit parameters for computing weighted loss.
        """
        super(LossWithLearnableWeights, self).__init__()
        self.logits = nn.Parameter(torch.tensor([0.0, 0.0]))  # Two learnable params
        
        
        

    def forward(self, ctc_loss, student_logits, teacher_logits):
        """
        Computes the combined loss using CTC loss and KL divergence loss.

        Args:
            ctc_loss (torch.Tensor): The CTC loss computed between model predictions and ground truth labels.
            student_logits (torch.Tensor): The output logits from the student model (shape: [T, B, V]).
            teacher_logits (torch.Tensor): The averaged logits from teacher models (shape: [T, B, V]).

        Returns:
            tuple:
                - total_loss (torch.Tensor): The combined loss: α * ctc_loss + β * kl_loss.
                - alpha (torch.Tensor): The weight applied to CTC loss.
                - beta (torch.Tensor): The weight applied to KL divergence loss.
                - ctc_loss (torch.Tensor): The original CTC loss (unchanged).
                - kl_loss (torch.Tensor): The computed KL divergence loss.
        """
        
        # KL Divergence Loss
        # kl_loss = F.kl_div(
        #     F.log_softmax(student_logits, dim=-1),
        #     F.softmax(teacher_logits, dim=-1),
        #     reduction='batchmean'
        # )

        mask = teacher_logits.abs().sum(dim=-1) != 0 
        
        student_probs = F.log_softmax(student_logits, dim=-1)  # Student (log probs)
        
        teacher_probs = F.softmax(teacher_logits, dim=-1)  # Teacher (probabilities)
        # teacher_probs = F.log_softmax(teacher_logits, dim=-1)
        
        kl_loss = F.kl_div(
            student_probs, 
            teacher_probs, 
            reduction='none',
            # log_target = True
        )  # (seq_len, batch, vocab_size)
        
        kl_loss = kl_loss.sum(dim=-1) 
        
        kl_loss = (kl_loss * mask).sum() / mask.sum()
        
        # Softmax to constrain alpha + beta = 1
        alpha, beta = F.softmax(self.logits, dim=0)
        
        # Apply regularization: if any value is below 0.1, enforce the constraint
        min_val = 0.1
        if alpha < min_val:
            alpha = torch.tensor(min_val, device=self.logits.device, dtype=self.logits.dtype)
            beta = torch.tensor(1 - min_val, device=self.logits.device, dtype=self.logits.dtype)
        elif beta < min_val:
            beta = torch.tensor(min_val, device=self.logits.device, dtype=self.logits.dtype)
            alpha = torch.tensor(1 - min_val, device=self.logits.device, dtype=self.logits.dtype)

        
        # alpha = torch.tensor(0.0)
        # beta = torch.tensor(1.0)

        total_loss = alpha * ctc_loss + beta * kl_loss
        return total_loss, alpha, beta, ctc_loss, kl_loss
    