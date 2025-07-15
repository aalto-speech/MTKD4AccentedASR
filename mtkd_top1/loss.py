import torch
import torch.nn.functional as F
from torch import nn



class LossWithLearnableWeights(nn.Module):
    """
    Combines a CTC loss with multiple KL divergence losses from teacher models,
    with learnable weights for dynamically adjusting their contributions during training.

    Attributes:
        alphabeta (nn.Parameter): Learnable parameters (2 values) for weighting the CTC and KL losses.
        logitspriority (nn.Parameter): Learnable parameters (10 values) representing importance weights 
                                       for each of the 10 teacher models.
    """
    
    
    def __init__(self):
        super(LossWithLearnableWeights, self).__init__()
        self.alphabeta = nn.Parameter(torch.zeros(2), requires_grad=True)
        self.logitspriority = nn.Parameter(torch.zeros(10), requires_grad=True)
        
        
        
    def forward(self, ctc_loss, student_logits, teachers_logits):
        """
        Computes the weighted sum of CTC and KL losses.

        Args:
            ctc_loss (Tensor): Precomputed CTC loss (scalar).
            student_logits (Tensor): Logits predicted by the student model, shape (batch_size, seq_len, vocab_size).
            teachers_logits (List[Tensor]): List of teacher logits tensors, each with the same shape as student_logits.

        Returns:
            total_loss (Tensor): Combined total loss (CTC + KL) with learned weights.
            alpha (float): Final weight for the CTC loss.
            beta (float): Final weight for the KL loss.
            normalized_kl_weights (Tensor): Softmax-normalized weights for each teacher model.
            ctc_loss (Tensor): Original CTC loss.
            kl_loss (Tensor): Combined weighted KL loss.
        """
        
        kl_losses = []

        for teacher_logits in teachers_logits:
        
            mask = teacher_logits.abs().sum(dim=-1) != 0 
        
            student_probs = F.log_softmax(student_logits, dim=-1)  
        
            teacher_probs = F.softmax(teacher_logits, dim=-1)  
        
            kl_loss = F.kl_div(
                student_probs, 
                teacher_probs, 
                reduction='none'
            )  
        
            kl_loss = kl_loss.sum(dim=-1) 
            kl_loss = (kl_loss * mask).sum() / mask.sum()
            
            kl_losses.append(kl_loss)
            
        kl_losses = torch.stack(kl_losses)  # Shape: (10,)
        
        max_index = torch.argmax(self.logitspriority)
        normalized_kl_weights = torch.zeros_like(self.logitspriority)
        normalized_kl_weights[max_index] = 1.0
        
        kl_loss = torch.sum(normalized_kl_weights * kl_losses)
        
        # Softmax to constrain alpha + beta = 1
        alpha, beta = F.softmax(self.alphabeta, dim=0)
        
        # Apply regularization: if any value is below 0.1, enforce the constraint
        min_val = 0.1
        if alpha < min_val:
            alpha = torch.tensor(min_val, device=self.alphabeta.device, dtype=self.alphabeta.dtype)
            beta = torch.tensor(1 - min_val, device=self.alphabeta.device, dtype=self.alphabeta.dtype)
        elif beta < min_val:
            beta = torch.tensor(min_val, device=self.alphabeta.device, dtype=self.alphabeta.dtype)
            alpha = torch.tensor(1 - min_val, device=self.alphabeta.device, dtype=self.alphabeta.dtype)

        total_loss = alpha * ctc_loss + beta * kl_loss
        
        return total_loss, alpha, beta, normalized_kl_weights, ctc_loss, kl_loss