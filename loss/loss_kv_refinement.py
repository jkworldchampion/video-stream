"""
KV Refinement Loss for Dynamic Attention Adaptation

Supervises Student's K/V refinement to match Teacher's dynamic K/V.

Author: GitHub Copilot (C Enhancement)
Date: 2025-11-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVRefinementLoss(nn.Module):
    """
    Loss to align Student's refined K/V with Teacher's dynamic K/V.
    
    Teacher's K/V change based on temporal window context.
    Student learns to refine static K/V to match Teacher's dynamic behavior.
    
    Args:
        loss_type: 'mse' or 'cosine'
        normalize: Whether to normalize before comparison
        separate_kv: If True, compute K and V losses separately and return both
    """
    
    def __init__(self, loss_type='mse', normalize=True, separate_kv=True):
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
        self.separate_kv = separate_kv
    
    def forward(self, teacher_k, teacher_v, student_k_refined, student_v_refined, mask=None):
        """
        Compute KV refinement loss.
        
        Args:
            teacher_k: Teacher's dynamic K [B, T, C] or [B, T, H, D]
            teacher_v: Teacher's dynamic V [B, T, C] or [B, T, H, D]
            student_k_refined: Student's refined K (same shape)
            student_v_refined: Student's refined V (same shape)
            mask: Optional temporal mask [B, T]
        
        Returns:
            loss: Scalar loss (or dict if separate_kv=True)
        """
        # Flatten attention heads if present [B, T, H, D] -> [B, T, H*D]
        if teacher_k.dim() == 4:
            B, T, H, D = teacher_k.shape
            teacher_k = teacher_k.reshape(B, T, H * D)
            student_k_refined = student_k_refined.reshape(B, T, H * D)
        
        if teacher_v.dim() == 4:
            B, T, H, D = teacher_v.shape
            teacher_v = teacher_v.reshape(B, T, H * D)
            student_v_refined = student_v_refined.reshape(B, T, H * D)
        
        # Normalize if requested
        if self.normalize:
            teacher_k = F.normalize(teacher_k, p=2, dim=-1)
            teacher_v = F.normalize(teacher_v, p=2, dim=-1)
            student_k_refined = F.normalize(student_k_refined, p=2, dim=-1)
            student_v_refined = F.normalize(student_v_refined, p=2, dim=-1)
        
        # Compute losses
        if self.loss_type == 'mse':
            loss_k = F.mse_loss(student_k_refined, teacher_k, reduction='none')  # [B, T, C]
            loss_v = F.mse_loss(student_v_refined, teacher_v, reduction='none')
        elif self.loss_type == 'cosine':
            # Cosine similarity loss: 1 - cos_sim
            loss_k = 1 - F.cosine_similarity(student_k_refined, teacher_k, dim=-1)  # [B, T]
            loss_v = 1 - F.cosine_similarity(student_v_refined, teacher_v, dim=-1)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        # Apply mask if provided
        if mask is not None:
            if loss_k.dim() == 3:  # MSE: [B, T, C]
                mask_expanded = mask.unsqueeze(-1).float()  # [B, T, 1]
                loss_k = (loss_k * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
                loss_v = (loss_v * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
            else:  # Cosine: [B, T]
                mask_float = mask.float()
                loss_k = (loss_k * mask_float).sum() / (mask_float.sum() + 1e-8)
                loss_v = (loss_v * mask_float).sum() / (mask_float.sum() + 1e-8)
        else:
            loss_k = loss_k.mean()
            loss_v = loss_v.mean()
        
        if self.separate_kv:
            return {'k': loss_k, 'v': loss_v, 'total': loss_k + loss_v}
        else:
            return loss_k + loss_v


def test_kv_refinement_loss():
    """Unit test for KV Refinement Loss."""
    print("Testing KVRefinementLoss...")
    
    B, T, C = 2, 16, 192
    
    # Test 1: MSE loss
    criterion_mse = KVRefinementLoss(loss_type='mse', normalize=False, separate_kv=True)
    teacher_k = torch.randn(B, T, C)
    teacher_v = torch.randn(B, T, C)
    student_k = torch.randn(B, T, C)
    student_v = torch.randn(B, T, C)
    
    loss_dict = criterion_mse(teacher_k, teacher_v, student_k, student_v)
    assert 'k' in loss_dict and 'v' in loss_dict and 'total' in loss_dict
    assert loss_dict['total'] > 0
    print(f"✓ Test 1 passed: MSE loss = {loss_dict['total']:.6f}")
    
    # Test 2: Cosine loss
    criterion_cos = KVRefinementLoss(loss_type='cosine', normalize=True, separate_kv=True)
    loss_dict = criterion_cos(teacher_k, teacher_v, student_k, student_v)
    assert 0 <= loss_dict['total'] <= 2  # Cosine range
    print(f"✓ Test 2 passed: Cosine loss = {loss_dict['total']:.6f}")
    
    # Test 3: With mask
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -3:] = False  # Mask last 3 frames
    loss_dict_masked = criterion_mse(teacher_k, teacher_v, student_k, student_v, mask)
    print(f"✓ Test 3 passed: Masked loss = {loss_dict_masked['total']:.6f}")
    
    # Test 4: Multi-head attention format [B, T, H, D]
    H, D = 8, 24
    teacher_k_mh = torch.randn(B, T, H, D)
    teacher_v_mh = torch.randn(B, T, H, D)
    student_k_mh = torch.randn(B, T, H, D)
    student_v_mh = torch.randn(B, T, H, D)
    
    loss_dict_mh = criterion_mse(teacher_k_mh, teacher_v_mh, student_k_mh, student_v_mh)
    print(f"✓ Test 4 passed: Multi-head loss = {loss_dict_mh['total']:.6f}")
    
    # Test 5: Perfect match (zero loss)
    loss_dict_zero = criterion_mse(teacher_k, teacher_v, teacher_k, teacher_v)
    assert loss_dict_zero['total'] < 1e-6, "Should be zero for perfect match"
    print(f"✓ Test 5 passed: Zero loss for perfect match")
    
    print("All tests passed! ✓")


if __name__ == "__main__":
    test_kv_refinement_loss()
