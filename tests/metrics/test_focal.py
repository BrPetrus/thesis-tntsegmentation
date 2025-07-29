import numpy as np
import torch
import pytest

from tntseg.utilities.metrics.metrics import calculate_batch_stats, Stats, focal_tversky_loss, tversky_index
from tntseg.utilities.metrics.metrics_torch import FocalTverskyLoss, TverskyLoss


class TestTverskyLossClass:
    """Test cases for TverskyLoss PyTorch module (expects float values 0-1)."""
    
    def test_initialization(self):
        """Test proper initialization of TverskyLoss."""
        loss_fn = TverskyLoss(alpha=0.3, beta=0.7)
        assert loss_fn.alpha == 0.3
        assert loss_fn.beta == 0.7
    
    def test_forward_pass_float_inputs(self):
        """Test forward pass with float inputs [0,1]."""
        loss_fn = TverskyLoss(alpha=0.5, beta=0.5)
        
        # PyTorch version expects float values [0,1]
        pred = torch.tensor([[0.8, 0.2, 0.9, 0.1]], dtype=torch.float32)
        target = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
        
        result = loss_fn(pred, target)
        assert isinstance(result, torch.Tensor)
        assert 0.0 <= result <= 1.0
    
    def test_perfect_predictions_high_tversky_index(self):
        """Test that perfect predictions give high Tversky index (close to 1)."""
        loss_fn = TverskyLoss(alpha=0.5, beta=0.5)
        
        pred = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        target = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        
        tversky_index = loss_fn(pred, target)
        assert torch.isclose(tversky_index, torch.tensor(1.0), atol=1e-6)
    
    def test_completely_wrong_predictions_low_tversky_index(self):
        """Test that completely wrong predictions give low Tversky index."""
        loss_fn = TverskyLoss(alpha=0.5, beta=0.5)
        
        pred = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        target = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
        
        tversky_index = loss_fn(pred, target)
        assert tversky_index < 0.1  # Should be very low
    
    def test_alpha_beta_effect(self):
        """Test that alpha/beta parameters affect the result."""
        pred = torch.tensor([[0.8, 0.9, 0.9, 0.1]], dtype=torch.float32)
        target = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        
        # High alpha penalizes false positives more
        loss_high_alpha = TverskyLoss(alpha=0.8, beta=0.2)
        result_high_alpha = loss_high_alpha(pred, target)
        
        # High beta penalizes false negatives more  
        loss_high_beta = TverskyLoss(alpha=0.2, beta=0.8)
        result_high_beta = loss_high_beta(pred, target)
        
        assert not torch.isclose(result_high_alpha, result_high_beta, atol=1e-6)


class TestFocalTverskyLossClass:
    """Test cases for FocalTverskyLoss PyTorch module (expects float values 0-1)."""
    
    def test_initialization(self):
        """Test proper initialization of FocalTverskyLoss."""
        loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.5)
        assert loss_fn.alpha == 0.3
        assert loss_fn.beta == 0.7
        assert loss_fn.gamma == 1.5
    
    def test_forward_pass_float_inputs(self):
        """Test forward pass with float inputs [0,1]."""
        loss_fn = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1.0)
        
        # PyTorch version expects float values [0,1]
        pred = torch.tensor([[0.8, 0.2, 0.9, 0.1]], dtype=torch.float32)
        target = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
        
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert 0.0 <= loss <= 1.0
    
    def test_perfect_predictions_zero_loss(self):
        """Test that perfect predictions give zero focal loss."""
        loss_fn = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1.0)
        
        pred = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        target = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        
        loss = loss_fn(pred, target)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    
    def test_gamma_focusing_effect(self):
        """Test that gamma parameter provides focusing effect."""
        pred = torch.tensor([[0.7, 0.3, 0.6, 0.4]], dtype=torch.float32)
        target = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
        
        loss_gamma_1 = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1.0)
        loss_gamma_2 = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=2.0)
        
        result_gamma_1 = loss_gamma_1(pred, target)
        result_gamma_2 = loss_gamma_2(pred, target)
        
        # Higher gamma should give different (usually higher) loss for hard examples
        assert not torch.isclose(result_gamma_1, result_gamma_2, atol=1e-6)
    
    def test_reduces_to_tversky_loss_when_gamma_1(self):
        """Test that focal reduces to (1 - tversky_index) when gamma=1."""
        pred = torch.tensor([[0.8, 0.2, 0.7, 0.3]], dtype=torch.float32)
        target = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
        
        tversky_loss = TverskyLoss(alpha=0.6, beta=0.4)
        focal_loss = FocalTverskyLoss(alpha=0.6, beta=0.4, gamma=1.0)
        
        tversky_index = tversky_loss(pred, target)
        focal_result = focal_loss(pred, target)
        
        # Focal with gamma=1 should be (1 - tversky_index)
        expected_focal = 1.0 - tversky_index
        assert torch.isclose(focal_result, expected_focal, atol=1e-6)


class TestNumpyVsTorchDifferences:
    """Test the differences between numpy (uint8 0-255) and torch (float 0-1) implementations."""
    
    def test_input_format_difference(self):
        """Test that we understand the input format difference."""
        # This test documents the expected input formats
        
        # NumPy version would expect:
        pred_numpy = np.array([[255, 0, 255, 0]], dtype=np.uint8)  # 0 or 255
        target_numpy = np.array([[255, 0, 255, 0]], dtype=np.uint8)  # 0 or 255
        tp, fp, fn, tn = calculate_batch_stats(pred_numpy, target_numpy)
        result_numpy = tversky_index(tp, fp, fn, alpha=0.5, beta=0.5)
        
        # PyTorch version expects:
        pred_torch = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)  # 0.0 to 1.0
        target_torch = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)  # 0.0 to 1.0
        
        loss_fn = TverskyLoss(alpha=0.5, beta=0.5)
        result = loss_fn(pred_torch, target_torch)
        
        # Perfect match should give Tversky index = 1.0
        assert torch.isclose(result, torch.tensor(1.0))
        assert np.isclose(result_numpy, 1.)
    
    def test_conversion_between_formats(self):
        """Test conversion between uint8 [0,255] and float [0,1] formats."""
        # Simulate uint8 data that would come from numpy implementation
        uint8_data = np.array([0, 255, 0, 255], dtype=np.uint8)
        
        # Convert to torch float format
        torch_data = torch.from_numpy(uint8_data.astype(np.float32) / 255.0)
        
        expected = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
        assert torch.allclose(torch_data, expected)
    
    def test_threshold_difference(self):
        """Test how thresholding differs between implementations."""
        # PyTorch version - continuous values that need thresholding
        pred_continuous = torch.tensor([[0.8, 0.3, 0.9, 0.1]], dtype=torch.float32)
        
        # Apply threshold for binary classification
        pred_binary = (pred_continuous > 0.5).float()
        target = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
        
        loss_fn = TverskyLoss(alpha=0.5, beta=0.5)
        result = loss_fn(pred_binary, target)
        
        # Should get perfect score since thresholded predictions match targets
        assert torch.isclose(result, torch.tensor(1.0))


class TestEdgeCasesTorchImplementation:
    """Test edge cases specific to the PyTorch implementation."""
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the loss."""
        loss_fn = TverskyLoss(alpha=0.5, beta=0.5)
        
        pred = torch.tensor([[0.5, 0.7, 0.3]], dtype=torch.float32, requires_grad=True)
        target = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
        
        result = loss_fn(pred, target)
        
        # For loss function, we want 1 - tversky_index to have gradients
        loss = 1.0 - result
        loss.backward()
        
        assert pred.grad is not None
        assert not torch.allclose(pred.grad, torch.zeros_like(pred.grad))
    
    def test_batch_processing(self):
        """Test that the loss works with batch dimensions."""
        loss_fn = TverskyLoss(alpha=0.5, beta=0.5)
        
        # Batch of 2 samples
        pred = torch.tensor([
            [1.0, 0.0, 1.0, 0.0],
            [0.8, 0.2, 0.9, 0.1]
        ], dtype=torch.float32)
        target = torch.tensor([
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0]
        ], dtype=torch.float32)
        
        result = loss_fn(pred, target)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Should be scalar (averaged over batch)
    
    def test_eta_parameter_prevents_division_by_zero(self):
        """Test that eta parameter prevents division by zero."""
        loss_fn = TverskyLoss(alpha=0.5, beta=0.5)
        
        # All zeros - would cause division by zero without eta
        pred = torch.zeros(1, 4, dtype=torch.float32)
        target = torch.zeros(1, 4, dtype=torch.float32)
        
        result = loss_fn(pred, target, eta=1e-6)
        assert torch.isfinite(result)
        assert not torch.isnan(result)
    
    def test_return_type_is_tensor(self):
        """Test that the loss returns a proper tensor, not float."""
        loss_fn = TverskyLoss(alpha=0.5, beta=0.5)
        
        pred = torch.tensor([[0.8, 0.2]], dtype=torch.float32, requires_grad=True)
        target = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        
        result = loss_fn(pred, target)
        
        # Should return tensor, not Python float (for proper gradient computation)
        assert isinstance(result, torch.Tensor)
        assert result.requires_grad  # Should be part of computation graph


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
