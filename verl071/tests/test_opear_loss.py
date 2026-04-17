"""Tests for O-PEaR loss computation."""

from __future__ import annotations

import pytest
import torch

from verl071.opear.loss import compute_opear_loss


class TestOPEaRLossBasic:
    """Test basic loss behaviour with known values."""

    def test_higher_compliant_lower_violating_gives_negative_loss(self):
        """With alpha=0.5, c_lp=-0.5, v_lp=-3.0:
        R = 0.5*(-0.5) - 0.5*(-3.0) = -0.25 + 1.5 = 1.25
        loss = -1.25
        """
        N, L = 1, 1
        compliant_lp = torch.full((N, L), -0.5)
        violating_lp = torch.full((N, L), -3.0)
        mask = torch.ones(N, L)

        loss, metrics = compute_opear_loss(compliant_lp, mask, violating_lp, mask, alpha=0.5)

        assert loss.item() == pytest.approx(-1.25, abs=1e-6)
        assert metrics["opear/loss"] == pytest.approx(-1.25, abs=1e-6)
        assert metrics["opear/compliant_logprob"] == pytest.approx(-0.5, abs=1e-6)
        assert metrics["opear/violating_logprob"] == pytest.approx(-3.0, abs=1e-6)
        assert metrics["opear/R_mean"] == pytest.approx(1.25, abs=1e-6)
        assert metrics["opear/num_pairs"] == 1

    def test_batch_of_pairs(self):
        """Test with multiple pairs (N > 1)."""
        N, L = 3, 1
        # Three pairs with varying quality
        compliant_lp = torch.tensor([[-0.5], [-1.0], [-0.2]])
        violating_lp = torch.tensor([[-3.0], [-2.0], [-4.0]])
        mask = torch.ones(N, L)

        loss, metrics = compute_opear_loss(compliant_lp, mask, violating_lp, mask, alpha=0.5)

        # R per pair: 0.5*(-0.5)-0.5*(-3.0)=1.25, 0.5*(-1.0)-0.5*(-2.0)=0.5, 0.5*(-0.2)-0.5*(-4.0)=1.9
        expected_R_mean = (1.25 + 0.5 + 1.9) / 3.0
        assert loss.item() == pytest.approx(-expected_R_mean, abs=1e-5)
        assert metrics["opear/num_pairs"] == 3


class TestOPEaRLossEqualProbs:
    """Test that equal probabilities yield zero loss."""

    def test_equal_probs_gives_zero_loss(self):
        """When compliant and violating have same log-probs and alpha=0.5,
        R = 0.5*x - 0.5*x = 0, loss = 0."""
        N, L = 2, 4
        lp = torch.full((N, L), -1.5)
        mask = torch.ones(N, L)

        loss, metrics = compute_opear_loss(lp, mask, lp, mask, alpha=0.5)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)
        assert metrics["opear/R_mean"] == pytest.approx(0.0, abs=1e-6)

    def test_equal_probs_asymmetric_alpha(self):
        """When probs are equal but alpha != 0.5, loss is nonzero
        because the weighting is asymmetric.
        R = alpha * x - (1-alpha) * x = (2*alpha - 1) * x
        """
        N, L = 1, 1
        x = -2.0
        lp = torch.full((N, L), x)
        mask = torch.ones(N, L)

        loss, _ = compute_opear_loss(lp, mask, lp, mask, alpha=0.7)

        expected_R = (2 * 0.7 - 1) * x  # 0.4 * (-2.0) = -0.8
        assert loss.item() == pytest.approx(-expected_R, abs=1e-6)


class TestOPEaRLossMaskHandling:
    """Test that masks correctly select valid tokens."""

    def test_mask_excludes_padding(self):
        """Only valid tokens should contribute to normalized log-prob."""
        N, L = 1, 4
        # Compliant: tokens [-1, -1, PAD, PAD] with mask [1, 1, 0, 0]
        compliant_lp = torch.tensor([[-1.0, -1.0, -999.0, -999.0]])
        compliant_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        # Violating: tokens [-3, -3, -3, PAD] with mask [1, 1, 1, 0]
        violating_lp = torch.tensor([[-3.0, -3.0, -3.0, -999.0]])
        violating_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

        loss, metrics = compute_opear_loss(
            compliant_lp, compliant_mask, violating_lp, violating_mask, alpha=0.5
        )

        # normalized compliant lp: (-1 + -1) / 2 = -1.0
        # normalized violating lp: (-3 + -3 + -3) / 3 = -3.0
        assert metrics["opear/compliant_logprob"] == pytest.approx(-1.0, abs=1e-6)
        assert metrics["opear/violating_logprob"] == pytest.approx(-3.0, abs=1e-6)

        expected_R = 0.5 * (-1.0) - 0.5 * (-3.0)  # 1.0
        assert loss.item() == pytest.approx(-expected_R, abs=1e-6)

    def test_all_zero_mask_no_crash(self):
        """An all-zero mask should not cause division by zero."""
        N, L = 1, 3
        lp = torch.tensor([[-1.0, -2.0, -3.0]])
        zero_mask = torch.zeros(N, L)
        ones_mask = torch.ones(N, L)

        # Should not raise
        loss, metrics = compute_opear_loss(lp, zero_mask, lp, ones_mask, alpha=0.5)

        # With zero mask, normalized lp = 0/1 = 0.0 (denominator clamped to 1)
        assert metrics["opear/compliant_logprob"] == pytest.approx(0.0, abs=1e-6)


class TestOPEaRLossGradientFlow:
    """Test that gradients flow through the loss."""

    def test_backward_produces_gradients(self):
        """loss.backward() should produce gradients on the input log-probs."""
        N, L = 2, 5
        compliant_lp = torch.randn(N, L, requires_grad=True)
        violating_lp = torch.randn(N, L, requires_grad=True)
        mask = torch.ones(N, L)

        loss, _ = compute_opear_loss(compliant_lp, mask, violating_lp, mask, alpha=0.5)
        loss.backward()

        assert compliant_lp.grad is not None
        assert violating_lp.grad is not None
        assert compliant_lp.grad.shape == (N, L)
        assert violating_lp.grad.shape == (N, L)
        # Compliant grads should be negative (minimizing loss = maximizing compliant lp)
        # Since loss = -alpha * mean(norm_lp_c) + ..., d(loss)/d(lp_c) = -alpha / (N * L)
        # All gradients on compliant should be negative (same sign)
        assert (compliant_lp.grad < 0).all()

    def test_gradient_magnitude(self):
        """Check gradient values match expected analytic form.
        For a single token with mask=1: d(loss)/d(c_lp) = -alpha / N
        """
        N, L = 1, 1
        compliant_lp = torch.tensor([[-0.5]], requires_grad=True)
        violating_lp = torch.tensor([[-3.0]], requires_grad=True)
        mask = torch.ones(N, L)

        loss, _ = compute_opear_loss(compliant_lp, mask, violating_lp, mask, alpha=0.5)
        loss.backward()

        # d(loss)/d(compliant_lp) = -alpha / N = -0.5
        assert compliant_lp.grad.item() == pytest.approx(-0.5, abs=1e-6)
        # d(loss)/d(violating_lp) = (1 - alpha) / N = 0.5
        assert violating_lp.grad.item() == pytest.approx(0.5, abs=1e-6)


class TestOPEaRLossAlpha:
    """Test alpha parameter behaviour."""

    def test_alpha_zero_ignores_compliant(self):
        """With alpha=0, only the violating term matters.
        R = 0 * c_lp - 1 * v_lp = -v_lp
        loss = -mean(-v_lp) = mean(v_lp)
        """
        N, L = 1, 1
        compliant_lp = torch.tensor([[-0.5]])
        violating_lp = torch.tensor([[-3.0]])
        mask = torch.ones(N, L)

        loss, _ = compute_opear_loss(compliant_lp, mask, violating_lp, mask, alpha=0.0)

        # loss = mean(v_lp) = -3.0
        assert loss.item() == pytest.approx(-3.0, abs=1e-6)

    def test_alpha_one_ignores_violating(self):
        """With alpha=1, only the compliant term matters.
        R = 1 * c_lp - 0 * v_lp = c_lp
        loss = -mean(c_lp)
        """
        N, L = 1, 1
        compliant_lp = torch.tensor([[-0.5]])
        violating_lp = torch.tensor([[-3.0]])
        mask = torch.ones(N, L)

        loss, _ = compute_opear_loss(compliant_lp, mask, violating_lp, mask, alpha=1.0)

        # loss = -mean(c_lp) = 0.5
        assert loss.item() == pytest.approx(0.5, abs=1e-6)
