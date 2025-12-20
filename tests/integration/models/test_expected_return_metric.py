"""
Integration tests for ExpectedReturnCalculator metric.

Tests the expected return calculation against pre-computed values for various scenarios.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from models.metrics.calculators import ExpectedReturnCalculator


class TestExpectedReturnCalculator:
    """Integration tests for expected return metric calculation."""

    def test_perfect_predictions_no_commission(self):
        """
        Scenario 1: Perfect predictions with no commission.
        All predictions match true labels exactly.
        Expected: Average return = 1.0 per sample (perfect directional accuracy)
        """
        # Create perfect predictions (high confidence in correct class)
        predictions = torch.tensor(
            [
                [5.0, 0.0, 0.0],  # Predicts class 0, truth is 0
                [0.0, 5.0, 0.0],  # Predicts class 1, truth is 1
                [0.0, 0.0, 5.0],  # Predicts class 2, truth is 2
                [5.0, 0.0, 0.0],  # Predicts class 0, truth is 0
                [0.0, 0.0, 5.0],  # Predicts class 2, truth is 2
            ],
            dtype=torch.float32,
        )

        targets = torch.tensor([0, 1, 2, 0, 2], dtype=torch.long)

        # With lambda=1.0, theta=0.0
        calc = ExpectedReturnCalculator(lambda_values=1.0, theta_values=0.0, aggregate="mean")
        results = calc.calculate(predictions, targets)

        # Expected: Only classes 0 and 2 contribute (not neutral class 1)
        # Sample 0: True=0 (-1), Pred≈0 (-1) -> +1 * lambda ≈ +1
        # Sample 1: True=1 (0), Pred=1 (0) -> 0 (neutral doesn't contribute)
        # Sample 2: True=2 (+1), Pred≈2 (+1) -> +1 * lambda ≈ +1
        # Sample 3: True=0 (-1), Pred≈0 (-1) -> +1 * lambda ≈ +1
        # Sample 4: True=2 (+1), Pred≈2 (+1) -> +1 * lambda ≈ +1
        # Average: ≈ (0.99 + 0 + 0.99 + 0.99 + 0.99) / 5 ≈ 0.79

        # Due to softmax, probabilities aren't exactly 1.0, but very close
        # The matrix formulation gives slightly lower values due to probability mass
        expected_return = 0.78  # Adjusted based on actual softmax probabilities
        assert (
            abs(results[0].value - expected_return) < 0.02
        ), f"Expected ≈{expected_return}, got {results[0].value}"

    def test_opposite_predictions_no_commission(self):
        """
        Scenario 2: All predictions are opposite to true labels.
        Expected: Average return = -1.0 per actionable sample
        """
        # Create opposite predictions
        predictions = torch.tensor(
            [
                [0.0, 0.0, 5.0],  # Predicts class 2 (+1), truth is 0 (-1)
                [5.0, 0.0, 0.0],  # Predicts class 0 (-1), truth is 2 (+1)
                [0.0, 0.0, 5.0],  # Predicts class 2 (+1), truth is 0 (-1)
                [5.0, 0.0, 0.0],  # Predicts class 0 (-1), truth is 2 (+1)
            ],
            dtype=torch.float32,
        )

        targets = torch.tensor([0, 2, 0, 2], dtype=torch.long)

        calc = ExpectedReturnCalculator(lambda_values=1.0, theta_values=0.0, aggregate="mean")
        results = calc.calculate(predictions, targets)

        # All predictions are opposite, so each should contribute ≈-1
        # Due to softmax not being exactly 1.0, we get slightly higher (less negative)
        expected_return = -0.98
        assert (
            abs(results[0].value - expected_return) < 0.02
        ), f"Expected ≈{expected_return}, got {results[0].value}"

    def test_neutral_predictions_and_targets(self):
        """
        Scenario 3: All predictions and/or targets are neutral (class 1).
        Expected: Return = 0 (neutral doesn't contribute to directional return)
        """
        # All neutral predictions
        predictions = torch.tensor(
            [
                [0.0, 5.0, 0.0],  # Predicts neutral
                [0.0, 5.0, 0.0],  # Predicts neutral
                [0.0, 5.0, 0.0],  # Predicts neutral
            ],
            dtype=torch.float32,
        )

        targets = torch.tensor([0, 1, 2], dtype=torch.long)

        calc = ExpectedReturnCalculator(lambda_values=1.0, theta_values=0.0, aggregate="mean")
        results = calc.calculate(predictions, targets)

        # Neutral predictions contribute 0 directional return
        expected_return = 0.0
        assert (
            abs(results[0].value - expected_return) < 0.01
        ), f"Expected ≈{expected_return}, got {results[0].value}"

    def test_with_commission_reduces_return(self):
        """
        Scenario 4: Perfect predictions but with commission.
        Expected: Return is reduced by commission * sum of non-neutral probabilities
        """
        # Perfect predictions with high confidence
        predictions = torch.tensor(
            [
                [5.0, 0.0, 0.0],  # Predicts class 0, P(non-neutral) ≈ 1.0
                [0.0, 0.0, 5.0],  # Predicts class 2, P(non-neutral) ≈ 1.0
                [5.0, 0.0, 0.0],  # Predicts class 0, P(non-neutral) ≈ 1.0
                [0.0, 0.0, 5.0],  # Predicts class 2, P(non-neutral) ≈ 1.0
            ],
            dtype=torch.float32,
        )

        targets = torch.tensor([0, 2, 0, 2], dtype=torch.long)

        # Without commission
        calc_no_comm = ExpectedReturnCalculator(
            lambda_values=1.0, theta_values=0.0, aggregate="mean"
        )
        results_no_comm = calc_no_comm.calculate(predictions, targets)

        # With commission
        theta = 0.1
        calc_with_comm = ExpectedReturnCalculator(
            lambda_values=1.0, theta_values=theta, aggregate="mean"
        )
        results_with_comm = calc_with_comm.calculate(predictions, targets)

        # Expected difference should be approximately -theta (since P(non-neutral) ≈ 1 for each sample)
        diff = results_no_comm[0].value - results_with_comm[0].value
        assert abs(diff - theta) < 0.01, f"Expected commission impact ≈{theta}, got {diff}"

    def test_lambda_linearity(self):
        """
        Scenario 5: Test that expected return scales linearly with lambda.
        Expected: ER(2λ) = 2 * ER(λ)
        """
        predictions = torch.tensor(
            [
                [3.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 3.0],
            ],
            dtype=torch.float32,
        )

        targets = torch.tensor([0, 2, 0, 2], dtype=torch.long)

        # Calculate with lambda=1.0
        calc_l1 = ExpectedReturnCalculator(lambda_values=1.0, theta_values=0.0, aggregate="mean")
        results_l1 = calc_l1.calculate(predictions, targets)

        # Calculate with lambda=2.0
        calc_l2 = ExpectedReturnCalculator(lambda_values=2.0, theta_values=0.0, aggregate="mean")
        results_l2 = calc_l2.calculate(predictions, targets)

        # Should be exactly 2x due to linearity
        assert (
            abs(results_l2[0].value - 2 * results_l1[0].value) < 1e-6
        ), f"Linearity violated: ER(2λ)={results_l2[0].value}, 2*ER(λ)={2*results_l1[0].value}"

    def test_theta_linearity(self):
        """
        Scenario 6: Test that commission impact scales linearly with theta.
        Expected: ER(θ₂) - ER(θ₁) = (θ₁ - θ₂) * avg(P(non-neutral))
        """
        predictions = torch.tensor(
            [
                [2.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],
                [2.0, 0.5, 0.0],
            ],
            dtype=torch.float32,
        )

        targets = torch.tensor([0, 2, 0], dtype=torch.long)

        # Calculate with different thetas
        calc_t0 = ExpectedReturnCalculator(lambda_values=1.0, theta_values=0.0, aggregate="mean")
        results_t0 = calc_t0.calculate(predictions, targets)

        calc_t1 = ExpectedReturnCalculator(lambda_values=1.0, theta_values=0.1, aggregate="mean")
        results_t1 = calc_t1.calculate(predictions, targets)

        calc_t2 = ExpectedReturnCalculator(lambda_values=1.0, theta_values=0.2, aggregate="mean")
        results_t2 = calc_t2.calculate(predictions, targets)

        # Differences should be proportional to theta differences
        diff1 = results_t0[0].value - results_t1[0].value  # Impact of 0.1 theta
        diff2 = results_t0[0].value - results_t2[0].value  # Impact of 0.2 theta

        assert (
            abs(diff2 - 2 * diff1) < 1e-6
        ), f"Theta linearity violated: diff(0.2)={diff2}, 2*diff(0.1)={2*diff1}"

    def test_sum_vs_mean_aggregation(self):
        """
        Scenario 7: Test that sum aggregation is exactly N times mean aggregation.
        """
        predictions = torch.tensor(
            [
                [1.0, 0.0, 2.0],
                [2.0, 0.0, 1.0],
                [0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
            ],
            dtype=torch.float32,
        )

        targets = torch.tensor([2, 0, 2, 0, 1], dtype=torch.long)

        N = len(targets)

        # Calculate with mean
        calc_mean = ExpectedReturnCalculator(
            lambda_values=1.0, theta_values=0.05, aggregate="mean"
        )
        results_mean = calc_mean.calculate(predictions, targets)

        # Calculate with sum
        calc_sum = ExpectedReturnCalculator(lambda_values=1.0, theta_values=0.05, aggregate="sum")
        results_sum = calc_sum.calculate(predictions, targets)

        assert (
            abs(results_sum[0].value - N * results_mean[0].value) < 1e-6
        ), f"Sum should be N*mean: sum={results_sum[0].value}, N*mean={N*results_mean[0].value}"

    def test_uniform_probabilities(self):
        """
        Scenario 8: Uniform probability distributions.
        Expected: With uniform probs, directional return depends only on true labels.
        """
        # All uniform predictions
        predictions = torch.tensor(
            [
                [1.0, 1.0, 1.0],  # Uniform
                [1.0, 1.0, 1.0],  # Uniform
                [1.0, 1.0, 1.0],  # Uniform
                [1.0, 1.0, 1.0],  # Uniform
            ],
            dtype=torch.float32,
        )

        targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)

        calc = ExpectedReturnCalculator(lambda_values=1.0, theta_values=0.0, aggregate="mean")
        results = calc.calculate(predictions, targets)

        # With uniform probs [1/3, 1/3, 1/3]:
        # For true=-1 (class 0): A1[0,:] @ [1/3,1/3,1/3] = [1,0,-1] @ [1/3,1/3,1/3] = 1/3 - 1/3 = 0
        # For true=0 (class 1): A1[1,:] @ [1/3,1/3,1/3] = [0,0,0] @ [1/3,1/3,1/3] = 0
        # For true=+1 (class 2): A1[2,:] @ [1/3,1/3,1/3] = [-1,0,1] @ [1/3,1/3,1/3] = -1/3 + 1/3 = 0
        # All should give 0 expected return
        expected_return = 0.0
        assert (
            abs(results[0].value - expected_return) < 1e-6
        ), f"Expected {expected_return}, got {results[0].value}"

    def test_manual_calculation_verification(self):
        """
        Scenario 9: Manually computed example with exact probabilities.
        Verify against hand-calculated expected return.
        """
        # Create specific logits that give known probabilities after softmax
        # We'll use simple values and compute expected return manually
        predictions = torch.tensor(
            [
                [2.0, 0.0, 0.0],  # After softmax: [0.8808, 0.0596, 0.0596]
                [0.0, 0.0, 2.0],  # After softmax: [0.0596, 0.0596, 0.8808]
            ],
            dtype=torch.float32,
        )

        targets = torch.tensor([0, 2], dtype=torch.long)

        # Compute softmax probabilities manually
        probs = torch.softmax(predictions, dim=1).numpy()

        # Manual calculation using matrix formulation:
        # Sample 0: True=0 (-1), probs after softmax
        #   y_onehot = [1, 0, 0]
        #   A1 = [[1,0,-1],[0,0,0],[-1,0,1]]
        #   directional_return = y_onehot^T @ A1 @ probs = [1,0,0] @ A1 @ probs
        #                       = [1,0,-1] @ probs
        #   commission = sum of probs[0] and probs[2]

        # Sample 1: True=2 (+1), probs after softmax
        #   y_onehot = [0, 0, 1]
        #   directional_return = [0,0,1] @ A1 @ probs = [-1,0,1] @ probs
        #   commission = sum of probs[0] and probs[2]

        lambda_val = 1.0
        theta_val = 0.1

        # Compute manually using actual probabilities
        A1 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])

        # Sample 0
        y0 = np.array([1, 0, 0])
        dir_ret_0 = y0 @ A1 @ probs[0]
        comm_0 = probs[0][0] + probs[0][2]

        # Sample 1
        y1 = np.array([0, 0, 1])
        dir_ret_1 = y1 @ A1 @ probs[1]
        comm_1 = probs[1][0] + probs[1][2]

        # Expected return per sample
        er_0 = lambda_val * dir_ret_0 - theta_val * comm_0
        er_1 = lambda_val * dir_ret_1 - theta_val * comm_1
        expected_mean = (er_0 + er_1) / 2

        calc = ExpectedReturnCalculator(
            lambda_values=lambda_val, theta_values=theta_val, aggregate="mean"
        )
        results = calc.calculate(predictions, targets)

        assert (
            abs(results[0].value - expected_mean) < 0.001
        ), f"Manual calculation mismatch: expected={expected_mean:.6f}, got={results[0].value:.6f}"

    def test_multiple_lambda_theta_combinations(self):
        """
        Scenario 10: Test multiple lambda and theta combinations simultaneously.
        """
        predictions = torch.randn(20, 3)
        targets = torch.randint(0, 3, (20,))

        lambdas = [0.5, 1.0, 2.0]
        thetas = [0.0, 0.01, 0.05, 0.1]

        calc = ExpectedReturnCalculator(
            lambda_values=lambdas, theta_values=thetas, aggregate="mean"
        )
        results = calc.calculate(predictions, targets)

        # Should return len(lambdas) * len(thetas) results
        expected_num_results = len(lambdas) * len(thetas)
        assert (
            len(results) == expected_num_results
        ), f"Expected {expected_num_results} results, got {len(results)}"

        # Verify all results have correct metadata
        for result in results:
            assert "lambda" in result.metadata
            assert "theta" in result.metadata
            assert "aggregate" in result.metadata
            assert result.metadata["lambda"] in lambdas
            assert result.metadata["theta"] in thetas

    def test_edge_case_single_sample(self):
        """
        Scenario 11: Test with a single sample.
        """
        predictions = torch.tensor([[1.0, 0.0, 2.0]], dtype=torch.float32)
        targets = torch.tensor([2], dtype=torch.long)

        calc = ExpectedReturnCalculator(lambda_values=1.0, theta_values=0.0, aggregate="mean")
        results = calc.calculate(predictions, targets)

        # Should not crash and should return a valid result
        assert len(results) == 1
        assert isinstance(results[0].value, (float, np.floating))
        assert not np.isnan(results[0].value)

    def test_edge_case_all_neutral_targets(self):
        """
        Scenario 12: All targets are neutral (class 1).
        Expected: No directional return (only commission impact if any)
        """
        predictions = torch.tensor(
            [
                [2.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        )

        targets = torch.tensor([1, 1, 1], dtype=torch.long)

        calc = ExpectedReturnCalculator(lambda_values=1.0, theta_values=0.0, aggregate="mean")
        results = calc.calculate(predictions, targets)

        # With neutral targets and theta=0, directional component should be 0
        # (A1[1,:] = [0,0,0])
        expected_return = 0.0
        assert (
            abs(results[0].value - expected_return) < 1e-6
        ), f"Expected {expected_return}, got {results[0].value}"

    def test_metadata_correctness(self):
        """
        Scenario 13: Verify that metadata is correctly attached to results.
        """
        lambda_val = 1.5
        theta_val = 0.03
        aggregate_method = "sum"

        predictions = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))

        calc = ExpectedReturnCalculator(
            lambda_values=lambda_val, theta_values=theta_val, aggregate=aggregate_method
        )
        results = calc.calculate(predictions, targets)

        assert len(results) == 1
        result = results[0]

        # Check metadata
        assert result.metadata["lambda"] == lambda_val
        assert result.metadata["theta"] == theta_val
        assert result.metadata["aggregate"] == aggregate_method

        # Check metric name format
        expected_name = f"expected_return_lambda_{lambda_val:.4f}_theta_{theta_val:.4f}"
        assert result.name == expected_name


@pytest.mark.integration
class TestExpectedReturnIntegrationWithTraining:
    """Integration tests for ExpectedReturnCalculator in training context."""

    def test_calculator_with_model_predictions(self):
        """
        Test that calculator works with actual model outputs (logits).
        """
        # Simulate model logits (before softmax)
        batch_size = 32
        num_classes = 3

        logits = torch.randn(batch_size, num_classes) * 2  # Scale for variety
        targets = torch.randint(0, num_classes, (batch_size,))

        calc = ExpectedReturnCalculator(
            lambda_values=[0.5, 1.0, 1.5], theta_values=[0.0, 0.001, 0.01], aggregate="mean"
        )

        results = calc.calculate(logits, targets)

        # Should return valid results for all combinations
        assert len(results) == 9  # 3 lambdas × 3 thetas
        for result in results:
            assert not np.isnan(result.value)
            assert not np.isinf(result.value)

    def test_requires_probabilities_flag(self):
        """
        Test that the calculator correctly reports it requires probabilities.
        """
        calc = ExpectedReturnCalculator()
        assert calc.requires_probabilities is True
