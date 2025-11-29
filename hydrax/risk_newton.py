from abc import ABC, abstractmethod

import newton
import warp as wp


class RiskStrategyNewton(ABC):
    """An abstract risk strategy interface.

    A risk strategy defines how we combine costs from different domains with
    domain randomization. For example, a very risk-averse strategy might take
    the worst-case cost over all randomizations, while a risk-seeking strategy
    might take the best-case cost.
    """

    @abstractmethod
    def combine_costs(self, costs: wp.Array) -> wp.Array:
        """Combine costs from different randomizations.

        Args:
            costs: rollout costs, size (randomizations, samples, horizon)

        Returns:
            The combined cost, size (samples, horizon).
        """
        pass


class AverageCostNewton(RiskStrategyNewton):
    """Average cost risk strategy.

    This is the standard expectation over randomizations that is often used in
    reinforcement learning.
    """

    def combine_costs(self, costs: wp.Array) -> wp.Array:
        """Take the average cost over all randomizations."""
        return wp.Array.mean(costs, axis=0)


class WorstCaseNewton(RiskStrategyNewton):
    """A pessimistic worst-case-cost risk strategy."""

    def combine_costs(self, costs: wp.Array) -> wp.Array:
        """Take the highest cost over all randomizations."""
        return wp.Array.max(costs, axis=0)


class BestCaseNewton(RiskStrategyNewton):
    """An optimistic best-case-cost risk strategy."""

    def combine_costs(self, costs: wp.Array) -> wp.Array:
        """Take the lowest cost over all randomizations."""
        return wp.min(costs, axis=0)


class ExponentialWeightedAverageNewton(RiskStrategyNewton):
    """An exponential weighted average risk strategy.

    Costs are combined using a weighted average with weights

        wᵢ = exp(γ cᵢ)/∑ⱼexp(γ cⱼ).

    The parameter γ controls the risk-aversion of the strategy: positive values
    encode a risk-averse strategy, while negative values lead to risk-seeking.
    """

    def __init__(self, gamma: float):
        """Set the risk-aversion parameter γ."""
        self.gamma = gamma

    def combine_costs(self, costs: wp.Array) -> wp.Array:
        """Combine costs using an exponential weighted average."""
        weights = wp.Array.nn.softmax(self.gamma * costs, axis=0)
        return wp.Array.sum(weights * costs, axis=0)


class ValueAtRiskNewton(RiskStrategyNewton):
    """Take the cost value at the (1 - α) quantile."""

    def __init__(self, alpha: float):
        """Set the quantile level α."""
        self.alpha = alpha

    def combine_costs(self, costs: wp.Array) -> wp.Array:
        """Take the cost value at the (1 - α) quantile."""
        return wp.Array.quantile(costs, 1.0 - self.alpha, axis=0)


class ConditionalValueAtRiskNewton(RiskStrategyNewton):
    """Take the expected cost in the tail beyond the (1 - α) quantile."""

    def __init__(self, alpha: float):
        """Set the quantile level α."""
        self.alpha = alpha

    def combine_costs(self, costs: wp.Array) -> wp.Array:
        """Take the expected cost in the tail beyond the (1 - α) quantile."""
        quant = wp.Array.quantile(costs, 1.0 - self.alpha, axis=0)
        return wp.Array.mean(costs, where=costs >= quant, axis=0)
