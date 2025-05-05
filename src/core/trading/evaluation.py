

from aithena.core.simulation.data_classes import TransitionResult
from experience_evaluator import EvaluationMetric


class RewardMetric(EvaluationMetric):
    def __init__(self):
        super().__init__()

    def calculate(self, experiences):
        return sum([exp.experience.reward for exp in experiences])


class InvalidActionMetric(EvaluationMetric):
    def __init__(self):
        super().__init__()

    def calculate(self, experiences):
        return sum([1 for exp in experiences
                    if isinstance(exp.meta, TransitionResult)
                    and 'invalid_action' in exp.meta.new_state.context])


class BalanceMetric(EvaluationMetric):
    def __init__(self):
        super().__init__()

    def calculate(self, experiences):
        last_result: TransitionResult = experiences[-1].meta
        if last_result.new_state.context['position_open']:
            return last_result.new_state.context['buy_in_price']
        else:
            return last_result.new_state.context['asset']['EUR']


class MetricFactory(EvaluationMetric.Factory):
    def create_metric(self, desc):
        if desc.type == 'reward':
            return RewardMetric()
        elif desc.type == 'invalid_actions':
            return InvalidActionMetric()
        elif desc.type == 'balance':
            return BalanceMetric()
        else:
            raise Exception(f"Unknown metric type: {desc.type}")
