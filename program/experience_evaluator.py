

from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List

from core.qlearning.dqn_trainer import DQNTrainer
from utils.config_utils import assert_fields_in_dict


class EvaluationMetric:

    @dataclass
    class Config:
        type: str
        params: object

        @staticmethod
        def from_dict(cfg: dict):
            assert_fields_in_dict(cfg, ['type', 'params'])
            return EvaluationMetric.Config(cfg['type'], cfg['params'])

    class Factory:
        @abstractmethod
        def create_metric(self, desc: 'EvaluationMetric.Config')\
                -> 'EvaluationMetric':
            pass

    def __init__(self):
        pass

    @abstractmethod
    def calculate(self, experiences: List[DQNTrainer.ExperienceTuple]):
        pass


class ExperienceEvaluator:

    @dataclass
    class Config:
        metrics: Dict[str, EvaluationMetric.Config]

        @staticmethod
        def from_dict(cfg: dict):
            assert_fields_in_dict(cfg, ['metrics'])

            return ExperienceEvaluator.Config(
                {name: EvaluationMetric.Config.from_dict(desc)
                 for name, desc in cfg['metrics'].items()}
            )

    @staticmethod
    def from_cfg(cfg: Config, factory: EvaluationMetric.Factory):
        metrics = {}
        for name, desc in cfg.metrics.items():
            metrics[name] = factory.create_metric(desc)
        return ExperienceEvaluator(metrics)

    def __init__(self, metrics: Dict[str, EvaluationMetric]):
        self._metrics = metrics

    def evaluate(self, experiences: List[DQNTrainer.ExperienceTuple]):
        return {
            name: metric.calculate(experiences)
            for name, metric in self._metrics.items()
        }
