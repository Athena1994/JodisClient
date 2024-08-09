from dataclasses import dataclass

from program.training_manager import TrainingManager, TrainingReporter
from utils.config_utils import assert_fields_in_dict


@dataclass
class JobReport:
    @dataclass
    class Progress:
        ix: int
        cnt: int
        av_time: float
        last_time: float

    episode: Progress
    tr_progress: Progress
    val_progress: Progress

    last_evaluation: dict

    def to_dict(self):
        res = self.__dict__.copy()
        res['episode'] = self.episode.__dict__
        res['tr_progress'] = self.tr_progress.__dict__
        res['val_progress'] = self.val_progress.__dict__
        return res

    @staticmethod
    def from_dict(d: dict):
        assert_fields_in_dict(d, ['episode', 'tr_progress', 'val_progress',
                                  'last_evaluation'])
        return JobReport(
            JobReport.Progress(**d['episode']),
            JobReport.Progress(**d['tr_progress']),
            JobReport.Progress(**d['val_progress']),
            d['last_evaluation']
        )

    def copy(self):
        return JobReport(
            self.episode,
            self.tr_progress,
            self.val_progress,
            self.last_evaluation.copy()
        )

    @staticmethod
    def from_state(state: TrainingReporter.State, info: TrainingManager.Info):
        return JobReport(
            JobReport.Progress(state.epoch.ix,
                               info.max_epoch,
                               state.epoch.av_time,
                               state.epoch.last_time),
            JobReport.Progress(
                state.training.ix / info.experiences_per_step,
                info.training_samples_cnt / info.experiences_per_step,
                state.training.av_time, -1),
            JobReport.Progress(
                state.validation.ix,
                info.validation_sample_cnt,
                state.validation.av_time, -1),
            state.last_evaluation
        )
