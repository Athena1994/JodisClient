

import logging
from time import sleep
from core.data.data_provider import ChunkType
from program.base_program import BaseProgram
from program.config_loader import ConfigFiles
from utils.console_util import ScreenBuilder, TableBuilderEx, print_progress_bar


def main():
    logging.basicConfig(level=logging.INFO)

    cfg_files = ConfigFiles(
        agent='config/sin_wave/agent.json',
        data='config/sin_wave/data.json',
        training='config/sin_wave/training.json',
        simulation='config/sin_wave/simulation.json',
        evaluation='config/sin_wave/evaluation.json'
    )
    prog = BaseProgram(cfg_files)

    prog.start_training()

    ep = prog._trainer._experience_provider

    total_tr_cnt = ep.get_experience_cnt(ChunkType.TRAINING)
    total_val_cnt = ep.get_experience_cnt(ChunkType.VALIDATION)

    sb = ScreenBuilder()

    while prog._trainer.is_running():
        tr_state = prog._trainer._reporter.get_state()

        def get_step_cnt(ix, cnt_per_step) -> int:
            return round(ix / cnt_per_step)

        exp_per_step = prog._cfg.training.iterations.experience_cnt
        tr_step_cnt = get_step_cnt(tr_state.current_training_ix, exp_per_step)
        tr_total_step_cnt = get_step_cnt(total_tr_cnt, exp_per_step)

        remaining_tr_time = round(
            tr_state.training_step_time*(tr_total_step_cnt - tr_step_cnt), 2)

        remaining_val_time \
            = round(tr_state.validation_sample_time
                    * (total_val_cnt - tr_state.current_validation_ix), 2)

        # sb.reset_position()
#        sb.goto_line(0)

        tb = TableBuilderEx(sb, 'table')
        tb.add_line(f'epoch: {tr_state.epoch}')

        evaluation = tr_state.last_evaluation or {
            k: 'n/a' for k in prog.evaluator._metrics
        }
        for k, v in evaluation.items():
            tb.add_line(f'{k}: {v}')

        tb.print()

        print_progress_bar("training: ", tr_step_cnt, tr_total_step_cnt,
                           f"(remaining: ~{remaining_tr_time}s)",
                           name="tr",
                           sb=sb)
        print_progress_bar("validation: ",
                           tr_state.current_validation_ix, total_val_cnt,
                           f"(remaining: ~{remaining_val_time}s)",
                           name="val",
                           sb=sb)

        sleep(1)
    print("finished!")


if __name__ == '__main__':
    main()
