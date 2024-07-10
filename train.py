import argparse
import json
import sys

import torch

from core.data.data_provider import DataProvider
from core.data.technical_indicators.collection import IndicatorCollection
from core.nn.dynamic_nn import DynamicNN
from core.qlearning.replay_buffer import ReplayBuffer
from core.qlearning.trainer import DQNTrainer
from program.data_manager import Asset, DataManager

def _load_file(path: str):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File {path} not found.")
        sys.exit(1)


def train(trainer: DQNTrainer):
    trainer.perform_exploration()
    

def _instantiate_trainer(cfg: dict, dqn: torch.nn.Module):
    
    bp_cfg = cfg['backpropagation']
    q_cfg = cfg['qlearning']

    replay_buffer = ReplayBuffer(q_cfg['replay_buffer_size'])

    if bp_cfg['optimizer'] != "adam":
        raise ValueError("Only Adam optimizer is supported.")

    optimizer = nn.Adam(dqn.parameters(), 
                        lr=bp_cfg['learning_rate'])

    return DQNTrainer(dqn, 
                         replay_buffer, 
                         optimizer, 
                         q_cfg['target_network_update_freq'], 
                         q_cfg['discount_factor'])


def main():
    parser = argparse.ArgumentParser(description='Train agent.')

    parser.add_argument('data_config', type=str, help='Path to data configuration file.')
    parser.add_argument('agent_config', type=str, help='Path to agent configuration file.')
    parser.add_argument('training_config', type=str, help='Path to training configuration file.')

    
    args = parser.parse_args()

    data_config = _load_file(args.data_config)
    agent_config = _load_file(args.agent_config)
    training_config = _load_file(args.training_config)

    dm = DataManager(data_config)
    data_provider = dm.get_provider(agent_config)

    trainer = _instantiate_trainer(training_config, DynamicNN(agent_config))

    print("tr", data_provider.get_chunk_cnt('tr'))
    print("val", data_provider.get_chunk_cnt('val'))
    print("test", data_provider.get_chunk_cnt('test'))

   # train(data_provider, trainer)



if __name__ == '__main__':
    main()