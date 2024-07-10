import argparse
import json
import sys

import torch

from core.data.data_provider import DataProvider
from core.data.technical_indicators.collection import IndicatorCollection
from core.nn.dynamic_nn import DynamicNN
from core.qlearning.q_arbiter import DQNWrapper, DeepQFunction, QSigArbiter
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

class Simulation:
    def __init__(self) -> None:
        pass

def training_step(data_provider: DataProvider, trainer: DQNTrainer, config: dict)\
    -> float:
    total_error = 0
    total_samples = 0

    trainer.perform_exploration(data_provider)

    e, c = trainer.perform_training(config['batch_size'], 
                                    config['batch_cnt'], 
                                    cuda=True)
    total_error += e
    total_samples += c

    return total_error / total_samples

def validation_step(data_provider: DataProvider, 
                    agent: QSigArbiter, 
                    config: dict,
                    use_test_data: bool):
    pass

def train(agent: QSigArbiter,
          data_provider: DataProvider, 
          trainer: DQNTrainer, 
          cfg: dict):

    epoch_data = []

    for epoch in range(cfg['training']['epochs']):
        error = training_step(data_provider, trainer, cfg['training'])
        validation_reward, episodes = validation_step(data_provider, 
                                            agent, 
                                            cfg, 
                                            use_test_data=False)
    test_reward = validation_step(data_provider, 
                                 agent, 
                                 cfg, 
                                 use_test_data=True)

def _instantiate_trainer(cfg: dict, dqn: torch.nn.Module):
    
    bp_cfg = cfg['backpropagation']
    q_cfg = cfg['qlearning']

    replay_buffer = ReplayBuffer(q_cfg['replay_buffer_size'])

    if bp_cfg['optimizer'] != "adam":
        raise ValueError("Only Adam optimizer is supported.")

    optimizer = torch.optim.Adam(dqn.parameters(), 
                        lr=bp_cfg['learning_rate'],
                        weight_decay=bp_cfg['weight_decay'])

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

    dm = DataManager(data_config, False)
    data_provider = dm.get_provider(agent_config)
    dqn = DynamicNN(agent_config)
    agent = QSigArbiter(DeepQFunction(dqn), 
                        sig=training_config['exploration']['sigma'])
    trainer = _instantiate_trainer(training_config, dqn)

    print("tr", data_provider.get_chunk_cnt('tr'))
    print("val", data_provider.get_chunk_cnt('val'))
    print("test", data_provider.get_chunk_cnt('test'))

    train(agent, data_provider, trainer, training_config)



if __name__ == '__main__':
    main()