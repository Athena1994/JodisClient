from program.config_loader import ConfigFiles
from program.program import Program


def main():
    prog = Program(ConfigFiles(
        'config/agent.json',
        'config/data.json',
        'config/training.json',
        'config/simulation.json'
    ))

    prog.train()

    # parser = argparse.ArgumentParser(description='Train agent.')

    # parser.add_argument('data_config', type=str, help='Path to data configuration file.')
    # parser.add_argument('agent_config', type=str, help='Path to agent configuration file.')
    # parser.add_argument('training_config', type=str, help='Path to training configuration file.')

    # args = parser.parse_args()

    # data_config = _load_file(args.data_config)
    # agent_config = _load_file(args.agent_config)
    # training_config = _load_file(args.training_config)

    # dm = DataManager(data_config, False)

    # def converter(state: State) -> dict:
    #     return {
    #         'time_series': state.sample,
    #         'context': torch.Tensor()
    #     }

    # data_provider = dm.get_provider(agent_config)
    # dqn = DQNWrapper(DynamicNN(agent_config), )
    # agent = QSigArbiter(DeepQFunction(dqn),
    #                     sig=training_config['exploration']['sigma'])
    # trainer = _instantiate_trainer(training_config, dqn)

    # print("tr", data_provider.get_chunk_cnt('tr'))
    # print("val", data_provider.get_chunk_cnt('val'))
    # print("test", data_provider.get_chunk_cnt('test'))

    # train(agent, data_provider, trainer, training_config)


if __name__ == '__main__':
    main()
