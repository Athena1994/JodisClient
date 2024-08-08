from dataclasses import dataclass
import torch
import torch.nn as nn
from utils.config_utils import assert_fields_in_dict


class DynamicNN(nn.Module):

    @dataclass
    class Config:
        @dataclass
        class Unit:
            name: str
            input: list[str]
            type: str
            params: object

            @staticmethod
            def from_dict(conf: dict) -> 'DynamicNN.Config.Unit':
                assert_fields_in_dict(conf, ["name", "input", "type", "params"])

                inp = conf['input']
                if not isinstance(inp, list):
                    inp = [inp]
                return DynamicNN.Config.Unit(conf['name'],
                                             inp,
                                             conf['type'],
                                             conf['params'])

        @dataclass
        class Input:
            @dataclass
            class Data:

                key: str
                type: str
                config: object

                @staticmethod
                def from_dict(conf: dict) -> 'DynamicNN.Config.Input':
                    from core.data.assets.asset_manager import AssetManager
                    from core.simulation.state_manager import StateProvider
                    assert_fields_in_dict(conf, ["key", "type", "params"])

                    if conf['type'].upper() == 'asset'.upper():
                        config = (AssetManager.Config.Provider
                                  .from_dict(conf['params']))
                    elif conf['type'].upper() == 'state'.upper():
                        config = StateProvider.Config.from_dict(conf['params'])
                    else:
                        raise ValueError(f"Unknown input type: {conf['type']}")

                    return DynamicNN.Config.Input.Data(conf['key'],
                                                       conf['type'],
                                                       config)

            input_window: int
            data: list["Data"]

            @staticmethod
            def from_dict(conf: dict) -> 'DynamicNN.Config.Input':
                assert_fields_in_dict(conf, ["input_window", "data"])

                return DynamicNN.Config.Input(
                    conf['input_window'],
                    [DynamicNN.Config.Input.Data.from_dict(d)
                     for d in conf['data']]
                )

        units: list["Unit"]
        output: str

        @staticmethod
        def from_dict(conf: dict) -> 'DynamicNN.Config':
            assert_fields_in_dict(conf, ["units", "output"])

            return DynamicNN.Config(
                [DynamicNN.Config.Unit.from_dict(d)
                 for d in conf['units']],
                conf['output']
            )

    @staticmethod
    def _create_sequential(input_size: int, structure: list):
        layers = []

        layer_input = input_size

        for layer_desc in structure:
            if layer_desc['type'].upper() == 'ReLU'.upper():
                layers.append(nn.ReLU())
            elif layer_desc['type'].upper() == 'Dropout'.upper():
                layers.append(nn.Dropout(layer_desc['p']))
            elif layer_desc['type'].upper() == 'linear'.upper():
                layers.append(nn.Linear(layer_input,
                                        layer_desc['size']))
                layer_input = layer_desc['size']
            else:
                raise ValueError(f"Unknown layer type: {layer_desc['type']}")
        return nn.Sequential(*layers), layer_input

    def __init__(self, nn_cfg: Config, input_cfg: Config.Input):
        from core.data.assets.asset_manager import AssetManager
        from core.simulation.state_manager import StateProvider
        from core.data.technical_indicators.collection \
            import IndicatorCollection

        super(DynamicNN, self).__init__()

        def determine_input_size(cfg: DynamicNN.Config.Input):
            if isinstance(cfg.config, AssetManager.Config.Provider):
                size = len(cfg.config.include)
                for indicator in cfg.config.indicators:
                    ind_desc = IndicatorCollection.get(indicator['name'])
                    size += ind_desc.get_value_cnt()
                return size
            elif isinstance(cfg.config, StateProvider.Config):
                return len(cfg.config.include)
            else:
                raise ValueError(f"Unknown input type: {type(cfg.config)}")

        def make_unit(type: str, input_size: int, params: dict):
            if type.upper() == 'LSTM'.upper():
                return nn.LSTM(input_size=input_size,
                               hidden_size=params['hidden_size'],
                               num_layers=params['num_layers'],
                               batch_first=True), params['hidden_size']
            elif type.upper() == 'Sequence'.upper():
                return self._create_sequential(input_size, params)
            else:
                raise ValueError(f"Unknown unit type: {type}")

        output_size_dict = {d.key: determine_input_size(d)
                            for d in input_cfg.data}

        self._units = []

        for unit_conf in nn_cfg.units:
            missing = [i for i in unit_conf.input if i not in output_size_dict]
            if len(missing) != 0:
                raise ValueError(f"Missing input data: {missing}")

            unit = {}
            unit['concat'] = len(unit_conf.input) > 1
            unit['name'] = unit_conf.name
            input_size = sum([output_size_dict[i] for i in unit_conf.input])
            unit['module'], output_size_dict[unit_conf.name]\
                = make_unit(unit_conf.type,
                            input_size,
                            unit_conf.params)
            unit['input'] = unit_conf.input
            self._units.append(unit)
            self.add_module(unit['name'], unit['module'])
        self._output_key = nn_cfg.output
        if self._output_key not in output_size_dict:
            raise ValueError(f"Output key {self._output_key} not found")

        # init modules
        for module in self.modules():
            if isinstance(module, nn.LSTM):
                nn.init.xavier_normal_(module.weight_ih_l0)
                nn.init.orthogonal_(module.weight_hh_l0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight.data)
                nn.init.uniform_(module.bias.data)

    def forward(self, x_dict):
        for unit in self._units:
            if unit['concat']:
                inputs = [x_dict[i] for i in unit['input']]
                x = torch.cat(inputs, dim=1)
            else:
                x = x_dict[unit['input'][0]]

            is_lstm = isinstance(unit['module'], nn.LSTM)

            if is_lstm:
                unit['module'].flatten_parameters()

            x = unit['module'](x)
            if is_lstm:
                unit['module'].flatten_parameters()
                x = x[0][:, 0, :]

            x_dict[unit['name']] = x

        return x_dict[self._output_key]
