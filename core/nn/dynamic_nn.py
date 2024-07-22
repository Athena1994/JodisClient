import torch
import torch.nn as nn

from core.data.technical_indicators.collection import IndicatorCollection


class DynamicNN(nn.Module):
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

    def __init__(self, nn_config: dict, input_config: dict):
        super(DynamicNN, self).__init__()

        def determine_input_size(input_desc: dict):
            size = len(input_desc['params'].get('include', []))

            for indicator in input_desc['params'].get('indicators', []):
                ind_desc = IndicatorCollection.get(indicator['name'])
                size += ind_desc.get_value_cnt()

            return size

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

        output_size = {d['key']: determine_input_size(d)
                       for d in input_config}

        self._units = []

        for unit_conf in nn_config['units']:
            inputs = unit_conf['input']
            if not isinstance(inputs, list):
                inputs = [inputs]

            missing = [i for i in inputs if i not in output_size]
            if len(missing) != 0:
                raise ValueError(f"Missing input data: {missing}")

            name = unit_conf['name']

            unit = {}
            unit['concat'] = len(inputs) > 1
            unit['name'] = name
            input_size = sum([output_size[i] for i in inputs])
            unit['module'], output_size[name] = make_unit(unit_conf['type'],
                                                          input_size,
                                                          unit_conf['params'])
            unit['input'] = inputs
            self._units.append(unit)

        if 'output' not in nn_config:
            raise ValueError("Output key missing in nn configuration.")

        self._output_key = nn_config['output']
        if self._output_key not in output_size:
            raise ValueError(f"Output key {self._output_key} not found")

    def forward(self, x_dict):
        for unit in self._units:
            if unit['concat']:
                inputs = [x_dict[i] for i in unit['input']]
                x = torch.cat(inputs, dim=1)
            else:
                x = x_dict[unit['input'][0]]

            x = unit['module'](x)
            if isinstance(unit['module'], nn.LSTM):
                x = x[0][:, 0, :]

            x_dict[unit['name']] = x

        return x_dict[self._output_key]
