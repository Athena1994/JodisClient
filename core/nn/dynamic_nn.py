import torch

import torch.nn as nn
import torch.nn.functional as F

from core.data.technical_indicators.collection import IndicatorCollection

class DynamicNN(nn.Module):
    @staticmethod
    def _determine_input_size(data_desc: dict):
        size = 0

        for data in data_desc:
            if data['ohcl']:
                size += 4

            for indicator in data['indicators']:
                ind_desc = IndicatorCollection.get(indicator)
                size += ind_desc.get_value_cnt()

        return size        

    @staticmethod
    def _create_layers(input_size: int, structure: list):
        layers = []

        layer_input = input_size

        for layer_desc in structure:
            if layer_desc['type'] == 'ReLU':
                layers.append(nn.ReLU())
            if layer_desc['type'] == 'Dropout':
                layers.append(nn.Dropout(layer_desc['p']))
            if layer_desc['type'] == 'linear':
                layers.append(nn.Linear(layer_input, 
                                        layer_desc['size']))
                layer_input = layer_desc['size']

        return nn.Sequential(*layers)

    def __init__(self, config: dict):
        super(DynamicNN, self).__init__()

        structure = config['architecture']

        input_size = self._determine_input_size(config['data'])
        hidden_size = structure['LSTM']['hidden_size']

        self.LSTM = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=structure['num_layers'], 
                            batch_first=True)

        self.classifier = self.create_layers(hidden_size, 
                                         structure['output']),
        
    
    def forward(self, x):
        x, _ = self.LSTM(x)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x
