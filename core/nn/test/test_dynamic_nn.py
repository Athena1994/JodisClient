

import unittest

from core.nn.dynamic_nn import DynamicNN


class TestDynamicNN(unittest.TestCase):

    def test_instantiation(self):

        dummy_conf = {
            "general":{
                "name": "AgentAlpha",
                "unnormalized_data": "zscore",
                "input_window": 128
            },
            
            "data": [
                {
                    "asset": {
                        "name": "BTCEUR",
                        "source": "default",
                        "interval": "ONE_MINUTE"
                    },
                    "indicators": [
                        {
                            "name": "AwesomeOscillator",
                            "params": {
                                "short_period": 5,
                                "long_period": 34
                            }
                        }
                    ],
                    "volume": True,
                    "ohcl": True
                }
            ],
            
            "architecture": {
                "LSTM": {
                    "num_layers": 1,
                    "hidden_size": 128,
                    "window_size": 128
                },

                "output":[
                    { "type": "ReLU"},
                    { "type": "Dropout", "p": 0.5},
                    { "type": "Linear", "size": 128},
                    { "type": "ReLU"},
                    { "type": "Dropout", "p": 0.5},
                    { "type": "Linear", "size": 3}
                ] 
            }
        }

        nn = DynamicNN(dummy_conf)

        self.assertEqual(nn.LSTM.input_size, 4 + 1 + 1)