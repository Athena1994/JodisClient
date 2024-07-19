

import unittest

import torch

from core.nn.dynamic_nn import DynamicNN


class TestDynamicNN(unittest.TestCase):

    def test_instantiation(self):

        dummy_conf = {
            "general": {
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
                    "hidden_size": 256,
                },

                "classifier": [
                    {"type": "ReLU"},
                    {"type": "Dropout", "p": 0.5},
                    {"type": "Linear", "size": 128},
                    {"type": "ReLU"},
                    {"type": "Dropout", "p": 0.6},
                    {"type": "Linear", "size": 3}
                ]
            }
        }

        dnn = DynamicNN(dummy_conf)

        self.assertEqual(dnn.LSTM.input_size, 4 + 1 + 1)

        self.assertEqual(dnn.LSTM.hidden_size, 256)
        self.assertEqual(dnn.LSTM.num_layers, 1)

        self.assertEqual(len(dnn.classifier), 6)

        self.assertTrue(isinstance(dnn.classifier[0], torch.nn.ReLU))
        self.assertTrue(isinstance(dnn.classifier[1], torch.nn.Dropout))
        self.assertEqual(dnn.classifier[1].p, 0.5)
        self.assertTrue(isinstance(dnn.classifier[2], torch.nn.Linear))
        self.assertEqual(dnn.classifier[2].in_features, 256)
        self.assertEqual(dnn.classifier[2].out_features, 128)
        self.assertTrue(isinstance(dnn.classifier[3], torch.nn.ReLU))
        self.assertTrue(isinstance(dnn.classifier[4], torch.nn.Dropout))
        self.assertEqual(dnn.classifier[4].p, 0.6)
        self.assertTrue(isinstance(dnn.classifier[5], torch.nn.Linear))
        self.assertEqual(dnn.classifier[5].in_features, 128)
        self.assertEqual(dnn.classifier[5].out_features, 3)

        test_data = torch.rand(16, 64, 6)
        output = dnn(test_data)
        self.assertEqual(output.size(), (16, 3))
