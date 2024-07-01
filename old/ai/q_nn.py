import torch
from torch import nn


class QNN(nn.Module):
    def __init__(self, input_size=24, hidden_size=128, num_layers=1):
        super().__init__()

        # LSTM layer with 60 input time samples
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(hidden_size+2, 64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x, additional_input):
        # LSTM layer with 60 input time samples
        lstm_out, _ = self.lstm(x)

        # Concatenate LSTM output and additional tensor output
        combined_output = torch.cat((lstm_out[:, -1, :], additional_input),
                                    dim=1)

        # Fully connected layer
        output = self.fc(combined_output)

        return output
