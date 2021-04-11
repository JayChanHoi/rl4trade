import torch.nn as nn

class StateEncoder(nn.Module):
    def __init__(self, dropout_p):
        super(StateEncoder, self).__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(16, 32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder_net(x)

class LSTMQNet(nn.Module):
    def __init__(self,
                 dropout_p,
                 hist_length):
        super(LSTMQNet, self).__init__()
        self.state_encoder = StateEncoder()
        self.temporal_encoder = nn.LSTM(
            input_size=32*hist_length,
            hidden_size=512,
            batch_first=True
        )

        self.state_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 1)
        )

        self.action_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 3)
        )

        self.hist_length = hist_length

    def forward(self, x, hidden_state):
        """
        """
        b = x.shape[0]
        l = x.shape[1]

        x_reshape = x.reshape(-1, 8)
        encoded_state = self.state_encoder(x_reshape).reshape(b, l, -1)

        self.temporal_encoder.flatten_parameters()
        temporal_out, last_hidden_state = self.temporal_encoder(encoded_state, hidden_state)

        state_value = self.state_layer(temporal_out.reshape(-1, temporal_out.shape[-1]))
        action_value = self.action_layer(temporal_out.reshape(-1, temporal_out.shape[-1]))
        q_value = (state_value + (action_value - action_value.mean(dim=1, keepdim=True)))
        q_value = q_value.reshape(b, l, action_value.shape[-1])

        return q_value, last_hidden_state