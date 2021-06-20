import torch.nn as nn

class StateEncoder(nn.Module):
    def __init__(self, dropout_p):
        super(StateEncoder, self).__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(62, 48),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(48, 48),
            # nn.ReLU(),
            # nn.Dropout(dropout_p),
            # nn.Linear(48, 48),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(32, 32),
            # nn.ReLU(),
            # nn.Dropout(dropout_p),
            # nn.Linear(32, 32),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder_net(x)

class QNet():
    def __init__(self,
                 dropout_p):
        super(QNet, self).__init__()
        self.state_encoder = StateEncoder(dropout_p)
        self.state_layer = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(16, 1)
        )

        self.action_layer = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        b = x.shape[0]
        l = x.shape[1]

        x_reshape = x[:, :, :, :-3].reshape(-1, 62)
        mask = x[:, :, 0, -3:]
        encoded_state = self.state_encoder(x_reshape).reshape(b, l, -1)

        state_value = self.state_layer(encoded_state)
        action_value = self.action_layer(encoded_state)
        q_value = (state_value + (action_value - action_value.mean(dim=1, keepdim=True)))
        q_value = q_value + (1 - mask) * (-1e33)

        return q_value

class LSTMQNet(nn.Module):
    def __init__(self,
                 dropout_p,
                 hist_length,
                 num_layer=1):
        super(LSTMQNet, self).__init__()
        self.state_encoder = StateEncoder(dropout_p)
        self.temporal_encoder = nn.LSTM(
            input_size=16*hist_length,
            hidden_size=512,
            batch_first=True,
            num_layers=num_layer
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

        x_reshape = x[:, :, :, :-3].reshape(-1, 62)
        mask = x[:, :, 0, -3:]
        encoded_state = self.state_encoder(x_reshape).reshape(b, l, -1)

        self.temporal_encoder.flatten_parameters()
        temporal_out, last_hidden_state = self.temporal_encoder(encoded_state, hidden_state)

        state_value = self.state_layer(temporal_out.reshape(-1, temporal_out.shape[-1]))
        action_value = self.action_layer(temporal_out.reshape(-1, temporal_out.shape[-1]))
        q_value = (state_value + (action_value - action_value.mean(dim=1, keepdim=True)))
        q_value = q_value.reshape(b, l, action_value.shape[-1]) + (1 - mask) * (-1e33)
        # q_value = q_value.reshape(b, l, action_value.shape[-1])

        return q_value, last_hidden_state