import torch.nn as nn
import torch

class StateEncoder(nn.Module):
    def __init__(self, dropout_p):
        super(StateEncoder, self).__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(182, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder_net(x)

class QNet(nn.Module):
    def __init__(self, dropout_p):
        super(QNet, self).__init__()
        self.state_encoder = StateEncoder(dropout_p)
        self.state_layer = nn.Sequential(
            nn.Linear(4*32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.action_layer = nn.Sequential(
            nn.Linear(4*32, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        b = x.shape[0]

        x_reshape = x[:, :, :-3].reshape(-1, 62)
        mask = x[:, 0, -3:]
        encoded_state = self.state_encoder(x_reshape).reshape(b, -1)

        state_value = self.state_layer(encoded_state)
        action_value = self.action_layer(encoded_state)
        q_value = (state_value + (action_value - action_value.mean(dim=1, keepdim=True)))
        q_value = q_value + mask.eq(0) * (-1e33)

        return q_value

class LSTMQNet(nn.Module):
    def __init__(self,
                 dropout_p,
                 hist_length,
                 num_layer=1):
        super(LSTMQNet, self).__init__()
        self.state_encoder = StateEncoder(dropout_p)
        self.temporal_encoder = nn.LSTMCell(
            input_size=64*hist_length,
            hidden_size=512
        )

        self.state_layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.action_layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.hist_length = hist_length

    def forward(self, x, hidden_states_tuple):
        """
        """
        b = x.shape[0]
        l = x.shape[1]

        x_reshape = x[:, :, :, :-3].reshape(-1, 182)
        mask = x[:, :, 0, -3:]
        encoded_state = self.state_encoder(x_reshape).reshape(b, l, -1)

        hns, cns = hidden_states_tuple
        hn = None
        cn = None
        temporal_output_list = []
        for iter in range(l):
            hn, cn = self.temporal_encoder(encoded_state[:, iter, :], (hns[iter, 0, :, :], cns[iter, 0, :, :]))
            temporal_output_list.append(hn)
        temporal_output = torch.stack(temporal_output_list, dim=1)

        state_value = self.state_layer(temporal_output.reshape(-1, temporal_output.shape[-1]))
        action_value = self.action_layer(temporal_output.reshape(-1, temporal_output.shape[-1]))
        q_value = (state_value + (action_value - action_value.mean(dim=1, keepdim=True)))
        q_value = q_value.reshape(b, l, action_value.shape[-1]) + (1 - mask) * (-1e33)
        # q_value = q_value.reshape(b, l, action_value.shape[-1])

        return q_value, (hn.unsqueeze(0), cn.unsqueeze(0))