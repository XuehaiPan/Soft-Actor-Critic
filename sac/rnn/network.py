import torch

from common.network_base import LSTMHidden
from sac.network import StateEncoderWrapper as OriginalStateEncoderWrapper


__all__ = ['StateEncoderWrapper']

DEVICE_CPU = torch.device('cpu')


class StateEncoderWrapper(OriginalStateEncoderWrapper):
    def encode(self, observation, hidden=None):
        with torch.no_grad():
            observation = torch.FloatTensor(observation).unsqueeze(dim=0).unsqueeze(dim=0).to(self.device)
            encoded, hidden = self(observation, hidden)
        encoded = encoded.cpu().numpy()[0, 0]
        return encoded, hidden

    def initial_hiddens(self, batch_size=1):
        init_hidden = LSTMHidden(hidden=list(zip(self.encoder.init_hiddens, self.encoder.init_cells)))
        return init_hidden.repeat(1, batch_size, 1)
