import torch

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
        return self.encoder.initial_hiddens(batch_size=batch_size)
