import torch

from sac.network import StateEncoderWrapper as OriginalStateEncoderWrapper


__all__ = ['StateEncoderWrapper']


class StateEncoderWrapper(OriginalStateEncoderWrapper):
    def encode(self, observation, hidden=None):
        with torch.no_grad():
            observation = torch.FloatTensor(observation).unsqueeze(dim=0).unsqueeze(dim=0).to(self.device)
            encoded, hidden_last, hidden_all = self(observation, hidden)
        encoded = encoded.cpu().numpy()[0, 0]
        return encoded, hidden_last, hidden_all

    def initial_hiddens(self, batch_size=1):
        return self.encoder.initial_hiddens(batch_size=batch_size)
