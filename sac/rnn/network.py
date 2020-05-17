import torch

from ..network import StateEncoderWrapper as OriginalStateEncoderWrapper


__all__ = ['StateEncoderWrapper']


class StateEncoderWrapper(OriginalStateEncoderWrapper):
    def __init__(self, encoder, device=None):
        super().__init__(encoder=encoder, device=device)
        self.hidden = None

    @torch.no_grad()
    def encode(self, observation):
        observation = torch.FloatTensor(observation).unsqueeze(dim=0).unsqueeze(dim=0).to(self.device)
        encoded, self.hidden, _ = self(observation, self.hidden)
        encoded = encoded.cpu().numpy()[0, 0]
        return encoded

    def initial_hiddens(self, batch_size=1):
        return self.encoder.initial_hiddens(batch_size=batch_size)

    def reset(self):
        self.hidden = None
