import torch

from common.collector import TrajectoryCollector
from sac.model import ModelBase, TrainerBase
from sac.rnn.network import StateEncoderWrapper


__all__ = ['Trainer', 'Tester']


class Trainer(TrainerBase):
    def __init__(self, *args, **kwargs):
        kwargs.update(state_encoder_wrapper=StateEncoderWrapper,
                      collector=TrajectoryCollector)
        super().__init__(*args, **kwargs)

    def update(self, batch_size, step_size=16,
               normalize_rewards=True, reward_scale=1.0,
               adaptive_entropy=True, target_entropy=-2.0,
               gamma=0.99, soft_tau=0.01, epsilon=1E-6):
        self.train()

        # size: (seq_len, batch_size, item_size)
        observation, action, reward, next_observation, done, hidden = tuple(map(lambda tensor: tensor.to(self.model_device),
                                                                                self.replay_buffer.sample(batch_size, step_size=step_size)))

        # size: (seq_len, batch_size, item_size)
        state, _ = self.state_encoder(observation, hidden)
        with torch.no_grad():
            # size: (1, batch_size, item_size)
            first_observation = observation[0].unsqueeze(dim=0)
            first_hidden = hidden[0].unsqueeze(dim=0)
            _, next_hidden = self.state_encoder(first_observation, first_hidden)

            # size: (seq_len, batch_size, item_size)
            next_state, _ = self.state_encoder(next_observation, next_hidden)

        return self.update_sac(state, action, reward, next_state, done,
                               normalize_rewards, reward_scale,
                               adaptive_entropy, target_entropy,
                               gamma, soft_tau, epsilon)


class Tester(ModelBase):
    def __init__(self, *args, **kwargs):
        kwargs.update(state_encoder_wrapper=StateEncoderWrapper,
                      collector=TrajectoryCollector)
        super().__init__(*args, **kwargs)

        self.eval()
