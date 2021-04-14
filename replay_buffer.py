import random
from torch import as_tensor as tensor, cat as concatenation, device as devicer, cuda, float32
from helpers import device
import torch

class replay_buffer:
    def __init__(self, size, sample_size, learn):
        self.counter = 0
        self.replay_size = size
        self.buffer = [None for _ in range(self.replay_size)]
        self.sample_size = sample_size
        self.learn = learn

    def stacker(self, sample):
        arays = list(zip(*sample))
        return concatenation(arays[0], 0).squeeze(1), concatenation(arays[1], 0).squeeze(1), concatenation(arays[2], 0).squeeze(1), concatenation(arays[3], 0).squeeze(1), concatenation(arays[4], 0).squeeze(1), concatenation(arays[5], 0).squeeze(1), concatenation(arays[6], 0).squeeze(1)
            
    def sample_data(self):
        if self.learn:
            samples = (random.sample(self.buffer[:min(self.counter, self.replay_size)], min(self.sample_size, self.counter)))
            if samples == []:
                return None, None, None, None, None
            return self.stacker(samples)

    def save_data(self, fields, pieces, obsBoard, obsPieces, actions, rewards, dones, intersects):
        ints = torch.nonzero(intersects)
        for idx in ints:
            data = (torch.clone(fields[idx]).unsqueeze(0), torch.clone(pieces[idx]).unsqueeze(0), torch.clone(obsBoard[idx]).unsqueeze(0), torch.clone(obsPieces[idx]).unsqueeze(0), torch.clone(actions[idx]).unsqueeze(0), torch.clone(rewards[idx]).unsqueeze(0), torch.clone(dones[idx]).unsqueeze(0))
            self.buffer[self.counter % self.replay_size] = data
            self.counter += 1

