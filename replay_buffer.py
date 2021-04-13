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

    def save_data(self, data, intersects):
        intersects = torch.nonzero(intersects).flatten().long()
        for j in intersects:
            good_data = []
            for dat in data:
                good_data.append(dat[j])
            self.buffer[self.counter % self.replay_size] = good_data
            self.counter += 1

    def stacker(self, sample):
        if self.learn and self.buffer[0] != None:
            arays = list(zip(*sample))
            return tensor(arays[0]).unsqueeze(1).float().to(device), tensor(arays[1]).float().to(device), tensor(arays[2]).unsqueeze(1).float().to(device), tensor(arays[3]).float().to(device), tensor(arays[4]).float().to(device), tensor(arays[5]).float().to(device), tensor(arays[6]).to(device)
        else:
            return None, None, None, None, None, None, None
            
    def sample_data(self):
        if self.learn and self.counter > self.replay_size:
            samples = (random.sample(self.buffer[:min(self.counter, self.replay_size)], min(self.sample_size, self.counter)))
            return self.stacker(samples)
        else:
            return None, None, None, None, None, None, None


