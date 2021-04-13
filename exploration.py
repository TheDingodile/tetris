from random import random
from torch.nn.functional import softmax
from numpy.random import choice
import torch

class Exploration():
    def __init__(self) -> None:
        self.counter = 1

    @property
    def epsilon(self):
        K = 1
        return max(0, 1 - self.counter / K)

    @property
    def K(self):
        C = 100000
        return max(1, C / self.counter)

    def softmax(self, vals):
        self.counter += 1
        if self.counter % 1000 == 1:
            print(f"({str(float(vals.max()))[:4]}, {str(float(vals.std()))[:4]})", end=", ")
        return int(choice(44, 1, p=softmax(vals.view(-1) / self.K, dim=0).detach().cpu().numpy()))

    def greedy(self, vals):
        self.counter += 1
        # if self.counter % 1000 == 1:
        #    print(f"({str(float(vals.max()))[:4]}, {str(float(vals.std()))[:4]})", end=", ")
        return torch.argmax(vals, dim=1)

    def epsilonGreedy(self, vals):
        self.counter += 1
        if self.counter % 10000 == 1:
            print(f"({str(float(vals.max()))[:4]}, {str(float(vals.std()))[:4]})", end=", ")
        return int(choice(44, 1)) if random() < self.epsilon else vals.detach().cpu().numpy().argmax()

    def epsintosoftmax(self, vals):
        self.counter += 1
        if self.counter % 1000000 == 1:
            print(f"({str(float(vals.max()))[:4]}, {str(float(vals.std()))[:4]})", end=", ")
        return int(choice(len(vals), 1)) if random() < self.epsilon else int(choice(len(vals), 1, p=softmax(vals / self.K, dim=0).detach().cpu().numpy()))


    def top_n_moves(self, vals, n=3):
        self.counter += 1
        if self.counter % 10000 == 1:
            print(f"({str(float(vals.max()))[:4]}, {str(float(vals.std()))[:4]})", end=", ")
        if random() < self.epsilon:
            return int(choice(44, 1))
        nump = vals.detach().cpu().numpy()
        for _ in range(n-1):
            if random() < 0.25:
                nump[0,0,nump.argmax()] = 0
        return nump.argmax()
