import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from exploration import Exploration
import pickle
from helpers import device


class Net:
    def __init__(self, batch, height, width):
        self.height = height
        self.width = width
        self.batch = batch
        self.network = Network().to(device)
        self.target_network = Network().to(device)
        self.placeholder_network = Network().to(device)
        self.gamma = 0.99
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.network.parameters(), lr=1e-4, weight_decay=1e-5)
        self.explorer = Exploration()
        self.fields = torch.zeros(self.batch, 1, self.height, self.width, device=device).long()
        self.pieces = torch.zeros(self.batch, 14, device=device).long()
        self.actions = torch.zeros(self.batch, device=device).long()

    def take_action(self, env):
        intersects = torch.nonzero(env.intersected).flatten().long()
        field = env.field
        pieces = env.AInext_pieces
        if len(intersects) > 0:
            vals = self.network(field[intersects], pieces[intersects])
            action = self.explorer.greedy(vals)
            for i in range(len(intersects)):
                inter = intersects[i]
                self.fields[inter] = field[i]
                self.pieces[inter] = pieces[i]
                self.actions[inter] = action[i]
            return action
        else:
            return None

    def DoubleQlearn(self, pre_AIfield, pre_AIpieces, AIfield, AIpieces, pre_action, last_reward, dones, learn):
        if learn:
            vals = self.network(pre_AIfield, pre_AIpieces)
            vals_next = self.network(AIfield, AIpieces)
            vals_target_next = self.target_network(AIfield, AIpieces)
            value_next = torch.gather(vals_target_next, 1, torch.argmax(vals_next, 1).unsqueeze(1))
            td_target = (value_next.view(-1) * self.gamma * (1 - dones) + last_reward)
            td_guess = torch.gather(vals, 1, pre_action.unsqueeze(1)).squeeze(1)
            loss_value_network = self.criterion(td_guess, td_target)
            loss_value_network.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def update_target_network(self):
        self.target_network = pickle.loads(pickle.dumps(self.placeholder_network))
        self.placeholder_network = pickle.loads(pickle.dumps(self.network))

class Network(nn.Module):
    def __init__(self, vis_pieces=1):
        super(Network, self).__init__()
        self.size_after_con = 3520
        self.conv = nn.Sequential(nn.Conv2d(1, 128, 5), nn.LeakyReLU(), nn.Conv2d(128, 32, 1), nn.LeakyReLU(), nn.Flatten())
        self.linear = nn.Sequential(nn.Linear(self.size_after_con + 7 * (1 + vis_pieces), 40), nn.LeakyReLU(), nn.Linear(40, 100), nn.LeakyReLU(), nn.Linear(100, 44))

    def forward(self, field, pieces):
        field = field[:,:,5:,:]
        padder = torch.ones(field.shape[0], 1, field.shape[2], 2, device=device)
        field_pad = torch.cat((padder, torch.cat((field, padder), 3)), 3)
        x = self.conv(field_pad)
        x = torch.cat((x, pieces), 1)
        x = self.linear(x)
        return x


class Random():
    def take_action(self, env):
        intersects = torch.nonzero(env.intersected).flatten().long()
        if len(intersects) > 0:
            return torch.ones(len(intersects), device=device).long()
        else:
            return None

