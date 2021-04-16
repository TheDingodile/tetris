from numpy.lib.function_base import copy
from pieces import Figure
import random
import numpy as np
from UI import UI
import torch
from helpers import device

class Tetris:
    def __init__(self, player1, player2, batch, Use_UI=True, fps=5, high_performance=0, level=3):
        self.start_level = level
        self.batch = batch
        self.Use_UI = Use_UI
        self.high_performance = high_performance
        self.UI = UI()
        if self.Use_UI is True:
            self.UI.initilize()
        self.fps = fps
        self.pressing_down = False
        self.player1 = player1
        self.player2 = player2
        self.x = 300
        self.y = 200
        self.zoom = 20
        self.height = 20
        self.width = 10
        self.score = [0 for _ in range(self.batch)]
        self.done = False
        self.figure = [None for _ in range(self.batch)]
        self.all_lines = [0 for _ in range(self.batch)]
        self.all_tetrises = [0 for _ in range(self.batch)]
        self.pressing_down = False
        self.all_figures = [
        [[4, 5, 6, 7], [1, 5, 9, 13]],
        [[4, 5, 9, 10], [2, 6, 5, 9]],
        [[6, 7, 9, 10], [1, 5, 6, 10]],
        [[4, 5, 6, 10], [1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8]],
        [[3, 5, 6, 7], [1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11]],
        [[4, 5, 6, 9], [1, 5, 6, 9], [1, 4, 5, 6], [1, 4, 5, 9]],
        [[1, 2, 5, 6]],
    ]
        self.colors = [
        (244, 101, 40),
        (120, 37, 179),
        (64, 120, 211),
        (80, 34, 22),
        (80, 134, 22),
        (180, 34, 22),
        (180, 34, 122),
    ]
        self.level = level
        self.visible_pieces = 1

        # use "random_sampler" or "seven_bag"
        self.piece_sampler = player2
        self.seven_bag_counter = 0
        self.seven_bag = None
        self.next_pieces = [[] for _ in range(batch)]
        self.AInext_pieces = torch.zeros(batch, 7 * (1 + self.visible_pieces), device=device).long()
        self.action = 5
        self.field = torch.zeros(self.batch, 1, self.height, self.width, device=device).long()
        self.max_field = [[0 for _ in range(self.width)] for _ in range(self.batch)]

        for j in range(batch):
            for _ in range(self.visible_pieces):
                self.draw_figure(j)
            self.draw_figure(j)
        self.score_list = []
        self.done_list = []
        self.intersected = torch.ones(self.batch, device=device).long()
        self.sumscore = 0
        self.sumdones = 0

    def draw_figure(self, batch):
        if self.piece_sampler == "random_sampler":
            piece = self.randomsample()
        elif self.piece_sampler == "seven_bag":
             piece = self.seven_bag_sample()
        figure, color = self.all_figures[piece], self.colors[piece]
        self.next_pieces[batch].append(Figure(figure, color, piece))
        onehot = torch.zeros(7, device=device)
        onehot[piece] = 1
        for i in range(self.visible_pieces):
            self.AInext_pieces[batch, i*7:(i+1)*7] = self.AInext_pieces[batch, (i+1)*7:(i+2)*7]
        self.AInext_pieces[batch, self.visible_pieces*7:(self.visible_pieces + 1)*7] = onehot
        self.figure[batch] = self.next_pieces[batch][0]
        if len(self.next_pieces[batch]) > self.visible_pieces:
            self.next_pieces[batch] = self.next_pieces[batch][1:]

    def randomsample(self):
        return random.randint(0, len(self.all_figures) - 1)

    def seven_bag_sample(self):
        if self.seven_bag_counter == 0:
            self.seven_bag = np.random.permutation(np.arange(len(self.all_figures)))
        self.seven_bag_counter += 1
        if self.seven_bag_counter % len(self.all_figures) == 0:
            self.seven_bag_counter = 0
        return self.seven_bag[self.seven_bag_counter - 1]

    def intersects(self, batch):
        for item in self.figure[batch].image():
            i = item // 4
            j = item - i * 4
            height = i + self.figure[batch].y
            width = j + self.figure[batch].x
            if height > self.height - 1 or width > self.width - 1 or width < 0 or self.field[batch, 0, height, width] != 0:
                return True
        return False 

    def break_lines(self, int_idx):
        rewards = torch.zeros(self.batch, device=device)
        summed_field = torch.sum(self.field[int_idx], dim=3)
        summed_field[summed_field < 10] = 0
        breaks = torch.nonzero(summed_field)
        for i in range(breaks.shape[0]):
            batch = breaks[i, 0]
            line = breaks[i, 2]

            rewards[batch] += 1
            self.all_lines[batch] += 1

            roof = torch.cat((torch.zeros(1, self.field.shape[3], device=device), self.field[batch, 0, :line, :]), 0)
            self.field[batch, 0, :(line + 1), :] = roof
            self.max_field[batch] = [x - int(x >= (20 - line)) for x in self.max_field[batch]]
        rewards = rewards ** 2
        for i in range(self.batch):
            self.score[i] += rewards[i].item()
            self.all_tetrises[i] += (rewards[i] == 16).item()
        return rewards

    def go_space(self, batch):
        min_delta = self.height
        for item in self.figure[batch].image():
            i = item // 4
            j = item - i * 4
            height = i + self.figure[batch].y
            width = j + self.figure[batch].x
            delta = (self.height - height) - self.max_field[batch][width]
            if delta < min_delta:
                min_delta = delta
        self.figure[batch].y += min_delta
        self.figure[batch].y -= 1
        self.intersected[batch] = 1

        return False


    def go_down(self, batch):
        self.figure[batch].y += 1
        if self.intersects(batch):
            self.figure[batch].y -= 1
            self.intersected[batch] = 1

    def freeze(self, int_idx):
        dones = torch.zeros(self.batch, device=device)
        fakerewards = torch.zeros(self.batch, device=device)
        for k in int_idx:
            var_before = sum([abs(x[0] - x[1]) for x in zip(self.max_field[k][1:], self.max_field[k][:self.width - 1])])
            holes_before = torch.sum(torch.tensor(self.max_field[k], device=device) - torch.sum(self.field[k], dim=1), dim=1)
            for item in self.figure[k].image():
                i = item // 4
                j = item - i * 4 
                height = i + self.figure[k].y
                width = j + self.figure[k].x
                self.field[k, 0, height, width] = 1
                if self.height - height > self.max_field[k][width]:
                    self.max_field[k][width] = self.height - height
            var_after = sum([abs(x[0] - x[1]) for x in zip(self.max_field[k][1:], self.max_field[k][:self.width - 1])])
            holes_after = torch.sum(torch.tensor(self.max_field[k], device=device) - torch.sum(self.field[k], dim=1), dim=1)
            fakerewards[k] += (var_before - var_after) * 0.05 + (holes_before - holes_after).item() * 0.1

            self.draw_figure(k)
            dones[k] = self.game_over(k)
        return fakerewards, dones

    def game_over(self, batch):
        done = False
        if self.intersects(batch):
            done = True
            self.sumscore += self.score[batch]
            self.sumdones += 1
            print("gameover //", "score: " + str(self.score[batch]), "// games played: " + str(self.sumdones + 100 * len(self.score_list)))
            self.restart(batch)
            if self.sumdones == 100:
                self.score_list.append(self.sumscore/self.sumdones)
                self.sumscore = 0
                self.sumdones = 0
        return done

    def go_side(self, dx, batch):
        old_x = self.figure[batch].x
        self.figure[batch].x += dx
        if self.intersects(batch):
            self.figure[batch].x = old_x

    def rotate_clock(self, amount, batch):
        old_rotation = self.figure[batch].rotation
        self.figure[batch].rotate_clock(amount)
        if self.intersects(batch):
            self.figure[batch].rotation = old_rotation

    def level_up(self):
        if self.all_lines // 10 > self.level:
            self.level += 1

    def step(self, action, intersects):
        if action != None:
            for inter, act in zip(torch.nonzero(intersects).flatten().long(), action):
                rotate = act//11
                self.rotate_clock(rotate, inter)
                self.go_side((act - 11 * rotate) - 5, inter)

        self.intersected = torch.zeros(self.batch, device=device)
        for i in range(self.batch):
            if self.player1 == "human":
                self.go_down(i)
            else:
                self.go_space(i)

        intersects_idx = torch.nonzero(self.intersected).flatten().long()
        fakerewards, dones = self.freeze(intersects_idx)
        rewards = self.break_lines(intersects_idx)

        if self.Use_UI is True:
            if self.high_performance == 1:
                self.UI.action(self)
            else:
                self.UI.draw_step(self)
                if self.player1 == "human":
                    self.UI.clock.tick(self.fps)
        return self.field, self.AInext_pieces, self.intersected, rewards + fakerewards, dones

    def restart(self, batch):
        self.figure[batch] = None
        self.score[batch] = 0
        self.all_lines[batch] = 0
        self.all_tetrises[batch] = 0
        self.level = self.start_level
        self.seven_bag_counter = 0
        self.next_pieces[batch] = []
        self.AInext_pieces[batch] = torch.zeros(7 * (1 + self.visible_pieces), device=device).long()
        self.field[batch] = torch.zeros(1, self.height, self.width, device=device).long()

        for _ in range(self.visible_pieces):
            self.draw_figure(batch)
        self.draw_figure(batch)
        self.max_field[batch] = [0 for _ in range(self.width)]