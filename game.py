from numpy.lib.function_base import copy
from pieces import Figure
import random
import numpy as np
from UI import UI
import torch
from helpers import device

class Tetris:
    def __init__(self, player1, player2, batch, Use_UI=True, fps=30, high_performance=0, level=3):
        self.batch = batch
        self.start_level = level
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
        self.score = 0
        self.mean_score_list = [0]
        self.done = False
        self.figure = [None for _ in range(batch)]
        self.all_lines = 0
        self.all_tetrises = 0
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
        self.level = self.start_level
        self.visible_pieces = 1

        # use "random_sampler" or "seven_bag"
        self.piece_sampler = player2
        self.seven_bag_counter = 0
        self.seven_bag = None
        self.next_pieces = [[] for _ in range(batch)]
        self.AInext_pieces = torch.zeros(batch, 7 * (1 + self.visible_pieces), device=device).long()
        self.action = 5
        self.field = torch.zeros(self.batch, 1, self.height, self.width, device=device).long()

        for j in range(batch):
            for _ in range(self.visible_pieces):
                self.draw_figure(j)
            self.draw_figure(j)
        self.score_list = []
        self.intersected = torch.ones(self.batch, device=device).long()

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
            self.AInext_pieces[batch, i] = self.AInext_pieces[batch, i + 1]
        self.figure[batch] = self.next_pieces[batch][0]
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
        intersection = False
        for item in self.figure[batch].image():
            i = item % 4
            j = item - i * 4
            if i + self.figure[batch].y > self.height - 1 or \
                    j + self.figure[batch].x > self.width - 1 or \
                    j + self.figure[batch].x < 0 or \
                    self.field[batch, 0, i + self.figure[batch].y, j + self.figure[batch].x] != 0:
                intersection = True
        return intersection



    def break_lines(self, batch):
        maxer = [0] * 10
        lines = 0
        broke = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[batch, 0, i, j] == 0:
                    zeros += 1
                elif maxer[j] == 0:
                    maxer[j] = self.height - i
            if zeros == 0:
                broke = 1
                lines += 1
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[batch, 0, i1, j] = self.field[batch, 0, i1 - 1, j]
        self.all_lines += lines
        if lines == 4:
            self.all_tetrises += 1
        reward = lines ** 2
        self.score += lines ** 2
        self.level_up()
        return reward

    def go_space(self, batch):
        while not self.intersects():
            self.figure[batch].y += 1
        self.figure[batch].y -= 1

    def go_down(self, batch):
        self.figure[batch].y += 1
        if self.intersects(batch):
            self.figure[batch].y -= 1
            self.intersected[batch] = 1

    def freeze(self):
        rewards = torch.zeros(self.batch, device=device)
        dones = torch.zeros(self.batch, device=device)
        for k in range(self.batch):
            if self.intersected[k] == 1:
                for item in self.figure[k].image():
                    i = item % 4
                    j = item - i * 4
                    self.field[k, 0, i + self.figure[i].y, j + self.figure[i].x] = 1
                    self.draw_figure(k)
                    rewards[k] = self.break_lines(k)
                    dones[k] = self.game_over(k)

        return rewards, dones

    def game_over(self, batch):
        done = False
        if self.intersects(batch):
            done = True
            print("gameover //", "score: " + str(self.score) + " //", "lines: " + str(self.all_lines))
            self.score_list.append(self.score)
            if len(self.mean_score_list) == 1 and len(self.score_list) > 1000:
                self.mean_score_list.append(sum(self.score_list)/1000)
            if len(self.score_list) > 1000:
                    self.mean_score_list.append((self.mean_score_list[-1]+(self.score_list[-1] - self.score_list[-1000])/1000)) 
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
        self.intersected = torch.zeros(self.batch, device=device)
        j = 0
        for i in range(self.batch):
            if intersects[i] == 1:
                rotate = action[j]//11
                self.rotate_clock(rotate, i)
                self.go_side(action[j] * 11 - rotate, i)
                j += 1
            self.go_down(i)

        rewards, dones = self.freeze()
        if self.Use_UI is True:
            if self.high_performance == 1:
                self.UI.action(self)
            else:
                self.UI.draw_step(self)
                self.UI.clock.tick(self.fps)
        return self.field, self.next_pieces, self.intersected, rewards, dones

    def restart(self):
        self.score = 0
        self.done = False
        self.figure = [None for _ in range(self.batch)]
        self.all_lines = 0
        self.all_tetrises = 0
        self.level = self.start_level

        self.seven_bag_counter = 0
        self.seven_bag = None
        self.next_pieces = []
        self.AInext_pieces = torch.zeros(self.batch, 7, device=device)
        self.action = 5

        self.field = torch.zeros(self.height, self.width, device=device)
        for j in range(self.batch):
            for _ in range(self.visible_pieces):
                self.draw_figure(j)
            self.draw_figure(j)

