import random

class Figure:
    def __init__(self, figure, color, fig_num):
        self.x = 3
        self.y = 0
        self.rotation = 0
        self.figure = figure
        self.color = color
        self.fig_num = fig_num

    def image(self):
        return self.figure[self.rotation]

    def rotate_clock(self, amount):
        self.rotation = (self.rotation + amount) % len(self.figure)
        
    def rotate_counter_clock(self, amount):
        self.rotation = (self.rotation - amount) % len(self.figure)