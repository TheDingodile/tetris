from helpers import device
from game import Tetris
from agent import Net, Random, Human
from replay_buffer import replay_buffer
from pynput import keyboard
from helpers import saveupdate
import torch

# Choose variables
player1 = "AI"
player2 = "random_sampler"
batch = 100
replay_size = 100000
sample_size = 64
env = Tetris(player1, player2, batch)
learn = True
replay_buffer = replay_buffer(replay_size, sample_size, learn)
learn_every = 1
showPrint = False

# Preprocess
intersects = torch.ones(batch, device=device)
if env.player1 == "AI":
    Agent = Net(batch, env.height, env.width)
elif env.player1 == "random":
    Agent = Random()
elif env.player1 == "human":
    Agent = Human()
def on_press(key):
    global showPrint, save
    if keyboard.Key.f2 == key:
        showPrint = True
    if keyboard.Key.f3 == key:
        save = True
    if keyboard.Key.f4 == key:
        showPrint = False
keyboard.Listener(on_press=on_press).start()

# Gameloop
for counter in range(10**10):
    actions = Agent.take_action(env)
    obsBoard, obsPieces, intersects, rewards, dones = env.step(actions, intersects)
    #replay_buffer.save_data(Agent.fields, Agent.pieces, obsBoard, obsPieces, Agent.actions, rewards, dones, intersects)
    #fields, pieces, fields2, pieces2, actions, rewards, dones = replay_buffer.sample_data()
    Agent.DoubleQlearn(Agent.fields, Agent.pieces, obsBoard, obsPieces, Agent.actions, rewards, dones, learn)
    showPrint = saveupdate(counter, Agent, showPrint, env)