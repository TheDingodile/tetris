from game import Tetris
from agent import Net
from replay_buffer import replay_buffer
from pynput import keyboard
from helpers import saveupdate

# Choose variables
player1 = "AI"
player2 = "random_sampler"
batch = 100
replay_size = 100000
sample_size = 256
env = Tetris(player1, player2, batch)
learn = False
replay_buffer = replay_buffer(replay_size, sample_size, learn)
learn_every = 1
start_learning_after = 50000

# Preprocess
intersects = [True for _ in range(batch)]
showPrint = False
if env.player1 == "AI":
    Agent = Net(batch)
def on_press(key):
    global showPrint, save
    if keyboard.Key.f2 == key:
        showPrint = True
    if keyboard.Key.f3 == key:
        save = True
keyboard.Listener(on_press=on_press).start()

# Gameloop
for counter in range(10**10):
    actions = Agent.take_action(env)
    obsBoard, obsPieces, intersects, rewards, dones = env.step(actions)
    replay_buffer.save_data([Agent.fields, Agent.pieces, obsBoard, obsPieces, Agent.actions, rewards, dones], intersects)
    fields, pieces, fields2, pieces2, actions, rewards, dones = replay_buffer.sample_data()
    Agent.DoubleQlearn(fields, pieces, fields2, pieces2, actions, rewards, dones, learn)
    saveupdate(counter, Agent, showPrint, env)