from torch import as_tensor as tensor, cat as concatenation, device as devicer, cuda, float32
from saver import saveAgent, loadagent
import matplotlib.pyplot as plt
from plots import returnplot

def Zero_dividor(x, y):
    return "100 %" if y == 0 else str(int(x / y)) + " %"

device = devicer('cpu')

def saveupdate(counter, Agent, showPrint, env):
    if counter % 100 == 0:
        print("frame so far: " + str(counter))
        if Agent.__class__.__name__ == "Net":
            Agent.update_target_network()
            saveAgent(Agent, "yo")
    if showPrint:
        plt.close('all')
        returnplot(env.score_list, x=1200, y=200, xlabel="games played in 100's", ylabel="Score (per game)")
        returnplot(env.reward_list, x=600, y=200, xlabel="games played in 100's", ylabel="Return (per game)")
        return False