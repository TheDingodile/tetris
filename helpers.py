from torch import as_tensor as tensor, cat as concatenation, device as devicer, cuda, float32
from saver import saveAgent, loadagent
import matplotlib.pyplot as plt
from plots import returnplot

def Zero_dividor(x, y):
    return "100 %" if y == 0 else str(int(x / y)) + " %"

device = devicer('cuda' if cuda.is_available() else 'cpu')

def saveupdate(counter, Agent, showPrint, env):
    if counter % 10000 == 0:
        print("frame so far: " + str(counter))
        print("A fake reward is currently worth: " + str(1/(8 + env.mean_score_list[-1])))
        Agent.update_target_network()
        saveAgent(Agent, "yo")
    if showPrint:
        plt.close('all')
        returnplot(env.score_list, x=1200, y=200)
        return False