import pickle
def saveAgent(agent, name: str):
    pickle.dump(agent, open(f"trained/{'-'.join(name.split('-')[:-1])}/{name}", "wb"))

def loadagent(name):
    agent = pickle.load(open(f"trained/{'-'.join(name.split('-')[:-1])}/{name}", "rb"))
    return agent