import matplotlib
import matplotlib.pyplot as plt

def returnplot(returns, x: int = 1000, y: int = 500, xlabel=None, ylabel="Return (per game)"):
    # runnings = [0] * len(returns)
    # for i in range(1, len(runnings) + 1):
    #     if i < 500:
    #         runnings[i - 1] = sum(returns[:i]) / i
    #     else:
    #         runnings[i - 1] = sum(returns[(i - 500):i]) / 500
    fig = plt.figure()
    move_figure(fig, x, y)
    plt.plot(returns)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show(block=False)

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)