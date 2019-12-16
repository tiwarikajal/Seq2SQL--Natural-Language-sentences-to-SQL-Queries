import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# plt.switch_backend('TkAgg')


def showPlot(points, plot_name):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(plot_name + ".png")
