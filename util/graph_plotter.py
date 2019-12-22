import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points, plot_name):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(plot_name + ".png")


def plot_data(x, y, xlabel = "x", ylabel = "y", label = 'plot'):
	plt.figure()
	plt.plot(x, y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	print("Generating plot for ", label)
	plt.savefig("./" + label + ".png")