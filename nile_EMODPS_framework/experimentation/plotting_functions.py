# Plotting functions
import matplotlib.pyplot as plt


def plot_two_lines_together(vector1, label1, vector2, label2, title):

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(vector1, label=label1)
    ax.plot(vector2, label=label2)
    ax.legend()
    plt.title(title)
    plt.show()


def line_graph_with_limits(vector1, label1, lb, ub, title):

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(vector1, label=label1)
    ax.legend()
    ax.hlines(y=[lb, ub], linewidth=2, xmin=0, xmax=240, color="r")
    plt.title(title)
    plt.show()
