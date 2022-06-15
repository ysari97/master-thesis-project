# Plotting functions
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

color_list = [
    "#2d3da3",
    "#1470d6",
    "#0195fb",
    "#00aaca",
    "#00bf8d",
    "#64d17e",
    "#edfb95",
    "#cbc98f",
    "#947567",
    "#765956",
]
month_list = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


class HydroModelPlotter:
    def __init__(
        self,
        hydro_model,
        colors=color_list,
        linewidth=3,
        n_months=12,
        months=month_list,
        figsize=(20, 6),
    ):
        self.hydro_model = hydro_model
        self.n_months = n_months
        self.months = month_list
        self.figsize = figsize
        self.n_years = int(self.hydro_model.simulation_horizon / n_months)

        # formatting related variables
        self.colors = colors
        self.palette = sns.set_palette(sns.color_palette(colors))
        self.cmap = mpl.colors.ListedColormap(colors)
        self.linewidth = linewidth
        self.limit_linestyle = ":"

        # naming conventions for the plots
        self.level_title = "Level (masl)"

    def plot_two_lines_together(
        self, vector1, label1, vector2, label2, title, x_title="", y_title=""
    ):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(vector1, label=label1, color=self.colors[0], linewidth=self.linewidth)
        ax.plot(vector2, label=label2, color=self.colors[5], linewidth=self.linewidth)
        ax.legend()
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)

        plt.title(title)
        plt.show()

    def line_graph_with_limits(
        self, vector1, label1, lb, ub, title, x_title="", y_title=""
    ):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(vector1, label=label1, color=self.colors[5], linewidth=self.linewidth)
        ax.legend()
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)

        ax.hlines(
            y=[lb, ub],
            linewidth=self.linewidth,
            xmin=0,
            xmax=240,
            color=self.colors[-1],
            linestyle=self.limit_linestyle,
        )

        plt.title(title)
        plt.show()

    def plot_received_vs_demand_for_district(self, irr_name):
        self.plot_two_lines_together(
            self.hydro_model.irr_districts[irr_name].received_flow,
            "Received",
            self.hydro_model.irr_districts[irr_name].demand,
            "Demand",
            f"{irr_name} demanded versus received water flow",
            "Months",
            "Water Flow (m3/s)",
        )

    def plot_level_with_limits(self, dam_name):
        self.line_graph_with_limits(
            self.hydro_model.reservoirs[dam_name].level_vector,
            f"{dam_name} Level",
            self.hydro_model.reservoirs[dam_name].rating_curve[0, 0],
            self.hydro_model.reservoirs[dam_name].rating_curve[0, -1],
            f"{dam_name} elevation overtime",
            "Months",
            "Level (masl)",
        )

    def plot_levels_condensed(self, dam_name):
        dam_level = self.hydro_model.reservoirs[dam_name].level_vector
        dam_level = np.reshape(dam_level, (self.n_years, self.n_months))
        avg = np.mean(dam_level, 0)
        mini = np.min(dam_level, 0)
        maxi = np.max(dam_level, 0)

        fig, ax = plt.subplots(figsize=self.figsize)
        # setting up limits
        ax.hlines(
            y=self.hydro_model.reservoirs[dam_name].rating_curve[0, 0],
            xmin=0,
            xmax=self.n_months - 1,
            linewidth=self.linewidth,
            color=self.colors[-1],
            linestyle=self.limit_linestyle,
        )
        ax.hlines(
            y=self.hydro_model.reservoirs[dam_name].rating_curve[0, -1],
            xmin=0,
            xmax=self.n_months - 1,
            linewidth=self.linewidth,
            color=self.colors[-1],
            linestyle=self.limit_linestyle,
        )
        ax.text(
            10,
            self.hydro_model.reservoirs[dam_name].rating_curve[0, 0],
            "Min level",
            bbox=dict(facecolor="gray", alpha=1),
            color="white",
            fontsize=8,
        )
        ax.text(
            10,
            self.hydro_model.reservoirs[dam_name].rating_curve[0, -1],
            "Max level",
            bbox=dict(facecolor="gray", alpha=1),
            color="white",
            fontsize=8,
        )

        # plotting min and max observations
        ax.fill_between(
            range(self.n_months), maxi, mini, alpha=0.5, color=self.colors[7]
        )
        # plotting the average
        ax.plot(avg, color=self.colors[8], linewidth=5)

        # setting up x and y axes, title
        ax.set_xticks(np.arange(self.n_months)) #, self.months, rotation=30)
        #ax.xlim([0, 11])
        ax.set_ylabel(self.level_title)

        plt.title(f"{dam_name} elevation summary per month")
        plt.show()
