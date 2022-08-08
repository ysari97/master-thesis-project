# Plotting functions
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas.plotting import parallel_coordinates
from matplotlib.lines import Line2D
import pandas as pd
import itertools
from collections import defaultdict

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

theme_colors = defaultdict(lambda x: "black")
theme_colors.update(
    {
        "darkblue": "#2d3da3",
        "blue": "#0195fb",
        "turquise": "#00bf8d",
        "lightyellow": "#edfb95",
        "beige": "#cbc98f",
        "brown": "#765956",
        "purple": "#6C0C86",
        "red": "red",
        "pink": "#c51b7d",
        "plum": "orchid",
        "gray": "#bdbdbd",
        "green": "#41ab5d",
        "yellow": "#fdaa09",
        "maroon": "#800000",
        "indianred": "indianred",
        "chocolate": "chocolate",
    }
)

country_color = defaultdict(lambda x: "black")
country_color.update(
    {
        "Ethiopia": theme_colors["green"],
        "Sudan": theme_colors["purple"],
        "Sudan-2": theme_colors["plum"],
        "Egypt": theme_colors["chocolate"],
    }
)

dam_color = defaultdict(lambda x: "black")
dam_color = {
    "GERD": country_color["Ethiopia"],
    "Roseires": country_color["Sudan"],
    "Sennar": country_color["Sudan-2"],
    "HAD": country_color["Egypt"],
}

irr_color = defaultdict(lambda x: "black")
irr_color = {"Gezira": country_color["Sudan"], "Egypt": country_color["Egypt"]}


def normalize_objs(df, directions):
    desirability_couples = list()
    working_df = df.copy()
    for i, col in enumerate(df.columns):
        if directions[i] == "min":
            best, worst = df[col].min(), df[col].max()
        elif directions[i] == "max":
            best, worst = df[col].max(), df[col].min()
        desirability_couples.append((worst, best))
        working_df[col] = (df[col] - worst) / (best - worst)

    return working_df, desirability_couples


def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def parallel_plots_many_policies(
    obj_df,
    solution_indices=[],
    solution_names=[],
    names_display=[
        "Egypt Irr. Deficit",
        "Egypt 90$^{th}$ Irr. Deficit",
        "Egypt Low HAD",
        "Sudan Irr. Deficit",
        "Sudan 90$^{th}$ Irr. Deficit",
        "Ethiopia Hydropower",
    ],
    units=["BCM/year", "BCM/month", "%", "BCM/year", "BCM/month", "TWh/year"],
    directions=["min", "min", "min", "min", "min", "max"],
):

    names = list(obj_df.columns)

    objectives_df = obj_df.copy()
    objectives_df.egypt_low_had = 100 * (objectives_df.egypt_low_had)

    norm_df, desirability_couples = normalize_objs(objectives_df, directions)

    uds = []  # undesired
    ds = []  # desired
    for i in desirability_couples:
        uds.append(str(round(i[0], 1)))
        ds.append(str(round(i[1], 1)))

    norm_df["Name"] = "All Solutions"
    for i, solution_index in enumerate(solution_indices):
        norm_df.loc[solution_index, "Name"] = solution_names[i]
        norm_df = norm_df.append(norm_df.loc[solution_index, :].copy())

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    parallel_coordinates(
        norm_df,
        "Name",
        color=[
            theme_colors["gray"],
            theme_colors["green"],
            theme_colors["plum"],
            theme_colors["purple"],
            theme_colors["chocolate"],
            theme_colors["yellow"],
            theme_colors["blue"],
            "red",
        ],
        linewidth=7,
        alpha=0.8,
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    handles_dict = dict(zip(labels, handles))
    labels = ["All Solutions"] + solution_names

    plt.legend(
        flip([handles_dict[label] for label in labels], 4),
        flip(labels, 4),
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=4,
        mode="expand",
        borderaxespad=1.5,
        fontsize=22,
    )

    ax1.set_xticks(np.arange(len(names)))

    ax1.set_xticklabels(
        [
            uds[i] + "\n" + "\n" + names_display[i] + "\n[" + units[i] + "]"
            for i in range(len(names))
        ],
        fontsize=22,
    )
    ax2 = ax1.twiny()
    ax2.set_xticks(np.arange(len(names)))
    ax2.set_xticklabels([ds[i] for i in range(len(names))], fontsize=22)

    ax1.get_yaxis().set_visible([])
    plt.text(
        1.02,
        0.5,
        "Direction of Preference $\\rightarrow$",
        {"color": "#636363", "fontsize": 26},
        horizontalalignment="left",
        verticalalignment="center",
        rotation=90,
        clip_on=False,
        transform=plt.gca().transAxes,
    )

    fig.set_size_inches(25, 12)


def parallel_plots_few_policies(
    obj_df, directions=["min", "min", "min", "min", "min"], solution_names=[]
):

    names = list(obj_df.columns)

    names_display = [
        "Egypt Irr. Deficit",
        "Egypt 90$^{th}$ Irr. Deficit",
        "Egypt Low HAD",
        "Sudan Irr. Deficit",
        "Ethiopia Hydropower",
    ]
    units = ["BCM/year", "BCM/month", "%", "BCM/year", "TWh/year"]

    objectives_df = obj_df.copy()
    objectives_df.egypt_low_had = 100 * (objectives_df.egypt_low_had)

    norm_df, desirability_couples = normalize_objs(objectives_df, directions)

    uds = []  # undesired
    ds = []  # desired
    for i in desirability_couples:
        uds.append(str(round(i[0], 1)))
        ds.append(str(round(i[1], 1)))

    if solution_names == []:
        norm_df["Name"] = obj_df.index
    else:
        norm_df["Name"] = solution_names
        print(norm_df["Name"])

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    parallel_coordinates(
        norm_df,
        "Name",
        color=[  # theme_colors["gray"],
            theme_colors["pink"],
            theme_colors["yellow"],
            theme_colors["blue"],
            theme_colors["purple"],
            theme_colors["green"],
            # theme_colors["brown"],
            "red",
        ],
        linewidth=7,
        alpha=0.8,
    )
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=4,
        mode="expand",
        borderaxespad=1.5,
        fontsize=22,
    )

    ax1.set_xticks(np.arange(len(names)))

    ax1.set_xticklabels(
        [
            uds[i] + "\n" + "\n" + names_display[i] + "\n[" + units[i] + "]"
            for i in range(len(names))
        ],
        fontsize=22,
    )
    ax2 = ax1.twiny()
    ax2.set_xticks(np.arange(len(names)))
    ax2.set_xticklabels([ds[i] for i in range(len(names))], fontsize=22)

    ax1.get_yaxis().set_visible([])
    plt.text(
        1.02,
        0.5,
        "Direction of Preference $\\rightarrow$",
        {"color": "#636363", "fontsize": 26},
        horizontalalignment="left",
        verticalalignment="center",
        rotation=90,
        clip_on=False,
        transform=plt.gca().transAxes,
    )

    fig.set_size_inches(25, 12)
    plt.show()


class HydroModelPlotter:
    def __init__(
        self,
        hydro_model,
        colors=color_list,
        linewidth=4,
        horizontal_line_width=3,
        n_months=12,
        months=month_list,
        figsize=(12, 8),
        landscape_figsize=(24, 8),
    ):
        self.hydro_model = hydro_model
        self.n_months = n_months
        self.months = months
        self.figsize = figsize
        self.landscape_figsize = landscape_figsize
        self.n_years = int(self.hydro_model.simulation_horizon / n_months)

        # formatting related variables
        self.colors = colors
        self.palette = sns.set_palette(sns.color_palette(colors))
        self.cmap = mpl.colors.ListedColormap(colors)
        self.linewidth = linewidth
        self.horizontal_line_width = horizontal_line_width
        self.limit_linestyle = ":"

        # naming conventions for the plots
        self.level_title = "Level (masl)"

    def plot_line(
        self, vector1, label1, title="", x_title="", y_title="", ax=None, color=None
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.set_xlabel(x_title)
            ax.set_ylabel(y_title)

        if color is None:
            color = self.colors[0]

        ax.plot(vector1, label=label1, color=color, linewidth=self.linewidth)
        ax.legend()

        plt.title(title)
        plt.show()

        return ax

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

    def plot_multiple_lines_together(
        self, vector_list, label_list, title, x_title="", y_title="", colors=color_list
    ):
        fig, ax = plt.subplots(figsize=self.landscape_figsize)

        for i, vector in enumerate(vector_list):
            ax.plot(
                vector, label=label_list[i], color=colors[i], linewidth=self.linewidth
            )

        ax.legend()
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)

        plt.title(title)
        plt.show()

    def line_graph_with_limits(
        self,
        vectors,
        vector_labels,
        vector_colors,
        horizontal_lines=[],
        horline_labels=[],
        horline_colors=[],
        title="",
        x_label="",
        y_label="",
        x_tick_frequency=5,
        ax=None,
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        for i, vector in enumerate(vectors):
            ax.plot(
                vector,
                label=vector_labels[i],
                color=vector_colors[i],
                linewidth=self.linewidth,
            )

        for i, line in enumerate(horizontal_lines):
            if line is not None:
                ax.hlines(
                    y=line,
                    linewidth=self.horizontal_line_width,
                    xmin=0,
                    xmax=self.n_years * self.n_months,
                    label=horline_labels[i],
                    color=horline_colors[i],
                    linestyle=self.limit_linestyle,
                )

        ax.legend()
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.set_xticks(
            np.arange(
                0, self.n_years * self.n_months + 1, x_tick_frequency * self.n_months
            )
        )
        ax.set_xticklabels(
            [
                f"Jan-{2022+i*x_tick_frequency}"
                for i in range(int(self.n_years / x_tick_frequency) + 1)
            ],
            fontsize=14,
        )
        plt.title(title)
        # plt.show()

        return ax

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

    def plot_received_vs_demand_for_district_raw(self, irr_name):
        self.plot_two_lines_together(
            self.hydro_model.irr_districts[irr_name].received_flow_raw,
            "Received",
            self.hydro_model.irr_districts[irr_name].demand,
            "Demand",
            f"{irr_name} demanded versus received water flow",
            "Months",
            "Water Flow (m3/s)",
        )

    def plot_level_with_limits(self, dam_name):
        if dam_name == "HAD":
            threshold = 159
        else:
            threshold = None

        self.line_graph_with_limits(
            vectors=[self.hydro_model.reservoirs[dam_name].level_vector],
            vector_labels=[f"{dam_name} Level"],
            vector_colors=[dam_color[dam_name]],
            horizontal_lines=[
                self.hydro_model.reservoirs[dam_name].rating_curve[0, 1],
                self.hydro_model.reservoirs[dam_name].rating_curve[0, -1],
                threshold,
            ],
            horline_labels=["Min/Max Level", "", "Minimum Operational Level"],
            horline_colors=["silver", "silver", "black"],
            x_label="Months",
            y_label="Level (masl)",
        )

    def plot_release(self, dam_name, label="", ax=None, color=None):
        if label == "":
            label = f"{dam_name} Release"

        return self.plot_line(
            self.hydro_model.reservoirs[dam_name].release_vector,
            label,
            x_title="Months",
            y_title="Release ($m^{3}$/s)",
            ax=ax,
            color=color,
        )

    def plot_inflow(self, dam_name, label="", ax=None, color=None):
        if label == "":
            label = f"{dam_name} Inflow"

        return self.plot_line(
            self.hydro_model.reservoirs[dam_name].inflow_vector,
            label,
            x_title="Months",
            y_title="Inflow ($m^{3}$/s)",
            ax=ax,
            color=color,
        )

    def plot_release_vs_inflow(self, dam_name):
        self.plot_two_lines_together(
            self.hydro_model.reservoirs[dam_name].inflow_vector,
            "Inflow",
            self.hydro_model.reservoirs[dam_name].release_vector,
            "Release",
            f"Inflow to versus release water flow from {dam_name}",
            "Months",
            "Water Flow (m3/s)",
        )

    def plot_condensed_figure(
        self,
        vectors,
        y_name,
        title="",
        labels=[],
        range_exists=[True],
        colors=color_list,
        range_alphas=[0.5],
        hor_line_positions=[],
        hor_line_colors=["grey", "grey", "grey"],
        text_on_horiz=[],
        bbox_to_anchor=(1, 1),
    ):

        fig, ax = plt.subplots(figsize=self.figsize)

        for i, vector in enumerate(vectors):
            vector = np.reshape(vector, (self.n_years, self.n_months))
            avg = np.mean(vector, 0)
            # plotting the average
            ax.plot(avg, color=colors[i], linewidth=self.linewidth, label=labels[i])
            if range_exists[i]:
                mini = np.min(vector, 0)
                maxi = np.max(vector, 0)
                # plotting min and max observations
                ax.fill_between(
                    range(self.n_months),
                    maxi,
                    mini,
                    alpha=range_alphas[i],
                    color=colors[i],
                )

        # setting up horizontal lines
        for i, h in enumerate(hor_line_positions):
            if h is not None:
                ax.hlines(
                    y=h,
                    xmin=0,
                    xmax=self.n_months - 1,
                    linewidth=self.horizontal_line_width,
                    color=hor_line_colors[i],
                    linestyle=self.limit_linestyle,
                    label=text_on_horiz[i],
                )

        # setting up x and y axes, title
        ax.set_xticks(np.arange(self.n_months))  # , self.months, rotation=30)
        ax.set_xticklabels(self.months)
        ax.set_ylabel(y_name, fontsize=16)
        ax.set_xlabel("Month", fontsize=16)
        ax.legend(loc="upper right", bbox_to_anchor=bbox_to_anchor)
        plt.title(title)
        # plt.show()

    def plot_condensed_demand(self, irr_district):
        self.plot_condensed_figure(
            [self.hydro_model.irr_districts[irr_district].demand], "Demand [m3/sec]"
        )

    def plot_condensed_inflow(
        self,
        dam_name,
    ):
        self.plot_condensed_figure(
            [self.hydro_model.reservoirs[dam_name].inflow_vector], "Inflow [m3/sec]"
        )

    def plot_condensed_release(self, dam_name):
        hor_line_positions = [
            self.hydro_model.reservoirs[dam_name].hydropower_plants[0].max_turbine_flow
        ]
        text_on_horiz = ["Turbined Release"]
        self.plot_condensed_figure(
            [self.hydro_model.reservoirs[dam_name].release_vector],
            "Release [m3/sec]",
            hor_line_positions=hor_line_positions,
            text_on_horiz=text_on_horiz,
        )

    def plot_condensed_release_versus_inflow(self, dam_name):
        hor_line_positions = [
            self.hydro_model.reservoirs[dam_name].hydropower_plants[0].max_turbine_flow
        ]
        text_on_horiz = ["Max Turbine Release"]

        if dam_name == "GERD":
            inflow_label = "Blue Nile Inflow"
        else:
            inflow_label = f"Inflow"

        self.plot_condensed_figure(
            vectors=[
                self.hydro_model.reservoirs[dam_name].release_vector,
                self.hydro_model.reservoirs[dam_name].inflow_vector,
            ],
            y_name="Flow [m3/sec]",
            labels=[f"{dam_name} Release", inflow_label],
            range_exists=[True, True],
            range_alphas=[0.5, 0.3],
            colors=[dam_color[dam_name], color_list[1]],
            hor_line_positions=hor_line_positions,
            text_on_horiz=text_on_horiz,
        )

    def plot_condensed_level(
        self,
        dam_name,
    ):

        if dam_name == "HAD":
            threshold = 159
            bbox_to_anchor = (0.32, 0.22)
        else:
            threshold = None
            bbox_to_anchor = (1, 0.93)

        hor_line_positions = [
            self.hydro_model.reservoirs[dam_name].rating_curve[0, 1],
            self.hydro_model.reservoirs[dam_name].rating_curve[0, -1],
            threshold,
        ]
        text_on_horiz = ["Min/Max Level", "", "Minimum Operational Level"]
        self.plot_condensed_figure(
            vectors=[self.hydro_model.reservoirs[dam_name].level_vector],
            y_name="Level [masl]",
            labels=[f"{dam_name} Level"],
            colors=[dam_color[dam_name]],
            hor_line_positions=hor_line_positions,
            text_on_horiz=text_on_horiz,
            hor_line_colors=["silver", "silver", "black"],
            bbox_to_anchor=bbox_to_anchor,
        )

    # WE USE THIS!!!!!!!
    def plot_received_vs_demand_for_district_raw_condensed(self, irr_name):

        self.plot_condensed_figure(
            vectors=[
                self.hydro_model.irr_districts[irr_name].demand,
                self.hydro_model.irr_districts[irr_name].received_flow_raw,
            ],
            y_name="Flow [m3/sec]",
            labels=[f"{irr_name} Demanded Flow", f"{irr_name} Received Flow"],
            range_exists=[True, True],
            range_alphas=[0.3, 0.3],
            colors=[irr_color[irr_name], color_list[1]],
        )
