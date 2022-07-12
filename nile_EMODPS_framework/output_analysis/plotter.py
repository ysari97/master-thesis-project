# Plotting functions
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas.plotting import parallel_coordinates
from matplotlib.lines import Line2D
import pandas as pd

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

theme_colors = {
    "darkblue": "#2d3da3",
    "blue": "#0195fb", "turquise": "#00bf8d",
    "lightyellow": "#edfb95", "beige": "#cbc98f",
    "brown": "#765956", "purple": "#6C0C86",
    "red": "red", "pink": "#c51b7d", "gray": '#bdbdbd',
    "green": '#41ab5d', "yellow": "#fdaa09"
}

def normalize_objs(df, directions):
    desirability_couples = list()
    working_df = df.copy()
    for i, col in enumerate(df.columns):
        if directions[i] == "min": best, worst = df[col].min(), df[col].max()
        elif directions[i] == "max": best, worst = df[col].max(), df[col].min()
        desirability_couples.append((worst, best))
        working_df[col] = (df[col] - worst) / (best - worst)
        
    return working_df, desirability_couples

def parallel_plots_many_policies(
    obj_df, solution_indices = [], solution_names = [], directions=["min", "min", "min", "min", "max"]
):
    file_name='Best_objectives'

    names= list(obj_df.columns)
    
    names_display = ['Egypt Irr. Deficit','Egypt 90$^{th}$ Irr. Deficit','Egypt Low HAD','Sudan Irr. Deficit','Ethiopia Hydropower']
    units=['BCM/year','m3/s','%','BCM/year','TWh/year']
    
    
    objectives_df = obj_df.copy()
    #objectives_df.egypt_irr = m3s_to_bcm_per_year(objectives_df.egypt_irr)
    #objectives_df.sudan_irr = m3s_to_bcm_per_year(objectives_df.sudan_irr)
    objectives_df.egypt_low_had = 100*(objectives_df.egypt_low_had)
    
    norm_df, desirability_couples = normalize_objs(objectives_df, directions)
    
    
    
    uds=[] #undesired
    ds=[] #desired
    for i in desirability_couples:
        uds.append(str(round(i[0], 1)))
        ds.append(str(round(i[1], 1)))
    
    norm_df['Name'] = "All Solutions"
    for i, solution_index in enumerate(solution_indices):
        norm_df.loc[solution_index, "Name"] = solution_names[i]
        norm_df = norm_df.append(norm_df.loc[solution_index,:].copy())
    
    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    parallel_coordinates(norm_df,'Name', color= [theme_colors["gray"],
                                                 theme_colors["pink"],
                                                 theme_colors["yellow"],
                                                 theme_colors["blue"],
                                                 theme_colors["purple"],
                                                 theme_colors["green"],
                                                 #theme_colors["brown"],
                                                 "red"], linewidth=7, alpha=.8)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=1.5, fontsize=22)
    
    ax1.set_xticks(np.arange(len(names)))
    
    ax1.set_xticklabels([uds[i]+'\n'+'\n'+names_display[i]+'\n['+units[i] + "]" for i in range(len(names))],
                        fontsize=22)
    ax2 = ax1.twiny()
    ax2.set_xticks(np.arange(len(names)))
    ax2.set_xticklabels([ds[i] for i in range(len(names))], 
                        fontsize=22)
    
    ax1.get_yaxis().set_visible([])
    plt.text(1.02, 0.5, 'Direction of Preference $\\rightarrow$', {'color': '#636363', 'fontsize': 26},
             horizontalalignment='left',
             verticalalignment='center',
             rotation=90,
             clip_on=False,
             transform=plt.gca().transAxes)

    fig.set_size_inches(25, 12)
    plt.show()
    
def parallel_plots_few_policies(
    obj_df, directions=["min", "min", "min", "min", "min"], solution_names = []
):
    file_name='Best_objectives'

    names= list(obj_df.columns)
    
    names_display = ['Egypt Irr. Deficit','Egypt 90$^{th}$ Irr. Deficit','Egypt Low HAD','Sudan Irr. Deficit','Ethiopia Hydropower']
    units=['BCM/year','m3/s','%','BCM/year','TWh/year']
    
    
    objectives_df = obj_df.copy()
    #objectives_df.egypt_irr = m3s_to_bcm_per_year(objectives_df.egypt_irr)
    #objectives_df.sudan_irr = m3s_to_bcm_per_year(objectives_df.sudan_irr)
    objectives_df.egypt_low_had = 100*(objectives_df.egypt_low_had)
    
    norm_df, desirability_couples = normalize_objs(objectives_df, directions)
    
    uds=[] #undesired
    ds=[] #desired
    for i in desirability_couples:
        uds.append(str(round(i[0], 1)))
        ds.append(str(round(i[1], 1)))
    
    if solution_names == []: norm_df['Name'] = obj_df.index
    else:
        norm_df['Name'] = solution_names
        print(norm_df['Name'])
#     for i, solution_index in enumerate(solution_indices):
#         norm_df.loc[solution_index, "Name"] = solution_names[i]
#         norm_df = norm_df.append(norm_df.loc[solution_index,:].copy())
    
    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    parallel_coordinates(norm_df,'Name', color= [#theme_colors["gray"],
                                                 theme_colors["pink"],
                                                 theme_colors["yellow"],
                                                 theme_colors["blue"],
                                                 theme_colors["purple"],
                                                 theme_colors["green"],
                                                 #theme_colors["brown"],
                                                 "red"], linewidth=7, alpha=.8)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=1.5, fontsize=22)
    
    ax1.set_xticks(np.arange(len(names)))
    
    ax1.set_xticklabels([uds[i]+'\n'+'\n'+names_display[i]+'\n['+units[i] + "]" for i in range(len(names))],
                        fontsize=22)
    ax2 = ax1.twiny()
    ax2.set_xticks(np.arange(len(names)))
    ax2.set_xticklabels([ds[i] for i in range(len(names))], 
                        fontsize=22)
    
    ax1.get_yaxis().set_visible([])
    plt.text(1.02, 0.5, 'Direction of Preference $\\rightarrow$', {'color': '#636363', 'fontsize': 26},
             horizontalalignment='left',
             verticalalignment='center',
             rotation=90,
             clip_on=False,
             transform=plt.gca().transAxes)

    fig.set_size_inches(25, 12)
    plt.show()


class HydroModelPlotter:
    def __init__(
        self,
        hydro_model,
        colors=color_list,
        linewidth=3,
        n_months=12,
        months=month_list,
        figsize=(12, 8),
        landscape_figsize = (24, 8)
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
            ax.plot(vector, label=label_list[i], color=colors[i], linewidth=self.linewidth)

        ax.legend()
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)

        plt.title(title)
        plt.show()


        
    def line_graph_with_limits(
        self, vector1, label1, lb, ub, threshold, title, x_title="", y_title="", ax=None
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            
        ax.plot(vector1, label=label1, color=self.colors[5], linewidth=self.linewidth)
        ax.legend()
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        
        if threshold != 0:
            ax.hlines(
                y=[threshold],
                linewidth=self.linewidth,
                xmin=0,
                xmax=240,
                color="red",
                linestyle=self.limit_linestyle,
            )

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
        if dam_name == "HAD": threshold = 147
        else: threshold = 0
        
        self.line_graph_with_limits(
            self.hydro_model.reservoirs[dam_name].level_vector,
            f"{dam_name} Level",
            self.hydro_model.reservoirs[dam_name].rating_curve[0, 0],
            self.hydro_model.reservoirs[dam_name].rating_curve[0, -1],
            threshold,
            f"{dam_name} elevation overtime",
            "Months",
            "Level (masl)",
        )
    
    def plot_release(self, dam_name, label="", ax=None,color=None):
        if label == "":
            label = f"{dam_name} Release"
        
        return self.plot_line(
            self.hydro_model.reservoirs[dam_name].release_vector,
            label,
            x_title="Months",
            y_title="Release ($m^{3}$/s)",
            ax=ax,
            color=color
        )

    def plot_inflow(self, dam_name, label="", ax=None,color=None):
        if label == "":
            label = f"{dam_name} Inflow"
        
        return self.plot_line(
            self.hydro_model.reservoirs[dam_name].inflow_vector,
            label,
            x_title="Months",
            y_title="Inflow ($m^{3}$/s)",
            ax=ax,
            color=color
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
    
        
    # def plot_levels_condensed(self, dam_name):
    #     dam_level = self.hydro_model.reservoirs[dam_name].level_vector
    #     dam_level = np.reshape(dam_level, (self.n_years, self.n_months))
    #     avg = np.mean(dam_level, 0)
    #     mini = np.min(dam_level, 0)
    #     maxi = np.max(dam_level, 0)
    #
    #     fig, ax = plt.subplots(figsize=self.figsize)
    #     # setting up limits
    #     ax.hlines(
    #         y=self.hydro_model.reservoirs[dam_name].rating_curve[0, 0],
    #         xmin=0,
    #         xmax=self.n_months - 1,
    #         linewidth=self.linewidth,
    #         color=self.colors[-1],
    #         linestyle=self.limit_linestyle,
    #     )
    #     ax.hlines(
    #         y=self.hydro_model.reservoirs[dam_name].rating_curve[0, -1],
    #         xmin=0,
    #         xmax=self.n_months - 1,
    #         linewidth=self.linewidth,
    #         color=self.colors[-1],
    #         linestyle=self.limit_linestyle,
    #     )
    #     ax.text(
    #         10,
    #         self.hydro_model.reservoirs[dam_name].rating_curve[0, 0],
    #         "Min level",
    #         bbox=dict(facecolor="gray", alpha=1),
    #         color="white",
    #         fontsize=8,
    #     )
    #     ax.text(
    #         10,
    #         self.hydro_model.reservoirs[dam_name].rating_curve[0, -1],
    #         "Max level",
    #         bbox=dict(facecolor="gray", alpha=1),
    #         color="white",
    #         fontsize=8,
    #     )
    #
    #     # plotting min and max observations
    #     ax.fill_between(
    #         range(self.n_months), maxi, mini, alpha=0.5, color=self.colors[7]
    #     )
    #     # plotting the average
    #     ax.plot(avg, color=self.colors[8], linewidth=5)
    #
    #     # setting up x and y axes, title
    #     ax.set_xticks(np.arange(self.n_months)) #, self.months, rotation=30)
    #     ax.set_ylabel(self.level_title)
    #
    #     plt.title(f"{dam_name} elevation summary per month")
    #     plt.show()

    def plot_condensed_figure(
            self, vector, y_name, title, label="",
            hor_line_positions=[], text_on_horiz=[],
            additional_vectors=[], vector_labels=[],
            colors=color_list
    ):
        vector = np.reshape(vector, (self.n_years, self.n_months))
        avg = np.mean(vector, 0)
        mini = np.min(vector, 0)
        maxi = np.max(vector, 0)

        fig, ax = plt.subplots(figsize=self.figsize)
        # setting up limits
        for i, h in enumerate(hor_line_positions):
            ax.hlines(
                y=h,
                xmin=0,
                xmax=self.n_months - 1,
                linewidth=self.linewidth,
                color=self.colors[-1],
                linestyle=self.limit_linestyle,
            )
            if text_on_horiz[i] is not None:
                ax.text(
                    10,
                    h,
                    text_on_horiz[i],
                    bbox=dict(facecolor="gray", alpha=1),
                    color="white",
                    fontsize=8,
                )
                
        for i, additional_vector in enumerate(additional_vectors):
            additional_vector = np.reshape(additional_vector, (self.n_years, self.n_months))
            additional_vector_avg = np.mean(additional_vector, 0)
            ax.plot(additional_vector_avg, color=colors[i], linewidth=5, label=vector_labels[i])
                
        # plotting min and max observations
        ax.fill_between(
            range(self.n_months), maxi, mini, alpha=0.5, color=self.colors[7]
        )
        # plotting the average
        ax.plot(avg, color=self.colors[8], linewidth=5, label=label)

        # setting up x and y axes, title
        ax.set_xticks(np.arange(self.n_months))  # , self.months, rotation=30)
        ax.set_xticklabels(self.months)
        ax.set_ylabel(y_name)
        ax.legend(loc='right')
        
        plt.title(title)
        plt.show()

    def plot_separated_condensed_figure(
            self, vector, y_name, title,
            hor_line_positions=[], text_on_horiz=[],
            separate_by=2
    ):
        new_vector = []
        avg = []
        mini = []
        maxi = []
        n_years_per_graph = int(self.n_years/separate_by)

        for i in range(2):
            new_vector.append(np.reshape(
                vector[120 * i: 120 * (i+1)],
                (n_years_per_graph, int(self.n_months))
            ))
            avg.append(np.mean(new_vector[i], 0))
            mini.append(np.min(new_vector[i], 0))
            maxi.append(np.max(new_vector[i], 0))

        fig, (ax1, ax2) = plt.subplots(
            ncols=2, sharex=True, sharey=True, figsize=self.landscape_figsize
        )
        # setting up limits
        for j, ax in enumerate([ax1, ax2]):
            for i, h in enumerate(hor_line_positions):
                ax.hlines(
                    y=h,
                    xmin=0,
                    xmax=self.n_months - 1,
                    linewidth=self.linewidth,
                    color=self.colors[-1],
                    linestyle=self.limit_linestyle,
                )

                if text_on_horiz[i] is not None:
                    ax.text(
                        10,
                        h,
                        text_on_horiz[i],
                        bbox=dict(facecolor="gray", alpha=1),
                        color="white",
                        fontsize=8,
                    )

            # plotting min and max observations
            ax.fill_between(
                range(self.n_months), maxi[j], mini[j], alpha=0.5, color=self.colors[7]
            )
            # plotting the average
            ax.plot(avg[j], color=self.colors[8], linewidth=5)

            # setting up x and y axes, title
            ax.set_xticks(np.arange(self.n_months)) #rotation = 30
            ax.set_xticklabels(self.months)
            ax.set_ylabel(y_name)
            ax.set_title(f"{title} for years {2022+j*10}-{2022+j*10+9}")

        #plt.title(title)
        fig.tight_layout()
        plt.show()

    def plot_condensed_demand(self, irr_district, policy_name):
        self.plot_condensed_figure(
            self.hydro_model.irr_districts[irr_district].demand,
            "Demand [m3/sec]",
            f"Average demand in {irr_district} with {policy_name} policy",
        )
        
        
    def plot_condensed_inflow(self, dam_name, policy_name):
        self.plot_condensed_figure(
            self.hydro_model.reservoirs[dam_name].inflow_vector,
            "Inflow [m3/sec]",
            f"Average inflows to {dam_name} with {policy_name} policy",
        )

    def plot_condensed_inflow_separated(self, dam_name, policy_name):
        self.plot_separated_condensed_figure(
            self.hydro_model.reservoirs[dam_name].inflow_vector,
            "Inflow [m3/sec]",
            f"Average inflows to {dam_name} with {policy_name} policy",
        )

    def plot_condensed_release(self, dam_name, policy_name,):
        hor_line_positions = [
            self.hydro_model.reservoirs[dam_name].hydropower_plants[0].max_turbine_flow
        ]
        text_on_horiz = ["Turbined Release"]
        self.plot_condensed_figure(
            self.hydro_model.reservoirs[dam_name].release_vector,
            "Release [m3/sec]",
            f"Average release from {dam_name} with {policy_name} policy",
            hor_line_positions=hor_line_positions,
            text_on_horiz=text_on_horiz
        )

    def plot_condensed_release_versus_inflow(self, dam_name, policy_name):
        hor_line_positions = [
            self.hydro_model.reservoirs[dam_name].hydropower_plants[0].max_turbine_flow
        ]
        text_on_horiz = ["Max Turbine Release"]
        
        additional_vectors = [
            self.hydro_model.reservoirs[dam_name].inflow_vector
        ]
        vector_labels = ["Blue Nile Inflow"]
        
        self.plot_condensed_figure(
            self.hydro_model.reservoirs[dam_name].release_vector,
            "Flow [m3/sec]",
            f"Inflow to versus Release from {dam_name}",
            label=f"{dam_name} Release",
            hor_line_positions=hor_line_positions,
            text_on_horiz=text_on_horiz,
            additional_vectors=additional_vectors,
            vector_labels=vector_labels
        )
        
        
    def plot_condensed_release_separated(self, dam_name, policy_name):
        hor_line_positions = [
            self.hydro_model.reservoirs[dam_name].hydropower_plants[0].max_turbine_flow
        ]
        text_on_horiz = ["Turbined Release"]
        self.plot_separated_condensed_figure(
            self.hydro_model.reservoirs[dam_name].release_vector,
            "Release [m3/sec]",
            f"Average release from {dam_name} with {policy_name} policy",
            hor_line_positions,
            text_on_horiz
        )

    def plot_condensed_level(self, dam_name, policy_name,):
        hor_line_positions = [
            self.hydro_model.reservoirs[dam_name].rating_curve[0, 0],
            self.hydro_model.reservoirs[dam_name].hydropower_plants[0].head_start_level,
            self.hydro_model.reservoirs[dam_name].rating_curve[0, -1]
        ]
        text_on_horiz = ["Minimum Level", "Turbine Level", "Maximum Level"]
        self.plot_condensed_figure(
            self.hydro_model.reservoirs[dam_name].level_vector,
            "Level [masl]",
            f"Average level of {dam_name} under {policy_name} policy",
            hor_line_positions=hor_line_positions,
            text_on_horiz=text_on_horiz
        )

    def plot_condensed_level_separated(self, dam_name, policy_name):
        hor_line_positions = [
            self.hydro_model.reservoirs[dam_name].rating_curve[0, 0],
            self.hydro_model.reservoirs[dam_name].hydropower_plants[0].head_start_level,
            self.hydro_model.reservoirs[dam_name].rating_curve[0, -1]
        ]
        text_on_horiz = ["Minimum Level", "Turbine Level", "Maximum Level"]
        self.plot_separated_condensed_figure(
            self.hydro_model.reservoirs[dam_name].level_vector,
            "Level [masl]",
            f"Average level of {dam_name} under {policy_name} policy",
            hor_line_positions,
            text_on_horiz
        )
