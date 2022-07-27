# Model class

# Importing libraries for functionality
import numpy as np
import pandas as pd

# Importing classes to generate the model
from .model_classes import Reservoir, Catchment, IrrigationDistrict, HydropowerPlant
from .smash import Policy


# import sys
# sys.path.append("..")
# from experimentation.data_generation import generate_input_data


class ModelNile:
    """
    Model class consists of three major functions. First, static
    components such as reservoirs, catchments, policy objects are
    created within the constructor. Evaluate function serves as the
    behaviour generating machine which outputs KPIs of the model.
    Evaluate does so by means of calling the simulate function which
    handles the state transformation via mass-balance equation
    calculations iteratively.
    """

    def __init__(self):
        """
        Creating the static objects of the model including the
        reservoirs, catchments, irrigation districts and policy
        objects along with their parameters. Also, reading both the
        model run configuration from settings, input data
        as well as policy function hyper-parameters.
        """

        self.read_settings_file("../settings/settings_file_Nile.xlsx")

        # Generating catchment and irrigation district objects
        self.catchments = dict()
        for name in self.catchment_names:
            new_catchment = Catchment(name)
            self.catchments[name] = new_catchment

        self.irr_districts = dict()
        for name in self.irr_district_names:
            new_irr_district = IrrigationDistrict(name)
            self.irr_districts[name] = new_irr_district

        # Generating reservoirs of the model. This includes also the generation
        # of hydropower plants when it exists in a reservoir
        self.reservoirs = dict()
        for name in self.reservoir_names:
            new_reservoir = Reservoir(name)

            new_plant = HydropowerPlant(new_reservoir)
            new_reservoir.hydropower_plants.append(new_plant)

            # Set initial storage values (based on excel settings)
            initial_storage = float(
                self.reservoir_parameters.loc[name, "Initial Storage(m3)"]
            )
            new_reservoir.storage_vector = np.append(
                new_reservoir.storage_vector, initial_storage
            )

            # Set hydropower production parameters (based on excel settings)
            variable_names_raw = self.reservoir_parameters.columns[-4:].values.tolist()

            for i, plant in enumerate(new_reservoir.hydropower_plants):
                for variable in variable_names_raw:
                    setattr(
                        plant,
                        variable.replace(" ", "_").lower(),
                        eval(self.reservoir_parameters.loc[name, variable])[i],
                    )

            self.reservoirs[name] = new_reservoir

        # Delete dataframe from memory after initialization
        del self.reservoir_parameters

        # Below the policy object (from the SMASH library) is generated
        self.overarching_policy = Policy()

        # Parameter values for the policy are inputted on the Excel settings
        # file. Each policy in the self.policies list is a dictionary with
        # variable name as the key and value as the value.
        for policy in self.policies:
            self.overarching_policy.add_policy_function(**policy)

        # As the policies are initialised, we can get rid of this list of
        # dictionaries to save memory space
        del self.policies

    def __call__(self, *args, **kwargs):
        lever_count = self.overarching_policy.get_total_parameter_count()
        input_parameters = [kwargs["v" + str(i)] for i in range(lever_count)]
        # uncertainty_parameters = {
        #     "yearly_demand_growth_rate": kwargs["yearly_demand_growth_rate"],
        #     "blue_nile_mean_coef": kwargs["blue_nile_mean_coef"],
        #     "white_nile_mean_coef": kwargs["white_nile_mean_coef"],
        #     "atbara_mean_coef": kwargs["atbara_mean_coef"],
        #     "blue_nile_dev_coef": kwargs["blue_nile_dev_coef"],
        #     "white_nile_dev_coef": kwargs["white_nile_dev_coef"],
        #     "atbara_dev_coef": kwargs["atbara_dev_coef"]
        # }
        (
            egypt_irr,
            egypt_90,
            egypt_low_had,
            sudan_irr,
            sudan_90,
            ethiopia_hydro,
        ) = self.evaluate(
            np.array(input_parameters)
        )  # , uncertainty_parameters
        return egypt_irr, egypt_90, egypt_low_had, sudan_irr, sudan_90, ethiopia_hydro

    def evaluate(self, parameter_vector):  # , uncertainty_dict
        """Evaluate the KPI values based on the given input
        data and policy parameter configuration.

        Parameters
        ----------
        self : ModelZambezi object
        parameter_vector : np.array
            Parameter values for the reservoir control policy
            object (NN, RBF etc.)

        Returns
        -------
        objective_values : list
            List of calculated objective values
        """

        self.reset_parameters()
        # self = generate_input_data(self, **uncertainty_dict)
        self.overarching_policy.assign_free_parameters(parameter_vector)
        self.simulate()

        bcm_def_egypt = [
            month * 3600 * 24 * self.nu_of_days_per_month[i % 12] * 1e-9
            for i, month in enumerate(self.irr_districts["Egypt"].deficit)
        ]

        egypt_agg_def = np.sum(bcm_def_egypt) / 20
        egypt_90_perc_worst = np.percentile(
            bcm_def_egypt, 90, interpolation="closest_observation"
        )
        egypt_freq_low_HAD = np.sum(self.reservoirs["HAD"].level_vector < 159) / len(
            self.reservoirs["HAD"].level_vector
        )

        sudan_irr_districts = [
            value for key, value in self.irr_districts.items() if key not in {"Egypt"}
        ]
        sudan_agg_def_vector = np.repeat(0.0, self.simulation_horizon)
        for district in sudan_irr_districts:
            sudan_agg_def_vector += district.deficit
        bcm_def_sudan = [
            month * 3600 * 24 * self.nu_of_days_per_month[i % 12] * 1e-9
            for i, month in enumerate(sudan_agg_def_vector)
        ]
        sudan_agg_def = np.sum(bcm_def_sudan) / 20
        sudan_90_perc_worst = np.percentile(
            bcm_def_sudan, 90, interpolation="closest_observation"
        )

        ethiopia_agg_hydro = (
            np.sum(self.reservoirs["GERD"].actual_hydropower_production)
        ) / (20 * 1e6)

        return (
            egypt_agg_def,
            egypt_90_perc_worst,
            egypt_freq_low_HAD,
            sudan_agg_def,
            sudan_90_perc_worst,
            ethiopia_agg_hydro,
        )

    def simulate(self):
        """Mathematical simulation over the specified simulation
        duration within a main for loop based on the mass-balance
        equations

        Parameters
        ----------
        self : ModelZambezi object
        """
        # Initial value for the total inflow (to be used in policy)
        total_monthly_inflow = self.inflowTOT00

        # To handle delay, I need to keep Taminiat leftovers in a list of two
        Taminiat_leftover = [0.0, 0.0]

        for t in np.arange(self.simulation_horizon):

            moy = (self.init_month + t - 1) % 12 + 1  # Current month
            nu_of_days = self.nu_of_days_per_month[moy - 1]

            # add the inputs for the function approximator (NN, RBF)
            # black-box policy. Watch out for verification if the below
            # sequence of reservoirs is the same as previous version

            storages = [
                reservoir.storage_vector[t] for reservoir in self.reservoirs.values()
            ]
            # In addition to storages, month of the year and total inflow
            # values are used by the policy function:
            input = storages + [moy, total_monthly_inflow]

            uu = self.overarching_policy.functions["release"].get_output_norm(
                np.array(input)
            )  # Policy function is called here!

            decision_dict = {
                reservoir.name: uu[index]
                for index, reservoir in enumerate(self.reservoirs.values())
            }

            # Integration of flows to storages
            self.reservoirs["GERD"].integration(
                nu_of_days,
                decision_dict["GERD"],
                self.catchments["BlueNile"].inflow[t],
                moy,
                self.integration_interval,
            )

            self.reservoirs["Roseires"].integration(
                nu_of_days,
                decision_dict["Roseires"],
                self.catchments["GERDToRoseires"].inflow[t]
                + self.reservoirs["GERD"].release_vector[-1],
                moy,
                self.integration_interval,
            )

            USSennar_input = (
                self.reservoirs["Roseires"].release_vector[-1]
                + self.catchments["RoseiresToAbuNaama"].inflow[t]
            )

            self.irr_districts["USSennar"].received_flow_raw = np.append(
                self.irr_districts["USSennar"].received_flow_raw, USSennar_input
            )

            self.irr_districts["USSennar"].received_flow = np.append(
                self.irr_districts["USSennar"].received_flow,
                min(USSennar_input, self.irr_districts["USSennar"].demand[t]),
            )

            USSennar_leftover = max(
                0, USSennar_input - self.irr_districts["USSennar"].received_flow[-1]
            )

            self.reservoirs["Sennar"].integration(
                nu_of_days,
                decision_dict["Sennar"],
                USSennar_leftover + self.catchments["SukiToSennar"].inflow[t],
                moy,
                self.integration_interval,
            )

            Gezira_input = self.reservoirs["Sennar"].release_vector[-1]

            self.irr_districts["Gezira"].received_flow_raw = np.append(
                self.irr_districts["Gezira"].received_flow_raw, Gezira_input
            )

            self.irr_districts["Gezira"].received_flow = np.append(
                self.irr_districts["Gezira"].received_flow,
                min(self.irr_districts["Gezira"].demand[t], Gezira_input),
            )

            Gezira_leftover = max(
                0, Gezira_input - self.irr_districts["Gezira"].received_flow[-1]
            )

            DSSennar_input = (
                Gezira_leftover
                + self.catchments["Dinder"].inflow[t]
                + self.catchments["Rahad"].inflow[t]
            )

            self.irr_districts["DSSennar"].received_flow_raw = np.append(
                self.irr_districts["DSSennar"].received_flow_raw, DSSennar_input
            )

            self.irr_districts["DSSennar"].received_flow = np.append(
                self.irr_districts["DSSennar"].received_flow,
                min(DSSennar_input, self.irr_districts["USSennar"].demand[t]),
            )

            DSSennar_leftover = max(
                0, DSSennar_input - self.irr_districts["DSSennar"].received_flow[-1]
            )

            Taminiat_input = DSSennar_leftover + self.catchments["WhiteNile"].inflow[t]

            self.irr_districts["Taminiat"].received_flow_raw = np.append(
                self.irr_districts["Taminiat"].received_flow_raw, Taminiat_input
            )

            self.irr_districts["Taminiat"].received_flow = np.append(
                self.irr_districts["Taminiat"].received_flow,
                min(Taminiat_input, self.irr_districts["Taminiat"].demand[t]),
            )

            Taminiat_leftover.append(
                max(
                    0, Taminiat_input - self.irr_districts["Taminiat"].received_flow[-1]
                )
            )
            del Taminiat_leftover[0]

            # Delayed reach of water to Hassanab:
            if t == 0:
                Hassanab_input = 934.2  # Last 5 years from GRDC Dongola data set
            else:
                Hassanab_input = (
                    Taminiat_leftover[0] + self.catchments["Atbara"].inflow[t - 1]
                )

            self.irr_districts["Hassanab"].received_flow_raw = np.append(
                self.irr_districts["Hassanab"].received_flow_raw, Hassanab_input
            )

            self.irr_districts["Hassanab"].received_flow = np.append(
                self.irr_districts["Hassanab"].received_flow,
                min(Hassanab_input, self.irr_districts["Hassanab"].demand[t]),
            )

            Hassanab_leftover = max(
                0, Hassanab_input - self.irr_districts["Hassanab"].received_flow[-1]
            )

            self.reservoirs["HAD"].integration(
                nu_of_days,
                decision_dict["HAD"],
                Hassanab_leftover,
                moy,
                self.integration_interval,
            )

            self.irr_districts["Egypt"].received_flow_raw = np.append(
                self.irr_districts["Egypt"].received_flow_raw,
                self.reservoirs["HAD"].release_vector[-1],
            )

            self.irr_districts["Egypt"].received_flow = np.append(
                self.irr_districts["Egypt"].received_flow,
                min(
                    self.reservoirs["HAD"].release_vector[-1],
                    self.irr_districts["Egypt"].demand[t],
                ),
            )

            total_monthly_inflow = sum([x.inflow[t] for x in self.catchments.values()])

            # Calculation of objectives:

            # Irrigation demand deficits

            for district in self.irr_districts.values():
                district.deficit = np.append(
                    district.deficit,
                    self.deficit_from_target(
                        district.received_flow[-1], district.demand[t]
                    ),
                )

            # Hydropower objectives

            for reservoir in self.reservoirs.values():
                hydropower_production = 0
                for plant in reservoir.hydropower_plants:
                    production = plant.calculate_hydropower_production(
                        reservoir.release_vector[-1],
                        reservoir.level_vector[-1],
                        nu_of_days,
                    )
                    hydropower_production += production

                reservoir.actual_hydropower_production = np.append(
                    reservoir.actual_hydropower_production, hydropower_production
                )

            if t == (self.GERD_filling_time * 12):
                self.reservoirs["GERD"].filling_schedule = None

    @staticmethod
    def deficit_from_target(realisation, target):
        """
        Calculates the deficit given the realisation of an
        objective and the target
        """
        return max(0, target - realisation)

    @staticmethod
    def squared_deficit_from_target(realisation, target):
        """
        Calculates the square of a deficit given the realisation of an
        objective and the target
        """
        return pow(max(0, target - realisation), 2)

    @staticmethod
    def squared_deficit_normalised(sq_deficit, target):
        """
        Scales down a squared deficit with respect to the square of the target
        """
        if target == 0:
            return 0
        else:
            return sq_deficit / pow(target, 2)

    def set_GERD_filling_schedule(self, duration):
        target_storage = 50e9
        difference = target_storage - self.reservoirs["GERD"].storage_vector[0]
        secondly_diff = difference / (duration * 365 * 24 * 3600)
        weights = self.catchments["BlueNile"].inflow[:12]
        self.reservoirs["GERD"].filling_schedule = (
            weights * 12 * secondly_diff
        ) / weights.sum()

    def reset_parameters(self):

        for reservoir in self.reservoirs.values():
            # Leaving only the initial value in the storages
            reservoir.storage_vector = reservoir.storage_vector[:1]
            # Resetting all other vectors:
            attributes = [
                "level_vector",
                "release_vector",
                "level_vector",
                "actual_hydropower_production",
                "hydropower_deficit",
            ]
            for var in attributes:
                setattr(reservoir, var, np.empty(0))

        for irr_district in self.irr_districts.values():
            attributes = ["received_flow", "deficit"]
            for var in attributes:
                setattr(irr_district, var, np.empty(0))

    def read_settings_file(self, filepath):

        model_parameters = pd.read_excel(filepath, sheet_name="ModelParameters")
        for _, row in model_parameters.iterrows():
            name = row["in Python"]
            if row["Data Type"] == "str":
                value = row["Value"]
            else:
                value = eval(str(row["Value"]))
            if row["Data Type"] == "np.array":
                value = np.array(value)
            setattr(self, name, value)

        self.reservoir_parameters = pd.read_excel(filepath, sheet_name="Reservoirs")

        self.reservoir_parameters.set_index("Reservoir Name", inplace=True)

        self.policies = list()
        full_df = pd.read_excel(filepath, sheet_name="PolicyParameters")
        splitpoints = list(full_df.loc[full_df["Parameter Name"] == "Name"].index)
        for i in range(len(splitpoints)):
            try:
                one_policy = full_df.iloc[splitpoints[i] : splitpoints[i + 1], :]
            except IndexError:
                one_policy = full_df.iloc[splitpoints[i] :, :]
            input_dict = dict()

            for _, row in one_policy.iterrows():
                key = row["in Python"]
                if row["Data Type"] != "str":
                    value = eval(str(row["Value"]))
                else:
                    value = row["Value"]
                if row["Data Type"] == "np.array":
                    value = np.array(value)
                input_dict[key] = value

            self.policies.append(input_dict)
