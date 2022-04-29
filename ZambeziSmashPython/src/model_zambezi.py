# Model class

# Importing libraries for functionality
import numpy as np
import pandas as pd

# Importing classes to generate the model
from reservoir import Reservoir
from catchment import Catchment
from irrigation_district import IrrigationDistrict
from hydropower_plant import HydropowerPlant
from smash import Policy

class ModelZambezi:
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

        self.read_settings_file("../settings/settings_file.xlsx")
    
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
            if name == "kariba":
                new_plant = HydropowerPlant(new_reservoir, "North", 0.488)
                new_reservoir.hydropower_plants.append(new_plant)
                new_plant = HydropowerPlant(new_reservoir, "South", 0.512)
                new_reservoir.hydropower_plants.append(new_plant)
            else:
                new_plant = HydropowerPlant(new_reservoir)
                new_reservoir.hydropower_plants.append(new_plant)
            
            # Set initial storage values (based on excel settings)
            initial_storage = float(self.reservoir_parameters.loc[name, \
                "Initial Storage(m3)"])
            new_reservoir.storage_vector = np.append\
                (new_reservoir.storage_vector, initial_storage)

            # Set hydropower production parameters (based on excel settings)
            variable_names_raw = self.reservoir_parameters.columns[-3:].\
                values.tolist()
        
            for i, plant in enumerate(new_reservoir.hydropower_plants):
                for variable in variable_names_raw:
                    setattr(plant, variable.replace(" ", "_").lower(),
                    eval(self.reservoir_parameters.loc[name,variable])[i])
            
            self.reservoirs[name] = new_reservoir
        
        # Delete dataframe from memory after initialization
        del self.reservoir_parameters 
        
        # Defining the objectives of the problem:

        # Assign hydropower production targets to reservoirs
        data_directory = "../data/"
        acronyms = {"itezhitezhi":"ITT", "kafuegorgeupper":"KGU",
        "kafuegorgelower":"KGL", "kariba":"KA", "cahorabassa":"CB"}
        for reservoir in self.reservoirs.values():
            setattr(reservoir,"target_hydropower_production",
            np.loadtxt(f"{data_directory}{acronyms[reservoir.name]}prod.txt"))

        # Load Minimum Environmental Flow requirement upstream of
        # Victoria Falls [m^3/sec]
        self.MEF_VictoriaFalls = np.loadtxt(f"{data_directory}MEF_VictoriaFalls.txt")

        # Load Minimum Environmental Flow requirement in the Zambezi
        # Delta for the months of February and March
        self.qDelta = np.loadtxt(f"{data_directory}MEF_Delta.txt") # [m^3/sec]

        # Initialize the vectors to hold objective deficits:
        self.total_env_deficit = np.empty(0)
        self.total_irr_deficit = np.empty(0)
        self.total_hydro_deficit = np.empty(0)

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

    def evaluate(self, parameter_vector):
        """ Evaluate the KPI values based on the given input
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
        self.overarching_policy.assign_free_parameters(parameter_vector)
        self.simulate()
        
        objective_values = list()
        objective_values.append(np.mean(self.total_env_deficit))
        objective_values.append(np.mean(self.total_irr_deficit))
        objective_values.append(np.mean(self.total_hydro_deficit))

        return objective_values

    def simulate(self):
        """ Mathematical simulation over the specified simulation
        duration within a main for loop based on the mass-balance
        equations

        Parameters
        ----------
        self : ModelZambezi object
        """
        # Initial value for the total inflow (to be used in policy)
        total_monthly_inflow = self.inflowTOT00

        for t in np.arange(self.simulation_horizon):
            
            moy = (self.init_month+t-1)%12+1 # Current month
            nu_of_days = self.nu_of_days_per_month[moy-1]

            # add the inputs for the function approximator (NN, RBF)
            # black-box policy. Watch out for validation if the below
            # sequence of reservoirs is the same as previous version

            storages = [reservoir.storage_vector[t] for reservoir \
                in self.reservoirs.values()]
            # In addition to storages, month of the year and total inflow
            # values are used by the policy function:
            input = storages + [moy, total_monthly_inflow]

            uu = self.overarching_policy.functions["release"].\
                get_output_norm(np.array(input)) # Policy function is called here!
            
            decision_dict = {reservoir.name: uu[index] \
                for index, reservoir in enumerate(self.reservoirs.values())}

            # Integration of flows to storages
            self.reservoirs["itezhitezhi"].integration(
                nu_of_days, decision_dict["itezhitezhi"],
                self.catchments["Itt"].inflow[t], moy, self.integration_interval)

            # Itezhitezhi delay handling:
            if t in [0, 1]:
                itt_release = self.ithezhitezhi_delay[t-2]
            else:
                itt_release = self.reservoirs["itezhitezhi"].release_vector[-3]

            # Keep the release simple for now, by saying the minimum of
            # demand and possible inflow        
            self.irr_districts["4"].received_flow = np.append(
                self.irr_districts["4"].received_flow,
                min(self.catchments["KafueFlats"].inflow[t] + itt_release,
                self.irr_districts["4"].demand[moy-1]))

            # Integration of flows to storages
            self.reservoirs["kafuegorgeupper"].integration(
                nu_of_days, decision_dict["kafuegorgeupper"],
                self.catchments["KafueFlats"].inflow[t] + itt_release - \
                self.irr_districts["4"].received_flow[-1],moy, self.integration_interval)

            self.reservoirs["kafuegorgelower"].integration(
                nu_of_days, decision_dict["kafuegorgelower"],
                self.reservoirs["kafuegorgeupper"].release_vector[-1], moy,
                self.integration_interval)

            self.irr_districts["2"].received_flow = np.append(
                self.irr_districts["2"].received_flow,
                min(self.catchments["Bg"].inflow[t] + \
                    self.catchments["Cuando"].inflow[t] +\
                    self.catchments["Ka"].inflow[t],
                self.irr_districts["2"].demand[moy-1]))

            self.reservoirs["kariba"].integration(
                nu_of_days, decision_dict["kariba"],
                self.catchments["Bg"].inflow[t] + \
                self.catchments["Cuando"].inflow[t] +\
                self.catchments["Ka"].inflow[t] - \
                self.irr_districts["2"].received_flow[-1], moy,
                self.integration_interval)

            self.irr_districts["3"].received_flow = np.append(
                self.irr_districts["3"].received_flow,
                min(self.reservoirs["kariba"].release_vector[-1],
                self.irr_districts["3"].demand[moy-1]))

            self.irr_districts["5"].received_flow = np.append(
                self.irr_districts["5"].received_flow,
                min(self.reservoirs["kafuegorgelower"].release_vector[-1],
                self.irr_districts["5"].demand[moy-1]))

            self.irr_districts["6"].received_flow = np.append(
                self.irr_districts["6"].received_flow,
                min(self.reservoirs["kariba"].release_vector[-1] - \
                    self.irr_districts["3"].received_flow[-1] + \
                    self.reservoirs["kafuegorgelower"].release_vector[-1] - \
                    self.irr_districts["5"].received_flow[-1],
                self.irr_districts["6"].demand[moy-1]))

            self.reservoirs["cahorabassa"].integration(
                nu_of_days, decision_dict["cahorabassa"],
                self.catchments["Cb"].inflow[t] + \
                self.reservoirs["kafuegorgelower"].release_vector[-1] +\
                self.reservoirs["kariba"].release_vector[-1] - \
                self.irr_districts["3"].received_flow[-1] - \
                self.irr_districts["5"].received_flow[-1] - \
                self.irr_districts["6"].received_flow[-1], moy,
                self.integration_interval)

            self.irr_districts["7"].received_flow = np.append(
                self.irr_districts["7"].received_flow,
                min(self.reservoirs["cahorabassa"].release_vector[-1],
                self.irr_districts["7"].demand[moy-1]))

            self.irr_districts["8"].received_flow = np.append(
                self.irr_districts["8"].received_flow,
                min(self.reservoirs["cahorabassa"].release_vector[-1] - \
                    self.irr_districts["7"].received_flow[-1],
                self.irr_districts["8"].demand[moy-1]))

            self.irr_districts["9"].received_flow = np.append(
                self.irr_districts["9"].received_flow,
                min(self.reservoirs["cahorabassa"].release_vector[-1] - \
                    self.irr_districts["7"].received_flow[-1] - \
                    self.irr_districts["8"].received_flow[-1] + \
                    self.catchments["Shire"].inflow[t],
                self.irr_districts["9"].demand[moy-1]))

            total_monthly_inflow = sum([x.inflow[t] for x in self.catchments.values()])

            # Calculation of objectives:

            # Irrigation deficits
            total_deficit = 0
            for district in self.irr_districts.values():
                district.squared_deficit = np.append(district.squared_deficit,
                    self.squared_deficit_from_target(district.received_flow[-1],
                        district.demand[moy-1]))
                
                district.normalised_deficit = np.append(district.normalised_deficit,
                    self.squared_deficit_normalised(district.squared_deficit[-1],
                        district.demand[moy-1]))
                
                total_deficit += district.squared_deficit[-1]

            self.total_irr_deficit = np.append(self.total_irr_deficit, total_deficit)

            # Hydropower objectives
            total_deficit = 0
            for reservoir in self.reservoirs.values():
                hydropower_production = 0
                for plant in reservoir.hydropower_plants:
                    production = plant.calculate_hydropower_production(
                        reservoir.release_vector[-1], reservoir.level_vector[-1],
                        nu_of_days)
                    hydropower_production += production

                reservoir.actual_hydropower_production = np.append(
                    reservoir.actual_hydropower_production, hydropower_production
                )
                reservoir.hydropower_deficit = np.append(
                    reservoir.hydropower_deficit, max(0,
                    reservoir.target_hydropower_production[-1] - hydropower_production)
                )
                
                total_deficit += reservoir.hydropower_deficit[-1]

            self.total_hydro_deficit = np.append(self.total_hydro_deficit, total_deficit)

            # Environmental flow objective
            flow_delta = self.reservoirs["cahorabassa"].release_vector[-1] - \
                    self.irr_districts["7"].received_flow[-1] - \
                    self.irr_districts["8"].received_flow[-1] + \
                    self.catchments["Shire"].inflow[t] - \
                    self.irr_districts["9"].received_flow[-1]
            self.total_env_deficit = np.append(self.total_env_deficit,
                self.squared_deficit_from_target(flow_delta, self.qDelta[moy-1]))
            
    @staticmethod
    def squared_deficit_from_target(realisation, target):
        """
        Calculates the square of a deficit given the realisation of an
        objective and the target
        """
        return pow(max(0, target-realisation),2)

    @staticmethod
    def squared_deficit_normalised(sq_deficit, target):
        """
        Scales down a squared deficit with respect to the square of the target
        """
        if target == 0: return 0
        else: return sq_deficit/pow(target, 2)

    def reset_parameters(self):

        for reservoir in self.reservoirs.values():
            # Leaving only the initial value in the storages
            reservoir.storage_vector = reservoir.storage_vector[:1]
            # Resetting all other vectors:
            attributes = ["level_vector", "release_vector", "level_vector",
                "actual_hydropower_production", "hydropower_deficit"]
            for var in attributes:
                setattr(reservoir, var, np.empty(0))

        for irr_district in self.irr_districts.values():
            attributes = ["received_flow", "squared_deficit",
                "normalised_deficit"]
            for var in attributes:
                setattr(irr_district, var, np.empty(0))

        # Reset the vectors that hold objective deficits:
        self.total_env_deficit = np.empty(0)
        self.total_irr_deficit = np.empty(0)
        self.total_hydro_deficit = np.empty(0)
        
    def read_settings_file(self, filepath):

        model_parameters = pd.read_excel(filepath, sheet_name="ModelParameters")
        for _, row in model_parameters.iterrows():
            name = row["in Python"]
            if row["Data Type"] == "str": value = row["Value"]
            else: value = eval(str(row["Value"]))
            if row["Data Type"] == "np.array": value = np.array(value)
            setattr(self, name, value)

        self.reservoir_parameters = pd.read_excel(filepath,
            sheet_name="Reservoirs")
        self.reservoir_parameters["Reservoir Name"] = pd.Series(map(str.lower,
            self.reservoir_parameters["Reservoir Name"]))
        self.reservoir_parameters.set_index("Reservoir Name", inplace=True)

        self.policies = list()
        full_df = pd.read_excel(filepath, sheet_name="PolicyParameters")
        splitpoints = list(full_df.loc[full_df["Parameter Name"] == "Name"]\
            .index)
        for i in range(len(splitpoints)):
            try:
                one_policy = full_df.iloc[splitpoints[i]:splitpoints[i+1],:]
            except IndexError:
                one_policy = full_df.iloc[splitpoints[i]:,:]
            input_dict = dict()
            
            for _, row in one_policy.iterrows():
                key = row["in Python"]
                if row["Data Type"] != "str": value = eval(str(row["Value"]))
                else: value = row["Value"]
                if row["Data Type"] == "np.array": value = np.array(value)
                input_dict[key] = value

            self.policies.append(input_dict)