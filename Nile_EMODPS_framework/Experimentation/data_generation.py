import numpy as np
import pickle
import pandas as pd

def generate_input_data(nile_model, myseed = 123, wh_set = "Baseline",
                        sim_horizon = 20, yearly_demand_growth_rate=0.02, GERD_filling=5):
    # streamflow + demand
    data_directory = "../StochasticDataGenerationInputs/"

    # start with the demand
    for district in nile_model.irr_districts.values():
        one_year = np.loadtxt(f"{data_directory}IrrDemand{district.name}.txt")
        demand_vector = np.empty(0)
        loop_counter = sim_horizon
        while loop_counter > 0:
            demand_vector = np.append(demand_vector, one_year)
            one_year *= (1+yearly_demand_growth_rate)
            loop_counter -= 1
        district.demand = demand_vector
    
    # time for streamflow, start by getting the appropriate Wheeler (2018) set:
    wh_data = pickle.load( open( f"{data_directory}{wh_set}_wheeler.p", "rb" ) )
    np.random.seed(myseed)
    set_number = np.random.randint(1, 101)
    numbered_catchments = wh_data[set_number]
    numbered_catchments = numbered_catchments.iloc[49:(49+sim_horizon*12)]

    # Focus on the major inflows (3 gaging stations)
    atbara_dist = pd.read_csv(f"{data_directory}atbara_distribution.csv")
    mogren_dist = pd.read_csv(f"{data_directory}mogren_distribution.csv")
    bluenile_dist = pd.read_csv(f"{data_directory}blue_nile_series.csv")
    loop_counter = sim_horizon
    atbara = np.empty(0)
    mogren = np.empty(0)
    bluenile = np.empty(0)
    while loop_counter > 0:
        a = list()
        m = list()
        b = list()
        for i in range(12):
            a.append(max(0,np.random.normal(atbara_dist.loc[i,"mean"], atbara_dist.loc[i,"std"])))
            m.append(np.random.triangular(mogren_dist.loc[i,"MinQ"], mogren_dist.loc[i,"MeanQ"], mogren_dist.loc[i,"MaxQ"]))
            b.append(np.random.uniform(bluenile_dist.loc[i,"0"]*0.7, bluenile_dist.loc[i,"0"]*1.3))
        atbara = np.append(atbara, a)
        mogren = np.append(mogren, m)
        bluenile = np.append(bluenile, b)

        loop_counter -= 1
    inflow_dict = get_inflow_dict(numbered_catchments,atbara,mogren,bluenile)
    for catchment in nile_model.catchments.values():
        catchment.inflow = np.array(inflow_dict[catchment.name])

    nile_model.set_GERD_filling_schedule(GERD_filling)

    return nile_model

def get_inflow_dict(wh_df, atbara, mogren, bluenile):
    
    output = dict()
    output["Dinder"] = wh_df["340.Inflow"] + wh_df["635.Inflow"] + \
        wh_df["1308.Inflow"]
    output["Rahad"] = wh_df["243.Inflow"] + wh_df["519.Inflow"] + \
        wh_df["524.Inflow"]
    output["GERDToRoseires"] = wh_df["33.Inflow"] + wh_df["530.Inflow"] + \
        wh_df["1374.Inflow"]
    output["RoseiresToAbuNaama"] = wh_df["1309.Inflow"]
    output["SukiToSennar"] = wh_df["470.Inflow"]
    output["WhiteNile"] = wh_df["1364.Inflow"] + wh_df["1338.Inflow"] + \
        wh_df["1317.Inflow"] + wh_df["31.Inflow"] + mogren
    output["Atbara"] = atbara
    output["BlueNile"] = bluenile
    
    return output
