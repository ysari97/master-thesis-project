# ===========================================================================
# Name        : model_zambezi_OPT.py
# Author      : YasinS, adapted from JazminZ & ?
# Version     : 0.05
# Copyright   : Your copyright notice
# ===========================================================================

# Importing model classes:
from catchment import catchment, catchment_param
from lake import reservoir_param, lake
from utils import utils
from smash import Policy

import numpy as np
import pandas as pd

class model_zambezi:
    """
    Model class consists of three major functions. First, static components
    such as reservoirs, catchments, policy objects are created within the
    constructor. Evaluate function serves as the behaviour generating machine
    which outputs KPIs of the model. Evaluate does so by means of calling
    the simulate function which handles the state transformation via
    mass-balance equation calculations iteratively.
    """

    def __init__(self):
        """
        Creating the static objects of the model including the reservoirs,
        catchements and policy objects along with their parameters. Also,
        reading both the model run configuration from settings,
        input data (flows etc.) as well as policy function hyper-parameters.
        """

        # initialize parameter constructs for objects (policy and catchment)
        # (has to be done before file reading)
        
        # initialize parameter constructs for to be created policy objects   
        self.p_param = policy_parameters_construct()
        self.irr_param = irr_function_parameters()

        # initialize parameter constructs for to be created
        # Catchment parameter objects (stored in a dictionary):
        catchment_list = ["Itt","KafueFlats","Ka","Cb","Cuando","Shire","Bg"]
        self.catchment_param_dict = dict()
        
        for catchment_name in catchment_list:
            catch_param_name = catchment_name + "_catch_param"
            self.catchment_param_dict[catch_param_name] = catchment_param()

        # Reservoir parameter objects (stored seperately
        # to facilitate settings file reference):
        self.KGU_param = reservoir_param()
        self.ITT_param = reservoir_param()
        self.KA_param = reservoir_param()
        self.CB_param = reservoir_param()
        self.KGL_param = reservoir_param()

        # read the parameter values from either CSV or UI
        self.readFileSettings()

        # Catchment objects (stored in a dictionary)
        self.catchment_dict = dict()
        
        for catchment_name in catchment_list:
            catch_param_name = catchment_name + "_catch_param"
            variable_name = catchment_name + "Catchment"
            
            # Specific parameter construct is used in instantiation
            self.catchment_dict[variable_name] = catchment(self.catchment_param_dict[catch_param_name])
        
        # create reservoirs
        # KAFUE GORGE UPPER reservoir
        self.KafueGorgeUpper = lake("kafuegorgeupper") # creating a new object from corresponding lake class
        self.KafueGorgeUpper.setEvap(1) # evaporation data: 0 = no evaporation, 1 = load evaporation from file, 2 = activate function
        self.KGU_param.evap_rates.filename = "../data/evap_KG_KF.txt"
        self.KGU_param.evap_rates.row = self.T # setting # of rows as simulation period (12) for file reading
        self.KafueGorgeUpper.setEvapRates(self.KGU_param.evap_rates)
        self.KGU_param.lsv_rel.filename = "../data/lsv_rel_KafueGorgeUpper.txt" # [m m2 m3]
        self.KGU_param.lsv_rel.row = 3 
        self.KGU_param.lsv_rel.col = 22 
        self.KafueGorgeUpper.setLSV_Rel(self.KGU_param.lsv_rel)
        self.KGU_param.rating_curve.filename = "../data/min_max_release_KafueGorgeUpper.txt" #  [m m3/s m3/s] # 
        self.KGU_param.rating_curve.row = 3
        self.KGU_param.rating_curve.col = 18
        self.KafueGorgeUpper.setRatCurve(self.KGU_param.rating_curve)
        self.KGU_param.tailwater.filename = "../data/tailwater_rating_KafueGorgeUpper.txt" # [m3/s m] = [Discharge Tailwater Level]
        self.KGU_param.tailwater.row = 2
        self.KGU_param.tailwater.col = 7
        self.KafueGorgeUpper.setTailwater(self.KGU_param.tailwater)
        self.KGU_param.minEnvFlow.filename = "../data/MEF_KafueGorgeUpper.txt" # [m^3/sec]
        self.KGU_param.minEnvFlow.row = self.T
        self.KafueGorgeUpper.setMEF(self.KGU_param.minEnvFlow)
        self.KafueGorgeUpper.setInitCond(self.KGU_param.initCond)

        # ITEZHITEZHI reservoir
        self.Itezhitezhi = lake("itezhitezhi")
        self.Itezhitezhi.setEvap(1)
        self.ITT_param.evap_rates.filename = "../data/evap_ITT.txt"
        self.ITT_param.evap_rates.row = self.T
        self.Itezhitezhi.setEvapRates(self.ITT_param.evap_rates)
        self.ITT_param.lsv_rel.filename = "../data/lsv_rel_Itezhitezhi.txt"
        self.ITT_param.lsv_rel.row = 3 
        self.ITT_param.lsv_rel.col = 19 # 
        self.Itezhitezhi.setLSV_Rel(self.ITT_param.lsv_rel)
        self.ITT_param.rating_curve.filename = "../data/min_max_release_Itezhitezhi.txt"
        self.ITT_param.rating_curve.row = 3
        self.ITT_param.rating_curve.col = 43
        self.Itezhitezhi.setRatCurve(self.ITT_param.rating_curve)
        self.ITT_param.tailwater.filename = "../data/tailwater_rating_Itezhitezhi.txt"
        self.ITT_param.tailwater.row = 2
        self.ITT_param.tailwater.col = 10
        self.Itezhitezhi.setTailwater(self.ITT_param.tailwater)
        self.ITT_param.minEnvFlow.filename = "../data/MEF_Itezhitezhi.txt"
        self.ITT_param.minEnvFlow.row = self.T
        self.Itezhitezhi.setMEF(self.ITT_param.minEnvFlow)
        self.Itezhitezhi.setInitCond(self.ITT_param.initCond)

        # KARIBA reservoir
        self.Kariba = lake("kariba")
        self.Kariba.setEvap(1)
        self.KA_param.evap_rates.filename = "../data/evap_KA.txt"
        self.KA_param.evap_rates.row = self.T
        self.Kariba.setEvapRates(self.KA_param.evap_rates)
        self.KA_param.lsv_rel.filename = "../data/lsv_rel_Kariba.txt" 
        self.KA_param.lsv_rel.row = 3
        self.KA_param.lsv_rel.col = 16 
        self.Kariba.setLSV_Rel(self.KA_param.lsv_rel)
        self.KA_param.rating_curve.filename = "../data/min_max_release_Kariba.txt" 
        self.KA_param.rating_curve.row = 3
        self.KA_param.rating_curve.col = 11
        self.Kariba.setRatCurve(self.KA_param.rating_curve)
        self.KA_param.rule_curve.filename = "../data/rule_curve_Kariba.txt"
        self.KA_param.rule_curve.row = 3
        self.KA_param.rule_curve.col = 12
        self.Kariba.setRuleCurve(self.KA_param.rule_curve)
        self.KA_param.tailwater.filename = "../data/tailwater_rating_Kariba.txt" 
        self.KA_param.tailwater.row = 2
        self.KA_param.tailwater.col = 9
        self.Kariba.setTailwater(self.KA_param.tailwater)
        self.KA_param.minEnvFlow.filename = "../data/MEF_Kariba.txt" 
        self.KA_param.minEnvFlow.row = self.T
        self.Kariba.setMEF(self.KA_param.minEnvFlow)
        self.Kariba.setInitCond(self.KA_param.initCond)

        # CahoraBassa reservoir
        self.CahoraBassa = lake("cahorabassa")
        self.CahoraBassa.setEvap(1)
        self.CB_param.evap_rates.filename = "../data/evap_CB.txt"
        self.CB_param.evap_rates.row = self.T
        self.CahoraBassa.setEvapRates(self.CB_param.evap_rates)
        self.CB_param.lsv_rel.filename = "../data/lsv_rel_CahoraBassa.txt"
        self.CB_param.lsv_rel.row = 3
        self.CB_param.lsv_rel.col = 10 # 
        self.CahoraBassa.setLSV_Rel(self.CB_param.lsv_rel)
        self.CB_param.rating_curve.filename = "../data/min_max_release_CahoraBassa.txt"
        self.CB_param.rating_curve.row = 3
        self.CB_param.rating_curve.col = 10
        self.CahoraBassa.setRatCurve(self.CB_param.rating_curve)
        self.CB_param.rule_curve.filename = "../data/rule_curve_CahoraBassa.txt" # [time(month) month-end level(m) month-end storage(m3)] 
        self.CB_param.rule_curve.row = 3
        self.CB_param.rule_curve.col = 12
        self.CahoraBassa.setRuleCurve(self.CB_param.rule_curve)
        self.CB_param.tailwater.filename = "../data/tailwater_rating_CahoraBassa.txt" 
        self.CB_param.tailwater.row = 2
        self.CB_param.tailwater.col = 9
        self.CahoraBassa.setTailwater(self.CB_param.tailwater)
        self.CB_param.minEnvFlow.filename = "../data/MEF_CahoraBassa.txt"
        self.CB_param.minEnvFlow.row = self.T
        self.CahoraBassa.setMEF(self.CB_param.minEnvFlow)
        self.CahoraBassa.setInitCond(self.CB_param.initCond)

        # KAFUE GORGE Lower reservoir
        self.KafueGorgeLower = lake("kafuegorgelower")
        self.KafueGorgeLower.setEvap(1)
        self.KGL_param.evap_rates.filename = "../data/evap_KGL.txt"
        self.KGL_param.evap_rates.row = self.T
        self.KafueGorgeLower.setEvapRates(self.KGL_param.evap_rates)
        self.KGL_param.lsv_rel.filename = "../data/lsv_rel_KafueGorgeLower.txt" 
        self.KGL_param.lsv_rel.row = 3
        self.KGL_param.lsv_rel.col = 10 
        self.KafueGorgeLower.setLSV_Rel(self.KGL_param.lsv_rel)
        self.KGL_param.rating_curve_minmax.filename = "../data/min_max_release_KafueGorgeLower.txt" 
        self.KGL_param.rating_curve.row = 1
        self.KGL_param.rating_curve.col = 3
        self.KafueGorgeLower.setRatCurve_MinMax(self.KGL_param.rating_curve_minmax)
        self.KGL_param.tailwater.filename = "../data/tailwater_rating_KafueGorgeLower.txt"
        self.KGL_param.tailwater.row = 2
        self.KGL_param.tailwater.col = 8
        self.KafueGorgeLower.setTailwater(self.KGL_param.tailwater)
        self.KGL_param.minEnvFlow.filename = "../data/MEF_KafueGorgeLower.txt"
        self.KGL_param.minEnvFlow.row = self.T
        self.KafueGorgeLower.setMEF(self.KGL_param.minEnvFlow)
        self.KafueGorgeLower.setInitCond(self.KGL_param.initCond)

        # Below the policy objects (from the SMASH library) are generated
        # this model requires two policy functions (to be used in seperate
        # places in the simulate function) which are the "release" and
        # "irrigation" policies. While the former is meant to be a generic
        # approximator such as RBF and ANN (to be optimized) the latter
        # the latter has a simple structure specified in the
        # alternative_policy_structures script. Firstly, a Policy object is
        # instantiated which is meant to own all policy functions within a
        # model (see the documentation of SMASH). Then, two separate policies
        # are added on to the overarching_policy.
       
        self.overarching_policy = Policy()

        self.overarching_policy.add_policy_function(name="irrigation",
            type="user_specified", n_inputs=4, n_outputs=1,
            class_name="irrigation_policy", n_irr_districts=8)
        
        self.overarching_policy.functions["irrigation"].setMinInput(self.irr_param.mParam)
        self.overarching_policy.functions["irrigation"].setMaxInput(self.irr_param.MParam)

        self.overarching_policy.add_policy_function(name="release",
            type="ncRBF",n_inputs=self.p_param.policyInput, n_outputs=self.p_param.policyOutput,
            n_structures=self.p_param.policyStr)

        self.overarching_policy.functions["release"].setMaxInput(self.p_param.MIn)
        self.overarching_policy.functions["release"].setMaxOutput(self.p_param.MOut)
        self.overarching_policy.functions["release"].setMinInput(self.p_param.mIn)
        self.overarching_policy.functions["release"].setMinOutput(self.p_param.mOut)

        # Load irrigation demand vectors (stored in a dictionary)
        irr_naming_list = range(2, 10, 1)
        self.irr_demand_dict = dict()

        for id in irr_naming_list:
            variable_name = "irr_demand" + str(id)
            file_name = "../data/IrrDemand" + str(id) + ".txt"
            self.irr_demand_dict[variable_name] = utils.loadVector(file_name, self.T)


        self.irr_district_idx = utils.loadVector("../data/IrrDistrict_idx.txt", self.irr_param.num_irr) # index referring to the position
                                                                                                           # of the first parameter (hdg) of
                                                                                                           # each irrigation district in the
                                                                                                           # decision variables vector

        # Target hydropower production for each reservoir
        self.tp_Itt = utils.loadVector("../data/ITTprod.txt", self.T) # Itezhitezhi target production
        self.tp_Kgu = utils.loadVector("../data/KGUprod.txt", self.T) # Kafue Gorge Upper target production
        self.tp_Ka = utils.loadVector("../data/KAprod.txt", self.T) # Kariba target production
        self.tp_Cb = utils.loadVector("../data/CBprod.txt", self.T) # Cahora Bassa target production
        self.tp_Kgl = utils.loadVector("../data/KGLprod.txt", self.T) # Kafue Gorge Lower target production
        
        # Load Minimum Environmental Flow requirement upstream of Victoria Falls
        self.MEF_VictoriaFalls = utils.loadVector("../data/MEF_VictoriaFalls.txt", self.T) # [m^3/sec]

        # Load Minimum Environmental Flow requirement in the Zambezi Delta for the months of February and March
        self.qDelta = utils.loadVector("../data/MEF_Delta.txt", self.T) # [m^3/sec]

    def getNobj(self):
        return self.Nobj
    
    def getNvar(self):
        return self.Nvar

    def evaluate(self, var):
        """ Evaluate the KPI values based on the given input
        data and policy parameter configuration.

        Parameters
        ----------
        self : model_zambezi object
        var : np.array
            Parameter values for the reservoir control policy
            object (NN, RBF etc.)

        Returns
        -------
        Either obj or None (just writing to a file) depending on
        the mode (simulation or optimization)
        """

        obj = np.empty(0)

        self.overarching_policy.assign_free_parameters(var)

        if (self.Nsim < 2): # single simulation
            J = self.simulate()
            obj = J
        
        else: # MC Simulation to be adjusted
            Jhyd = np.empty() 
            Jenv = np.empty()  
            Jirr_def = np.empty()  

            for _ in range(self.Nsim):
                J = self.simulate()
                Jhyd = np.append(Jhyd, J[0])
                Jenv = np.append(Jenv, J[1])
                Jirr_def = np.append(Jirr_def, J[2])

            # objectives aggregation (average of Jhyd + worst 1st percentile for Jenv, Jirr)
            obj = np.append(obj, np.mean(Jhyd))
            obj = np.append(obj, np.percentile(Jenv, 99))
            obj = np.append(obj, np.percentile(Jirr_def, 99))

        # re-initialize policy parameters for further runs in the optimization mode
        self.overarching_policy.functions["release"].clearParameters()
        self.overarching_policy.functions["irrigation"].clearParameters()

        return list(obj)

    def simulate(self):
        """ Mathematical simulation over the specified simulation
        duration within a main for loop based on the mass-balance
        equation

        Parameters
        ----------
        self : model_zambezi object
            
        Returns
        -------
        JJ : np.array
            Array of calculated KPI values
        """

        ## INITIALIZATION: storage (s), level (h), decision (u), release(r) (Hydropower) : np.array
        s_kgu = np.full(self.H + 1, -999)  
        s_itt = np.full(self.H + 1, -999)  
        s_ka = np.full(self.H + 1, -999) 
        s_cb = np.full(self.H + 1, -999)
        s_kgl = np.full(self.H + 1, -999)
        h_kgu = np.full(self.H + 1, -999)
        h_itt = np.full(self.H + 1, -999)
        h_ka = np.full(self.H + 1, -999)
        h_cb = np.full(self.H + 1, -999)
        h_kgl = np.full(self.H + 1, -999)
        u_kgu = np.full(self.H,-999)
        u_itt = np.full(self.H,-999)
        u_ka = np.full(self.H,-999) 
        u_cb = np.full(self.H,-999)
        u_kgl = np.full(self.H + 1, -999)
        r_kgu = np.full(self.H + 1, -999)
        r_itt = np.full(self.H + 1, -999) 
        r_itt_delay = np.full(self.H+3,-999) # 2 months delay between Itezhitezhi and Kafue Gorge Upper  
        r_ka = np.full(self.H + 1, -999)
        r_cb = np.full(self.H + 1, -999) 
        r_kgl = np.full(self.H + 1, -999) 
        moy = np.full(self.H,-999, np.int64) # Month of the year: integer vector (others float!)

        # r_irr1 = np.full(self.H + 1, -999)
        r_irr2 = np.full(self.H + 1, -999)
        r_irr3 = np.full(self.H + 1, -999)
        r_irr4 = np.full(self.H + 1, -999)
        r_irr5 = np.full(self.H + 1, -999)
        r_irr6 = np.full(self.H + 1, -999)
        r_irr7 = np.full(self.H + 1, -999)
        r_irr8 = np.full(self.H + 1, -999)
        r_irr9 = np.full(self.H + 1, -999)

        # simulation variables Python -. (initialized as float of value 0 and empty np array)
        q_Itt, q_KafueFlats, q_KaLat, q_Bg, q_Cb, q_Cuando, q_Shire,\
        qTurb_Temp, qTurb_Temp_N, qTurb_Temp_S, headTemp,\
        hydTemp, hydTemp_dist, hydTemp_N, hydTemp_S, irrDef_Temp, irrDefNorm_Temp,\
        envDef_Temp, qTotIN, qTotIN_1 = tuple(20 * [float()]) # 
        sd_rd = np.empty(0) # storage and release resulting from daily integration
        uu = np.empty(0)
        gg_hydKGU, gg_hydITT, gg_hydKA, gg_hydCB, gg_hydKGL, gg_hydVF, \
        deficitHYD_tot, gg_irr2, gg_irr3, gg_irr4, gg_irr5, gg_irr6, gg_irr7, \
        gg_irr8, gg_irr9, gg_irr2_NormDef, gg_irr3_NormDef, gg_irr4_NormDef,\
        gg_irr5_NormDef, gg_irr6_NormDef, gg_irr7_NormDef, gg_irr8_NormDef, \
        gg_irr9_NormDef, deficitIRR_tot, gg_env, deficitENV_tot = tuple(26 * [np.empty(0)])
        input, outputDEF = tuple([np.empty(0), np.empty(0)])

        # initial condition
        s_kgu[0] = self.KafueGorgeUpper.getInitCond()
        s_itt[0] = self.Itezhitezhi.getInitCond() 
        s_ka[0] = self.Kariba.getInitCond()
        s_cb[0] = self.CahoraBassa.getInitCond()
        s_kgl[0] = self.KafueGorgeLower.getInitCond()

        qTotIN_1 = self.inflowTOT00 # initial inflow

        r_itt_delay[0] = 0 # September 1985 [m^3/sec] # 
        r_itt_delay[1] = 56.183290322580640 # October 1985 [m^3/sec] but divided for the number of days in January 1986 when it actually enters KGU # 
        r_itt_delay[2] = 59.670678571428570 # November 1985 [m^3/sec] but divided for the number of days in February 1986 when it actually enters KGU # 
        r_itt_delay[3] = 101.7307419354839 # December 1985 [m^3/sec] but divided for the number of days in March 1986 when it actually enters KGU # 

        # Run simulation

        for t in range(self.H):
            
            # month of the year 
            moy[t] = (self.initMonth+t-1)%(self.T)+1

            # inflows
            q_Itt = self.catchment_dict["IttCatchment"].getInflow(t) # Itezhitezhi inflow @ Kafue Hook Bridge # 
            q_KafueFlats = self.catchment_dict["KafueFlatsCatchment"].getInflow(t) # lateral flow @ Kafue Flats (upstream of Kafue Gorge Upper) # 
            q_KaLat = self.catchment_dict["KaCatchment"].getInflow(t) # Kariba inflow @ Victoria Falls increased by +10% # 
            q_Cb = self.catchment_dict["CbCatchment"].getInflow(t) #Cahora Bassa inflow (Luangwa and other tributaries) # 
            q_Cuando = self.catchment_dict["CuandoCatchment"].getInflow(t) # Kariba inflow @ Cuando river # 
            q_Shire = self.catchment_dict["ShireCatchment"].getInflow(t) # Shire discharge (upstream of the Delta) # 
            q_Bg = self.catchment_dict["BgCatchment"].getInflow(t) # Kariba inflow @ Victoria Falls increased by +10% # 
            qTotIN = q_Itt + q_KafueFlats + q_KaLat + q_Cb + q_Cuando + q_Shire + q_Bg

            # add the inputs for the function approximator (NN, RBF) black-box policy
            input = np.array([s_itt[t], s_kgu[t], s_ka[t], s_cb[t], s_kgl[t], moy[t], qTotIN_1])

            uu = self.overarching_policy.functions["release"].get_NormOutput(input) # Policy function is called here!
            u_itt[t], u_kgu[t], u_ka[t], u_cb[t], u_kgl[t] = tuple(uu) # decision per reservoir assigned

            # daily integration and assignment of monthly storage and release values
            sd_rd = self.Itezhitezhi.integration(12*self.integrationStep[moy[t]-1],t,s_itt[t],u_itt[t],q_Itt,moy[t]) # 2 # 24*integrationStep[moy[t]-1] ORIG
            s_itt[t+1] = sd_rd[0]
            r_itt[t+1] = sd_rd[1]
            r_itt_delay[t+3] = r_itt[t+1]*(self.integrationStep[moy[t]-1])/(self.integrationStep_delay[moy[t]-1])

            r_irr4[t+1] = self.overarching_policy.functions["irrigation"].get_output([q_KafueFlats+r_itt_delay[t+1],self.irr_demand_dict["irr_demand4"][moy[t]-1],4,self.irr_district_idx]) # compute the irrigation water diversion volume [m3/s]

            sd_rd = self.KafueGorgeUpper.integration(12*self.integrationStep[moy[t]-1],t,s_kgu[t],u_kgu[t],q_KafueFlats+r_itt_delay[t+1]-r_irr4[t+1],moy[t]) # 2 24*integrationStep[moy[t]-1] ORIG
            s_kgu[t+1] = sd_rd[0]
            r_kgu[t+1] = sd_rd[1]

            sd_rd = self.KafueGorgeLower.integration(12*self.integrationStep[moy[t]-1],t,s_kgl[t],u_kgl[t],r_kgu[t+1],moy[t]) # 2 24*integrationStep[moy[t]-1]
            s_kgl[t+1] = sd_rd[0]
            r_kgl[t+1] = sd_rd[1]
   
            r_irr2[t+1] = self.overarching_policy.functions["irrigation"].get_output([q_Bg+q_Cuando+q_KaLat,self.irr_demand_dict["irr_demand2"][moy[t]-1],2,self.irr_district_idx]) # compute the irrigation water diversion volume [m3/s] # 
    
            sd_rd = self.Kariba.integration_daily(self.integrationStep[moy[t]-1],t,s_ka[t],u_ka[t],q_Bg+q_Cuando+q_KaLat-r_irr2[t+1],moy[t]) # 2
            s_ka[t+1] = sd_rd[0]
            r_ka[t+1] = sd_rd[1] 
    
            r_irr3[t+1] = self.overarching_policy.functions["irrigation"].get_output([r_ka[t+1],self.irr_demand_dict["irr_demand3"][moy[t]-1],3,self.irr_district_idx]) # compute the irrigation water diversion volume [m3/s] # 
            r_irr5[t+1] = self.overarching_policy.functions["irrigation"].get_output([r_kgl[t+1],self.irr_demand_dict["irr_demand5"][moy[t]-1],5,self.irr_district_idx]) # compute the irrigation water diversion volume [m3/s] # 
            r_irr6[t+1] = self.overarching_policy.functions["irrigation"].get_output([r_ka[t+1]-r_irr3[t+1]+r_kgl[t+1]-r_irr5[t+1],self.irr_demand_dict["irr_demand6"][moy[t]-1],6,self.irr_district_idx]) # compute the irrigation water diversion volume [m3/s] # 

            sd_rd = self.CahoraBassa.integration(12*self.integrationStep[moy[t]-1],t,s_cb[t],u_cb[t],q_Cb+r_kgl[t+1]+r_ka[t+1]-(r_irr3[t+1]+r_irr5[t+1]+r_irr6[t+1]),moy[t])
            s_cb[t+1] = sd_rd[0] # 
            r_cb[t+1] = sd_rd[1] # 
            del sd_rd # 
    
            r_irr7[t+1] = self.overarching_policy.functions["irrigation"].get_output([r_cb[t+1],self.irr_demand_dict["irr_demand7"][moy[t]-1],7,self.irr_district_idx]) # compute the irrigation water diversion volume [m3/s] # 
            r_irr8[t+1] = self.overarching_policy.functions["irrigation"].get_output([r_cb[t+1]-r_irr7[t+1],self.irr_demand_dict["irr_demand8"][moy[t]-1],8,self.irr_district_idx]) # compute the irrigation water diversion volume [m3/s] # 
            r_irr9[t+1] = self.overarching_policy.functions["irrigation"].get_output([r_cb[t+1]-r_irr7[t+1]+q_Shire-r_irr8[t+1],self.irr_demand_dict["irr_demand9"][moy[t]-1],9,self.irr_district_idx]) # compute the irrigation water diversion volume [m3/s] # 
    
            qTotIN_1 = qTotIN

            # TIME-SEPARABLE OBJECTIVES

            # HYDROPOWER PRODUCTION (MWh/day)
            # Itezhitezhi
            h_itt[t] = self.Itezhitezhi.storageToLevel(s_itt[t]) 
            qTurb_Temp = min(r_itt[t+1],2*306) 
                
            headTemp = (40.50-(1030.5-h_itt[t])) 
            hydTemp = ((qTurb_Temp*headTemp*1000*9.81*0.89*(24*self.integrationStep[moy[t]-1]))/1000000)*12/1000000 # [TWh/year] 
            hydTemp_dist = abs( hydTemp - self.tp_Itt[moy[t]-1] )
            gg_hydITT = np.append(gg_hydITT, hydTemp_dist ) 

            # Kafue Gorge Upper
            h_kgu[t] = self.KafueGorgeUpper.storageToLevel(s_kgu[t]) 
            qTurb_Temp = min(r_kgu[t+1],6*42) 
                
            headTemp = (397-(977.6-h_kgu[t])) 
            hydTemp = ((qTurb_Temp*headTemp*1000*9.81*0.61*(24*self.integrationStep[moy[t]-1]))/1000000)*12/1000000 # [TWh/year] # 
            hydTemp_dist = abs( hydTemp - self.tp_Kgu[moy[t]-1] )
            gg_hydKGU = np.append(gg_hydKGU, hydTemp_dist ) # 

            # Kariba North
            h_ka[t] = self.Kariba.storageToLevel(s_ka[t]) # 
            qTurb_Temp_N = min(r_ka[t+1]*0.488,6*200) # Kariba North has an efficiency of 48% -. 49% of the total release goes through Kariba North # 
            headTemp = (108-(489.5-h_ka[t])) # 
            hydTemp_N = ((qTurb_Temp_N*headTemp*1000*9.81*0.48*(24*self.integrationStep[moy[t]-1]))/1000000)*12/1000000 # [TWh/year] # 

            # Kariba South
            qTurb_Temp_S = min(r_ka[t+1]*0.512,6*140) # 
                
            headTemp = (110-(489.5-h_ka[t])) # 
            hydTemp_S = ((qTurb_Temp_S*headTemp*1000*9.81*0.51*(24*self.integrationStep[moy[t]-1]))/1000000)*12/1000000 # [TWh/year] # 

            hydTemp = hydTemp_N + hydTemp_S # 
            hydTemp_dist = abs( hydTemp - self.tp_Ka[moy[t]-1] )
            gg_hydKA = np.append(gg_hydKA, hydTemp_dist ) # 

            # Cahora Bassa
            h_cb[t] = self.CahoraBassa.storageToLevel(s_cb[t]) # 
            qTurb_Temp = min(r_cb[t+1],5*452) # 
                            
            headTemp = (128-(331-h_cb[t])) # 
            hydTemp = ((qTurb_Temp*headTemp*1000*9.81*0.73*(24*self.integrationStep[moy[t]-1]))/1000000)*12/1000000 # [TWh/year] # 
            hydTemp_dist = abs( hydTemp - self.tp_Cb[moy[t]-1] )
            gg_hydCB = np.append(gg_hydCB, hydTemp_dist ) # 

            # Kafue Gorge Lower
            h_kgl[t] = self.KafueGorgeLower.storageToLevel(s_kgl[t]) # 
            qTurb_Temp = min(r_kgl[t+1],97.4*5) # 
                
            headTemp = (182.7-(586-h_kgl[t])) # 
            hydTemp = ((qTurb_Temp*headTemp*1000*9.81*0.88*(24*self.integrationStep[moy[t]-1]))/1000000)*12/1000000 # [TWh/year] # 
            hydTemp_dist = abs( hydTemp - self.tp_Kgl[moy[t]-1] )
            gg_hydKGL = np.append(gg_hydKGL, hydTemp_dist ) # 

            # Victoria Falls (ROR)
            qTurb_Temp = min(max(q_Bg+q_Cuando-self.MEF_VictoriaFalls[moy[t]-1],0),(5*1.2+6*12+6*12)) # 
            headTemp = 100 # 
            hydTemp = ((qTurb_Temp*headTemp*1000*9.81*0.88*(24*self.integrationStep[moy[t]-1]))/1000000)*12/1000000 # [TWh/year] # 
            gg_hydVF = np.append(gg_hydVF,hydTemp ) # 

            deficitHYD_tot = np.append(deficitHYD_tot, gg_hydITT[t]+gg_hydKGU[t]+gg_hydKA[t]+gg_hydCB[t]+gg_hydKGL[t]) # energy production 
        
            irrDef_Temp = pow(max(self.irr_demand_dict["irr_demand2"][moy[t]-1]-r_irr2[t+1],0),2) 
            gg_irr2 = np.append(gg_irr2, irrDef_Temp ) # SQUARED irrigation deficit
            irrDefNorm_Temp = self.g_deficit_norm( irrDef_Temp, self.irr_demand_dict["irr_demand2"][moy[t]-1] )
            gg_irr2_NormDef = np.append(gg_irr2_NormDef, irrDefNorm_Temp )

            irrDef_Temp = pow(max(self.irr_demand_dict["irr_demand3"][moy[t]-1]-r_irr3[t+1],0),2) 
            gg_irr3 = np.append(gg_irr3, irrDef_Temp ) # SQUARED irrigation deficit
            irrDefNorm_Temp = self.g_deficit_norm( irrDef_Temp, self.irr_demand_dict["irr_demand3"][moy[t]-1] )
            gg_irr3_NormDef = np.append(gg_irr3_NormDef, irrDefNorm_Temp )

            irrDef_Temp = pow(max(self.irr_demand_dict["irr_demand4"][moy[t]-1]-r_irr4[t+1],0),2) 
            gg_irr4 = np.append(gg_irr4, irrDef_Temp ) # SQUARED irrigation deficit 
            irrDefNorm_Temp = self.g_deficit_norm( irrDef_Temp, self.irr_demand_dict["irr_demand4"][moy[t]-1] )
            gg_irr4_NormDef = np.append(gg_irr4_NormDef, irrDefNorm_Temp )

            irrDef_Temp = pow(max(self.irr_demand_dict["irr_demand5"][moy[t]-1]-r_irr5[t+1],0),2) 
            gg_irr5 = np.append(gg_irr5, irrDef_Temp ) # SQUARED irrigation deficit 
            irrDefNorm_Temp = self.g_deficit_norm( irrDef_Temp, self.irr_demand_dict["irr_demand5"][moy[t]-1] )
            gg_irr5_NormDef = np.append(gg_irr5_NormDef, irrDefNorm_Temp )

            irrDef_Temp = pow(max(self.irr_demand_dict["irr_demand6"][moy[t]-1]-r_irr6[t+1],0),2) 
            gg_irr6 = np.append(gg_irr6, irrDef_Temp ) # SQUARED irrigation deficit 
            irrDefNorm_Temp = self.g_deficit_norm( irrDef_Temp, self.irr_demand_dict["irr_demand6"][moy[t]-1] )
            gg_irr6_NormDef = np.append(gg_irr6_NormDef, irrDefNorm_Temp )

            irrDef_Temp = pow(max(self.irr_demand_dict["irr_demand7"][moy[t]-1]-r_irr7[t+1],0),2)
            gg_irr7 = np.append(gg_irr7, irrDef_Temp ) # SQUARED irrigation deficit
            irrDefNorm_Temp = self.g_deficit_norm( irrDef_Temp, self.irr_demand_dict["irr_demand7"][moy[t]-1] )
            gg_irr7_NormDef = np.append(gg_irr7_NormDef, irrDefNorm_Temp )

            irrDef_Temp = pow(max(self.irr_demand_dict["irr_demand8"][moy[t]-1]-r_irr8[t+1],0),2)
            gg_irr8 = np.append(gg_irr8, irrDef_Temp ) # SQUARED irrigation deficit
            irrDefNorm_Temp = self.g_deficit_norm( irrDef_Temp, self.irr_demand_dict["irr_demand8"][moy[t]-1] )
            gg_irr8_NormDef = np.append(gg_irr8_NormDef, irrDefNorm_Temp )

            irrDef_Temp = pow(max(self.irr_demand_dict["irr_demand9"][moy[t]-1]-r_irr9[t+1],0),2)
            gg_irr9 = np.append(gg_irr9, irrDef_Temp ) # SQUARED irrigation deficit
            irrDefNorm_Temp = self.g_deficit_norm( irrDef_Temp, self.irr_demand_dict["irr_demand9"][moy[t]-1] )
            gg_irr9_NormDef = np.append(gg_irr9_NormDef, irrDefNorm_Temp )

            deficitIRR_tot = np.append(deficitIRR_tot, gg_irr2_NormDef[t]+gg_irr3_NormDef[t]+gg_irr4_NormDef[t]+gg_irr5_NormDef[t]+gg_irr6_NormDef[t]+gg_irr7_NormDef[t]+gg_irr8_NormDef[t]+gg_irr9_NormDef[t] ) # SQUARED irrigation deficit

            # DELTA ENVIRONMENT DEFICIT 
            envDef_Temp = pow(max(self.qDelta[moy[t]-1]-(r_cb[t+1]-r_irr7[t+1]-r_irr8[t+1]+q_Shire-r_irr9[t+1]),0),2)
            gg_env = np.append(gg_env, envDef_Temp )

            deficitENV_tot = np.append(deficitENV_tot, gg_env[t] ) # Delta environment deficit

            # clear
            input = np.empty(0)
            uu = np.empty(0)

        # NOT Super clear if below implementation is correct. Check!!!!
        # time-aggregation = average of step costs starting from month 1 (i.e., January 1974) // 

        JJ = np.empty(0)
        JJ = np.append(JJ, np.mean(deficitHYD_tot))
        JJ = np.append(JJ, np.mean(deficitENV_tot))
        JJ = np.append(JJ, np.mean(deficitIRR_tot))

        return JJ

    # Deficit
    def g_deficit(self, q, w):

        d = w - q
        if (d < 0.0):
            d = 0.0

        return d*d

    # Normalized SQUARED deficit
    def g_deficit_norm(self, defp, w):
        """Takes two floats and divides the first by
        the square of the second.

        Parameters
        ----------
        defp : float
        w : float
            
        Returns
        -------
        def_norm : float
        """

        def_norm = 0
        if (w == 0.0):
            def_norm = 0.0
        else:
            def_norm = defp/(pow(w,2))
        
        return def_norm

    def readFileSettings(self):

        def nested_getattr(object, nested_attr_list):
            
            obj_copy = object
            for item in nested_attr_list:
                obj_copy = getattr(obj_copy, item)
            return obj_copy

        input_model = pd.read_excel("../settings/excel_settings.xlsx", usecols=["AttributeName", "Value", "Type"],
         sheet_name="ModelParameters", skiprows=3)

        input_policy = pd.read_excel("../settings/excel_settings.xlsx", usecols=["AttributeName", "Value", "Type"],
         sheet_name="PolicyParameters", skiprows=3)

        input_df = pd.concat([input_model, input_policy], ignore_index=True)

        for _, row in input_df.iterrows():
            
            attribute_name_list = row["AttributeName"].split(".")
            if len(attribute_name_list) == 1:
                object, name = self, attribute_name_list[0]
                
            else:
                name = attribute_name_list.pop(-1)
                object = nested_getattr(self, attribute_name_list)
            
            if row.Type == "int":
                setattr(object, name, int(row["Value"]))
            elif row.Type == "float":
                setattr(object, name, float(row["Value"]))
            elif row.Type == "np.array":
                value = np.array(object=(row.Value.replace(" ", "")).split(";"), dtype=float)
                setattr(object, name, value)
            elif row.Type == "str":
                setattr(object, name, str(row["Value"]))
                    
        self.integrationStep = utils.loadIntVector("../data/number_days_month.txt", self.T)
        self.integrationStep_delay = utils.loadIntVector("../data/number_days_month_delay.txt", self.T)

        #self.moy_file = utils.loadIntVector("../data/moy_1986_2005.txt", self.H)

        self.catchment_param_dict["Itt_catch_param"].CM = 1
        self.catchment_param_dict["Itt_catch_param"].inflow_file.filename = "../data/qInfItt_1January1986_31Dec2005.txt"
        self.catchment_param_dict["Itt_catch_param"].inflow_file.row = self.H

        self.catchment_param_dict["KafueFlats_catch_param"].CM = 1
        self.catchment_param_dict["KafueFlats_catch_param"].inflow_file.filename = "../data/qKafueFlats_1January1986_31Dec2005.txt"
        self.catchment_param_dict["KafueFlats_catch_param"].inflow_file.row = self.H

        self.catchment_param_dict["Ka_catch_param"].CM = 1
        self.catchment_param_dict["Ka_catch_param"].inflow_file.filename = "../data/qInfKaLat_1January1986_31Dec2005.txt"
        self.catchment_param_dict["Ka_catch_param"].inflow_file.row = self.H

        self.catchment_param_dict["Cb_catch_param"].CM = 1
        self.catchment_param_dict["Cb_catch_param"].inflow_file.filename = "../data/qInfCb_1January1986_31Dec2005.txt"
        self.catchment_param_dict["Cb_catch_param"].inflow_file.row = self.H

        self.catchment_param_dict["Cuando_catch_param"].CM = 1
        self.catchment_param_dict["Cuando_catch_param"].inflow_file.filename = "../data/qCuando_1January1986_31Dec2005.txt"
        self.catchment_param_dict["Cuando_catch_param"].inflow_file.row = self.H

        self.catchment_param_dict["Shire_catch_param"].CM = 1
        self.catchment_param_dict["Shire_catch_param"].inflow_file.filename = "../data/qShire_1January1986_31Dec2005.txt"
        self.catchment_param_dict["Shire_catch_param"].inflow_file.row = self.H

        self.catchment_param_dict["Bg_catch_param"].CM = 1
        self.catchment_param_dict["Bg_catch_param"].inflow_file.filename = "../data/qInfBg_1January1986_31Dec2005.txt"
        self.catchment_param_dict["Bg_catch_param"].inflow_file.row = self.H

#struct
class policy_parameters_construct:

    def __init__(self):
        self.tPolicy = int()
        self.policyInput = int()
        self.policyOutput = int()
        self.policyStr = int()

        self.mIn, self.mOut, self.MIn, self.MOut = tuple(4 * [np.empty(0)])
        self.muIn, self.muOut, self.stdIn, self.stdOut = tuple(4 * [np.empty(0)])

class irr_function_parameters:
    def __init__(self):
        self.num_irr = int()
        self.mParam = np.empty(0)
        self.MParam = np.empty(0)
