import numpy as np
import math

'''
This file's purpose is to conduct the costing caluclations of the process for evaluation of its performance

Costs will be taken from various sources of the literature and should be cited when defined here

Inputs should be the following:
 - Membranes total area
 - Total compressors duty
 - Total water coolers duty
 - Total cryogenic coolers duty
 - Performance criterion (purity and recovery)

 This module will return the evalution to be used in the optimisation algorithm
'''
'''
    Process_specs = {
        "Membrane_1": Membrane_1,
        "Membrane_2": Membrane_2,
        "Pre_cond_1": Train1,
        "Pre_cond_2": Train2,
        "Comp_train": Liquefaction,
        "Ret1_Savings": Ret1_savings,
        "Ret2_Savings": Ret2_savings,
        "Process_param": Process_param,
    }
'''


def Costing(Process_specs, Process_param, Membrane_1, Membrane_2): #process specs is dictionary with mem1, mem2, Pre_cond1, Pre_cond2, Compressor_train, Process_param

    ''' Constants'''

    Compressor_param = [490000,16800,0.6] # Sinnott - purchased equipment cost on a U.S. Gulf Coast basis, Jan. 2007 (CE index (CEPCI) = 509.7, NF refinery inflation index = 2059.1)
    Compressor_IF = 2.5 #Installation factor for compressors (Hand 1958) - this is the ratio of installed cost to purchased cost
    Cooler_param = [24000,46,1.2] # Sinnott eur/m2    
    HEx_IF = 3.5 #Installation factor for heat exchangers (Hand 1958)
    Cryogenic_cooler_param = [2.4647 , 0.01812] # $(2017) per GJ, equation from Lyuben (2017) with Cost ($/GJ) = exp( 2.4647 - 0.01812 * T ) where T is the temperature in Celsius.
    Membrane_param = 42 # eur/m2
    Membrane_IF = 1.5

    Electricity_cost = 58.1*1e-3 # eur/kWh - 2014 - CEMCAP

    Water_cost = 0.39 # eur/m3 (water) 
    Cooling_fluid = 50 # eur/kWh
    Water_dT = 30 # In/out difference for cooling water (assumed constant for all coolers)
    Cp_water = 4.180 # kJ/kg.K - specific heat capacity of water
    
    TDC_cement_plant = 149.8 * 1e6 # Total direct cost of cement plant (No CC) in 2014 (CEMCAP)
    Carbon_Tax = 100 # eur/tCO2 - assumed carbon tax


    Index_2007 = 525.4 # Chemical Engineering Plant Cost Index (CEPCI) average 2007
    Index_2014 = 571.6 # Chemical Engineering Plant Cost Index (CEPCI) average 2014
    Index_2017 = 567.5 # Chemical Engineering Plant Cost Index (CEPCI) for Jan 2017 - for cryogenic costing
    Discount_rate = 0.08 # 8% - CEMCAP
    Process_Contingency = 0.30 # 30% - contingency for process design for TRL-7
    Indirect_Cost = 0.14  # CEMCAP
    Owner_Cost = 0.07 # CEMCAP
    Project_Contingency = 0.15 # CEMCAP

    # Assumption for cost of equipment based on size parameters. Taking the average duty across the number of units is quite close to doing it unit by unit and is much simpler.
    def cost_sinnott(param, size):
        cost = param[0] + param[1] * (size ** param[2]) 
        return cost

    #---------------------------#
    #---------- CAPEX ----------#
    #---------------------------#

    C_compressor_1 = (Process_specs["Pre_cond_1"][-1] + 1) * cost_sinnott(Compressor_param, (Process_specs["Pre_cond_1"][0])/ (Process_specs["Pre_cond_1"][-1] + 1)) # Train 101  has 1 compressor, Train 102 has 2 etc. 
    C_compressor_2 = (Process_specs["Pre_cond_2"][-1] + 1) * cost_sinnott(Compressor_param, Process_specs["Pre_cond_2"][0]/(Process_specs["Pre_cond_2"][-1] + 1)) # Train 201 has 1 compressor, Train 202 has 2 etc.
    C_compressor_train = (Process_specs["Comp_train"][-1] + 4) * cost_sinnott(Compressor_param, Process_specs["Comp_train"][0])/((Process_specs["Comp_train"][-1] + 4)) # Train 301 has 4 compressors, Train 302 has 5 etc.
    C_compressor = (C_compressor_1 + C_compressor_2 + C_compressor_train) * Compressor_IF # Installed cost of compressors
    #print(f'Compressor Capex: {C_compressor/1e6:.0f} million euros')
    
    C_cooler_1 = (Process_specs["Pre_cond_1"][-1] + 2) * cost_sinnott(Cooler_param, Process_specs["Pre_cond_1"][1]/(Process_specs["Pre_cond_1"][-1] + 2)) # Train 101  has 3 (2 coolers + 1 for retentate integration) heat exchangers, Train 102 has 4 etc. 
    C_cooler_2 = (Process_specs["Pre_cond_2"][-1] + 1) * cost_sinnott(Cooler_param, Process_specs["Pre_cond_2"][1]*(Process_specs["Pre_cond_2"][-1] + 1)) # Train 101  has 2 heat exchangers, Train 102 has 2 etc.  
    C_cooler_train = (Process_specs["Comp_train"][-1] + 4) * cost_sinnott(Cooler_param, (Process_specs["Comp_train"][1]/(Process_specs["Comp_train"][-1] + 4)))
    C_cooler = (C_cooler_1 + C_cooler_2 + C_cooler_train) * HEx_IF # Intalled cost of coolers
    #print(f'Cooler Capex: {C_cooler/1e6:.0f} million euros')
    
    C_membrane_1 = Process_specs["Membrane_1"]["Area"] * Membrane_param # Cost of membranes initially - replacement is taken as part of the OPEX
    C_membrane_2 = Process_specs["Membrane_2"]["Area"] * Membrane_param
    C_membrane = (C_membrane_1 + C_membrane_2) * Membrane_IF # Installed cost of membranes
    #print(f'Membrane Capex: {C_membrane/1e6:.0f} million euros')
    
    #Capex for energy recovery in retentate streams
    C_expander_1 = cost_sinnott(Compressor_param, Process_specs["Ret1_Savings"][0])  # Cost of expander 100
    C_expander_2 = cost_sinnott(Compressor_param, Process_specs["Ret2_Savings"][0])   # Cost of expander 200
    C_expander = (C_expander_1 + C_expander_2) * Compressor_IF # Installed cost of expanders
    # Assume heat exchanger cost from retentate energy recovery is taken as part of the 'normal' compression without heat recovery.

    Capex_tot_2007 = C_compressor + C_cooler + C_membrane + C_expander # Total installed cost of the process in 2007 money
    Capex_tot_2014 = Capex_tot_2007 * Index_2014/Index_2007 #convert to 2014 money
    Capex_tot_2014 *= 0.7541  #convert to euros using average convertion rate in 2014 (0.7541 according to https://www.exchangerates.org.uk/USD-EUR-spot-exchange-rates-history-2014.html )
    
    DEC = Capex_tot_2014 # (eur) Direct Equipment Cost 
    TDC = DEC * Process_Contingency # Total direct cost with process contingency
    TPC = TDC * ( 1 + Indirect_Cost + Owner_Cost + Project_Contingency) # Total Plant Cost

    #--------------------------#
    #---------- OPEX ----------#
    #--------------------------#

    O_compressor_1 = Process_specs["Pre_cond_1"][0] * Process_param["Operating_hours"] * Electricity_cost
    O_compressor_2 = Process_specs["Pre_cond_2"][0] * Process_param["Operating_hours"] * Electricity_cost
    O_compressor_train = Process_specs["Comp_train"][0] * Process_param["Operating_hours"] * Electricity_cost
    O_compressor = (O_compressor_1 + O_compressor_2 + O_compressor_train) # Opex of compressors
    #print(f'Compressor Opex: {O_compressor/1e6:.0f} million euros / yr')
    
    O_cooler_1 = Process_specs["Pre_cond_1"][2] * Process_param["Operating_hours"] / 998 * Water_cost  # conversion from kg/hr to m3/yr times cost per m3
    O_cooler_2 = Process_specs["Pre_cond_2"][2] * Process_param["Operating_hours"] / 998 * Water_cost
    O_cooler_train = Process_specs["Comp_train"][2] * Process_param["Operating_hours"] / 998 * Water_cost
    O_cooler = (O_cooler_1 + O_cooler_2 + O_cooler_train) # Opex of coolers
    #print(f'Cooler Opex: {O_cooler/1e6:.0f} million euros / yr')
   
    O_membrane = (C_membrane_1 + C_membrane_2) / Process_param["Replacement_rate"]# Opex of membranes, assuming replacement every Replacement_rate years without installation factor

    ### Opex Savings from retentate energy recovery ###
    
    # Retentate power recovery is done through expanders.
    O_expander_1 = - 0.95 * Process_specs["Ret1_Savings"][0] * Process_param["Operating_hours"] * Electricity_cost # 95% efficiency assumed for the expander
    O_expander_2 = - 0.95 * Process_specs["Ret2_Savings"][0] * Process_param["Operating_hours"] * Electricity_cost
    O_expander = (O_expander_1 + O_expander_2) # Opex of expanders

    # Consider the heat integration from retentate as an equivalent water saving, as it is not a direct cost but a saving on the water cooling system.
    O_heater = - ( (Process_specs["Ret1_Savings"][1] + Process_specs["Ret2_Savings"][1]) / (Cp_water * Water_dT) ) * Process_param["Operating_hours"] / 998 * Water_cost

    Variable_Opex = (O_compressor + O_cooler + O_membrane + O_expander + O_heater) #cost per year

    Annual_Maintenance = 0.025 * TPC # (eur/yr) ; 2.5% of total plant cost per year (CEMCAP)
    Maintenance_Labour = 0.4 * Annual_Maintenance # 40% of annual maintenance cost (CEMCAP)
    Insurance_et_al = 0.02 * TPC # Insurance and location tax, including overhead and miscellaneous regulatory fees set 2% of total plant cost per year (CEMCAP)
    Operating_Labour = 20 * 60e3 # 20 operators at 60k eur/yr each (CEMCAP)
    Admin_labour = 0.3 * (Maintenance_Labour + Operating_Labour) # cost for administrative and support labour fix at 30% of maintenance and operating labour costs (CEMCAP)
    Fixed_Opex = Annual_Maintenance + Maintenance_Labour + Insurance_et_al + Operating_Labour + Admin_labour # Fixed Opex per year

    Total_Opex = Variable_Opex + Fixed_Opex #eur/yr

    #-------------------------------------#
    #--- Cryogenic cost taken as a TAC ---#
    #-------------------------------------#
    
    Cryogenic_1_Cost = math.exp(Cryogenic_cooler_param[0] - Cryogenic_cooler_param[1] * (Membrane_1["Temperature"] - 273.15)) #$(2017) per GJ
    TAC_Cryo_1 = Cryogenic_1_Cost * Process_specs["Pre_cond_1"][-1] * Process_param["Operating_hours"] * 0.8865 #convert to cost per year (eur(2017)/yr)
    TAC_Cryo_1 *= Index_2014/Index_2017 #convert to 2014 money
    Cryogenic_2_Cost = math.exp(Cryogenic_cooler_param[0] - Cryogenic_cooler_param[1] * (Membrane_2["Temperature"] - 273.15)) 
    TAC_Cryo_2 = Cryogenic_2_Cost * Process_specs["Pre_cond_2"][-1] * Process_param["Operating_hours"] * 0.8865
    TAC_Cryo_2 *= Index_2014/Index_2017 
    TAC_Cryo = TAC_Cryo_1 + TAC_Cryo_2

    ### Penalty evaluation for Purity / Recovery ###
    Penalty_purity = 2.5 * 1e8 * (Process_param["Target_Purity"] - Process_specs["Purity"]) if (Process_param["Target_Purity"] - Process_specs["Purity"]) > 0 else 0 #10 million pounds per year penalty per percentage purity under target
    
    CO2_emission = (1- Process_specs["Recovery"]) * Process_specs["Feed"]["Feed_Composition"][0] * Process_specs["Feed"]["Feed_Flow"] #(mol/s) CO2 emissions from the process
    CO2_emission *= Process_param["Operating_hours"] * 3600 # convert to mol/yr
    CO2_emission *= 44.01 * 1e-6 # convert to tonnes/yr (44.01 g/mol)
    Penalty_CO2_emission = CO2_emission * Carbon_Tax #eur/yr 

    Penalty = Penalty_purity + Penalty_CO2_emission # Total penalty for purity and CO2 emissions

    ### Estimate cost of carbon capture process as a TAC, add cryogenic cost if any here.
    TAC_CC = Capex_tot_2014/Process_param["Lifetime"] + Total_Opex + TAC_Cryo

    ### Evaluation ###
    Evaluation = TAC_CC + Penalty

    Economics = {
        "Evaluation": Evaluation,
        "Purity": Process_specs["Purity"],  # Purity of the product
        "Recovery": Process_specs["Recovery"],  # Recovery of the product
        "TAC_CC": TAC_CC,  # Total Annualised Cost of Carbon Capture
        "Capex_tot": Capex_tot_2014,  # Total Capital Expenditure in 2014 money"
        "Opex_tot": Total_Opex,  # Total Operational Expenditure per year
        "Variable_Opex": Variable_Opex,  # Variable Operational Expenditure per year
        "Fixed_Opex": Fixed_Opex,  # Fixed Operational Expenditure per year
        "TAC_Cryo": TAC_Cryo,  # Total Annualised Cost of Cryogenic cooling
        "CO2_emission": CO2_emission,  # CO2 emissions from the process in tonnes per year
        "C_compressor": C_compressor,  # Capital cost of compressors
        "C_cooler": C_cooler,  # Capital cost of coolers
        "C_membrane": C_membrane,  # Capital cost of membranes
        "C_expander": C_expander,  # Capital cost of expanders
        "O_compressor": O_compressor,  # Operational cost of compressors per year
        "O_cooler": O_cooler,  # Operational cost of coolers per year
        "O_membrane": O_membrane,  # Operational cost of membranes per year
        "O_expander": O_expander,  # Operational cost of expanders per year
        "O_heater": O_heater,  # Operational cost of heat integration from retentate per year
        "Penalty_purity": Penalty_purity,  # Penalty for purity under target
        "Penalty_CO2_emission": Penalty_CO2_emission,  # Penalty for CO2 emissions under target

        }


    return Economics
