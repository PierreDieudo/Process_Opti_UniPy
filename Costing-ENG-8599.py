import numpy as np
import math

'''
This file is a modification of the original Costing script. 

Its purpose is to bring a general costing function that can be used for any process, not just the one from the Ferrari Paper.

Each cost is categorised (Capex, Opex, etc.) and the costing function returns a dictionary with all the relevant information.

The costing function is made so any number of data sets falling in a certain category can be handled: e.g., multiple sets of compressors from different trains all taken care under the compressor Capex and Opex categories.

'''


'''
    Process dependant variables:
Process_param = {
    "Target_Purity":  - ,
    "Target_Recovery": - ,
    "Replacement_rate": - , # Replacement rate of the membranes (in yr)
    "Operating_hours": - , # Operating hours per year
    "Lifetime": - , # Lifetime of the plant (in yr)
    "Base Plant Cost": - , # Total direct cost of plant (no CCS)
    "Contingency" = 0.3 , # or 0.4 (30% or 40% contingency for process design - based on TRL)
    }

    Process data: 
Process_specs = {

"Compressor_trains" : ((duty1, number_of_compressors1), ... , (dutyi, number_of_compressorsi)), # Compressor trains data)
"Cooler_trains" : ((area1, waterflow1, number_of_coolers1), ... , (areai, waterflowi, number_of_coolersi)), # Cooler trains data
"Membranes" : (Membrane_1, ..., Membrane_i), # Membrane data)
"Expanders" : ((expander1_duty), ...(expanderi_duty)), # Expander data
"Heaters" : ((heater1_duty), ...(heateri_duty)), # Heater data
"Cryogenics" = ((cooling_power1, temperature1), ... (cooling_poweri, temperaturei)), # Cryogenic cooling data
}

    Independent variables:
Cost from sizing factors - Sinnott
Installation factors
Electricity Cost
Cooling water cost
Carbon Tax
CEPCI
Discount Rate
Project Contingency / Fixed Opex factors
'''




def Costing(Process_specs, Process_param): #process specs is dictionary with mem1, mem2, Pre_cond1, Pre_cond2, Compressor_train, Process_param

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
    Water_dT = 30 # In/out difference for cooling water (assumed constant for all coolers)
    Cp_water = 4.180 # kJ/kg.K - specific heat capacity of water
    Carbon_Tax = 100 # eur/tCO2 - assumed carbon tax
    Index_2007 = 525.4 # Chemical Engineering Plant Cost Index (CEPCI) average 2007
    Index_2014 = 571.6 # Chemical Engineering Plant Cost Index (CEPCI) average 2014
    Index_2017 = 567.5 # Chemical Engineering Plant Cost Index (CEPCI) for Jan 2017 - for cryogenic costing
    Discount_rate = 0.08 # 8% - CEMCAP
    Indirect_Cost = 0.14  # CEMCAP
    Owner_Cost = 0.07 # CEMCAP
    Project_Contingency = 0.15 # CEMCAP
    Indirect_Emission_rate = 262*1e-6 # Electricity generation specific emissions (EU 2014) - in tnCO2/kWh

    
    '''
    Process_specs = {
    ...
    "Compressor_trains" : ((duty1, number_of_compressors1), ... , (dutyi, number_of_compressorsi)), # Compressor trains data)
    "Cooler_trains" : ((area1, waterflow1, number_of_coolers1), ... , (areai, waterflowi, number_of_coolersi)), # Cooler trains data
    "Membranes" : (Membrane_1, ..., Membrane_i), # Membrane data)
    "Expanders" : ((expander1_duty), ...(expanderi_duty)), # Expander data
    "Heaters" : ((heater1_duty), ...(heateri_duty)), # Heater data
    "Cryogenics" = ((cooling_power1, temperature1), ... (cooling_poweri, temperaturei)), # Cryogenic cooling data
    }
    '''

    # Assumption for cost of equipment based on size parameters. Taking the average duty across the number of units is quite close to doing it unit by unit and is much simpler.
    def cost_sinnott(param, size):
        cost = param[0] + param[1] * (size ** param[2]) 
        return cost

    #---------------------------#
    #---------- CAPEX ----------#
    #---------------------------#

    C_compressor = 0 # Total installed cost of compressors
    for train in Process_specs["Compressor_trains"]:
        C_compressor += (train[1]) * cost_sinnott(Compressor_param, train[0]/train[1]) if train[0] > 0 else 0 # Compressor trains data is a list of [duty, number_of_compressors] pairs
    C_compressor *= Compressor_IF # Installed cost of compressors
    #print(f'Compressor Capex: {C_compressor/1e6:.0f} million euros')
    
    C_cooler = 0 # Total installed cost of coolers
    for train in Process_specs["Cooler_trains"]:
        C_cooler += (train[2]) * cost_sinnott(Cooler_param, train[0]/train[2]) if train[0] > 0 else 0
    C_cooler *= HEx_IF # Installed cost of coolers
    #print(f'Cooler Capex: {C_cooler/1e6:.0f} million euros')

    C_membrane = 0 # Total installed cost of membranes
    for Mem in Process_specs["Membranes"]:
        C_membrane += Mem["Area"] * Membrane_param
    C_membrane *= Membrane_IF # Installed cost of membranes
    #print(f'Membrane Capex: {C_membrane/1e6:.0f} million euros')
    
    C_expander = 0 #Installed cost for energy recovery in retentate streams
    for Expander_duty in Process_specs["Expanders"]:
        C_expander += cost_sinnott(Compressor_param, Expander_duty)
    C_expander *= Compressor_IF 

    Capex_tot_2007 = C_compressor + C_cooler + C_membrane + C_expander # Total installed cost of the process in 2007 money
    Capex_tot_2014 = Capex_tot_2007 * Index_2014/Index_2007 #convert to 2014 money
    Capex_tot_2014 *= 0.7541  #convert to euros using average convertion rate in 2014 (0.7541 according to https://www.exchangerates.org.uk/USD-EUR-spot-exchange-rates-history-2014.html )
    
    DEC = Capex_tot_2014 # (eur) Direct Equipment Cost 
    TDC = DEC * ( 1 + Process_param["Contingency"]) # Total direct cost with process contingency
    TPC = TDC * ( 1 + Indirect_Cost + Owner_Cost + Project_Contingency) # Total Plant Cost

    #--------------------------#
    #---------- OPEX ----------#
    #--------------------------#

    O_compressor = 0 # Operational cost of compressors
    for train in Process_specs["Compressor_trains"]:
        O_compressor += train[0] * Process_param["Operating_hours"] * Electricity_cost
    Power_Consumption = O_compressor / Electricity_cost # Power consumption in kWh/yr
    #print(f'Compressor Opex: {O_compressor/1e6:.0f} million euros / yr')
    
    O_cooler = 0 # Operational cost of coolers
    for train in Process_specs["Cooler_trains"]:
        O_cooler += train[1] * Process_param["Operating_hours"] / 998 * Water_cost
    #print(f'Cooler Opex: {O_cooler/1e6:.0f} million euros / yr')
   
    O_membrane = (C_membrane) / Process_param["Replacement_rate"]# Opex of membranes, assuming replacement every Replacement_rate years without installation factor

    ### Opex Savings from retentate energy recovery ###
    O_expander = 0 # Retentate power recovery is done through expanders.
    for Expander_duty in Process_specs["Expanders"]:
        O_expander += -0.95 * Expander_duty * Process_param["Operating_hours"] * Electricity_cost
    Power_Consumption += O_expander / Electricity_cost # Update power consumption with the savings from retentate energy recovery

    # Consider the heat integration from retentate as an equivalent water saving, as it is not a direct cost but a saving on the water cooling system.
    O_heater = 0 # Consider the heat integration from retentate as an equivalent water saving, as it is not a direct cost but a saving on the water cooling system.
    for Heater_duty in Process_specs["Heaters"]:
        O_heater += - (Heater_duty / (Cp_water * Water_dT) * Process_param["Operating_hours"] / 998 * Water_cost)
    
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
    TAC_Cryo = 0  # Total Annualised Cost of Cryogenic cooling, if any
    for Cryo in Process_specs["Cryogenics"]:
        Cryo_Cost = math.exp(Cryogenic_cooler_param[0] - Cryogenic_cooler_param[1] * (Cryo[1] - 273.15)) #$(2017) per GJ
        TAC_Cryo += Cryo_Cost * Cryo[0] * Process_param["Operating_hours"] * 0.8865 #convert to cost per year (eur(2017)/yr)
    TAC_Cryo *= Index_2014/Index_2017 #convert to 2014 money

    ### Penalty evaluation for Purity / Recovery ###
    Penalty_purity = 5e11 * (Process_param["Target_Purity"] - Process_specs["Purity"]) if (Process_param["Target_Purity"] - Process_specs["Purity"]) > 0 else 0 #5 billion pounds per year penalty per percentage purity under target
    
    Primary_emission = (1- Process_specs["Recovery"]) * Process_specs["Feed"]["Feed_Composition"][0] * Process_specs["Feed"]["Feed_Flow"] #(mol/s) CO2 emissions from the process
    Primary_emission *= Process_param["Operating_hours"] * 3600 # convert to mol/yr
    Primary_emission *= 44.01 * 1e-6 # convert to tonnes/yr (44.01 g/mol)
    Secondary_Emission = Power_Consumption * Indirect_Emission_rate # (tonnes/yr) CO2 emissions from electricity consumption
    Penalty_CO2_emission = (Primary_emission+Secondary_Emission) * Carbon_Tax #eur/yr 
    
    #Extra_Penalty = 1e10 * (0.90 - Process_specs["Recovery"]) if (0.90 - Process_specs["Recovery"]) > 0 else 0 # Extra penalty for recovery under 90% 

    Penalty = Penalty_purity + Penalty_CO2_emission #+ Extra_Penalty # Total penalty for purity and CO2 emissions

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
        "Total_Plant_Cost": TPC,  # Total Plant Cost
        "Opex_tot": Total_Opex,  # Total Operational Expenditure per year
        "Variable_Opex": Variable_Opex,  # Variable Operational Expenditure per year
        "Fixed_Opex": Fixed_Opex,  # Fixed Operational Expenditure per year
        "TAC_Cryo": TAC_Cryo,  # Total Annualised Cost of Cryogenic cooling
        "Direct_CO2_emission": Primary_emission,  # CO2 emissions from the process in tonnes per year
        "Indirect_CO2_emission": Secondary_Emission,  # CO2 emissions from electricity consumption in tonnes per year"
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
