
import numpy as np
import pickle
import os
from Hub import Hub_Connector
from UNISIMConnect import UNISIMConnector
from scipy.optimize import differential_evolution  
import pandas as pd  
import tqdm
import os
import time
import sys

"""
Slight modification on the unisim python connection. For some reason there has been issues with inconstancy errors in unisim while the simulation is running.

The code has been modified to have the connection between each membrane/pre-conditioning stages directly in python with manual export/import of stream flows.

This may result in slightly longer computation times and potentially small mass balance errors, but should be more robust to consistency errors.

"""

Filename_input = input("Enter the version of the file: Original, Copy, Copy2, or Copy3: ")
if Filename_input.lower() == "original":
    filename = 'Cement_CRMC_3mem.usc'
    directory = 'C:\\Users\\s1854031\\OneDrive - University of Edinburgh\\Python\\Cement_Plant_2021\\'
    results_dir = 'C:\\Users\\s1854031\\OneDrive - University of Edinburgh\\Opti_results_Graveyard' #Directory to save results files
    checkpoint_dir = 'C:\\Users\\s1854031\\OneDrive - University of Edinburgh\\Python\\Checkpoint_Files' #Directory to save checkpoint files
elif Filename_input.lower() == "copy":
    filename = 'Cement_CRMC_3mem_Copy.usc'
    directory = 'C:\\Users\\Simulation Machine\\OneDrive - University of Edinburgh\\Python\\Cement_Plant_2021\\'
    checkpoint_dir = 'C:\\Users\\Simulation Machine\\OneDrive - University of Edinburgh\\Python\\Checkpoint_Files' 
    results_dir = 'C:\\Users\\Simulation Machine\\OneDrive - University of Edinburgh\\Opti_results_Graveyard'
elif Filename_input.lower() == "copy2":
    filename = 'Cement_CRMC_3mem_Copy2.usc'
    directory = 'C:\\Users\\Simulation Machine\\OneDrive - University of Edinburgh\\Python\\Cement_Plant_2021\\'
    checkpoint_dir = 'C:\\Users\\Simulation Machine\\OneDrive - University of Edinburgh\\Python\\Checkpoint_Files' 
    results_dir = 'C:\\Users\\Simulation Machine\\OneDrive - University of Edinburgh\\Opti_results_Graveyard'
elif Filename_input.lower() == "copy3":
    filename = 'Cement_CRMC_3mem_Copy3.usc'
    directory = 'C:\\Users\\s1854031\\OneDrive - University of Edinburgh\\Python\\Cement_Plant_2021\\'
    checkpoint_dir = 'C:\\Users\\s1854031\\OneDrive - University of Edinburgh\\Python\\Checkpoint_Files' 
    results_dir = 'C:\\Users\\s1854031\\OneDrive - University of Edinburgh\\Opti_results_Graveyard'

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

debug_number = 0

unisim_path = os.path.join(directory, filename)

#-------------------------------#
#--- Optimisation Parameters ---#
#-------------------------------#
Method = "Optimisation" # Method is either by Brute_Force or Optimisation or Both
if Method == "Both":
   print(f"Using software path: {filename}; Running both Bruteforce and Optimisation methods")
else:
   print(f"Running the {Method} method for path {filename}")


# Set bounds of optimisation parameters - comment unused parameters
Opti_Param = {
    "Q_A_ratio_1" : [0.5, 20], # Flow/Area ratio for the second stage
    "P_up_1" : [2, 20],  # Upper pressure range for the second stage in bar    
    "Q_A_ratio_2" : [1, 50], # Flow/Area ratio for the first stage"
    "P_up_2" : [2, 20],  # Upper pressure range for the first stage in bar"
    "Q_A_ratio_3" : [2, 75], # Flow/Area ratio for the third stage 
    "P_up_3" : [2, 20],  # Upper pressure range for the third stage in bar
    "Temperature_1" : [-40, 80],  # Temperature range in Celcius
    "Temperature_2" : [-40, 80],  # Temperature range in Celcius
    "Temperature_3" : [-40, 80],  # Temperature range in Celcius
    }


#--------------------------#
#--- Default Parameters ---#
#--------------------------#

Membrane_1 = {
    "Name": 'Membrane_1',
    "Solving_Method": 'CC',                 # 'CC' or 'CO' - CC is for counter-current, CO is for co-current
    "Temperature": 35+273.15,               # Kelvin
    "Pressure_Feed": 5.80,                  # bar
    "Pressure_Permeate": 1,                 # bar
    "Q_A_ratio": 1.92,                      # ratio of the membrane feed flowrate to its area (in m3(stp)/m2.hr)
    "Permeance": [360, 13, 60, 360],        # GPU at 35C  - aged - https://doi.org/10.1016/j.memsci.2017.02.012
    "Pressure_Drop": True,
    }

Membrane_2 = {
    "Name": 'Membrane_2',
    "Solving_Method": 'CC',                   
    "Temperature": 35+273.15,                   
    "Pressure_Feed": 3.34562438356631,                       
    "Pressure_Permeate": 1,                  
    "Q_A_ratio": 8.81807746737551,                          
    "Permeance": [360, 13, 60, 360],        
    "Pressure_Drop": True,
    }


Membrane_3 = {
    "Name": 'Membrane_3',
    "Solving_Method": 'CC',                   
    "Temperature": 35+273.15,                   
    "Pressure_Feed": 3.34562438356631,                       
    "Pressure_Permeate": 1,                  
    "Q_A_ratio": 8.81807746737551,                           
    "Permeance": [360, 13, 60, 360],        
    "Pressure_Drop": True,
    }


Process_param = {
"Recycling_Ratio" : 1,      # Ratio of the retentate flow from Membrane 2 that is recycled back to Membrane 1 feed    
"Target_Purity" : 0.95,     # Target purity of the dry permeate from Membrane 2
"Target_Recovery" : 0.9,    # Target recovery from Membrane 2 - for now not a hard limit, but a target to be achieved
"Replacement_rate": 4,      # Replacement rate of the membranes (in yr)
"Operating_hours": 8000,    # Operating hours per year
"Lifetime": 20,             # Lifetime of the plant (in yr)
"Base_Clinker_Production": 9.65e5, #(tn/yr) 
"Base Plant Cost": 149.8 * 1e6,     # Total direct cost of plant (no CCS) in 2014 money
"Base_Plant_Primary_Emission": (846)*9.65e5 ,# (kgCo2/tn_clk to kgCO2/yr) primary emissions of the base cement plant per year 
"Base_Plant_Secondary_Emission": (34)*9.65e5 ,# (kgCo2/tn_clk to kgCO2/yr) primary emissions of the base cement plant per year 
"Contingency": 0.3,         # or 0.4 (30% or 40% contingency for process design - based on TRL)
}

Component_properties = {
    "Viscosity_param": ([0.0479,0.6112],[0.0466,3.8874],[0.0558,3.8970], [0.03333, -0.23498]),  # Viscosity parameters for each component: slope and intercept for the viscosity correlation wiht temperature (in K) - from NIST
    "Molar_mass": [44.009, 28.0134, 31.999,18.01528],                                           # Molar mass of each component in g/mol"
    "Activation_Energy_Aged": ([12750,321019],[25310,2186946],[15770,196980],[12750,321019]),   # ([Activation energy - J/mol],[pre-exponential factor - GPU])
    "Activation_Energy_Fresh": ([2880,16806],[16520,226481],[3770,3599],[2880,16806]),          #Valid only if temperature is -20 C or under - not considered for now
    }

Fibre_Dimensions = {
"D_in" : 150 * 1e-6,    # Inner diameter in m (from um)
"D_out" : 300 * 1e-6,   # Outer diameter in m (from um)
}
  
J = len(Membrane_1["Permeance"]) #number of components


with UNISIMConnector(unisim_path, close_on_completion=False) as unisim:
    
    unisim.wait_solution(timeout=10, check_pop_ups=2, check_consistency_error=3)

    class ConvergenceError(Exception): #allowing to skip iteration if convergence error appears in one of the mass balance
        pass

    def CMRC_3_Main(Param):
    
        Membrane_1 , Membrane_2, Membrane_3, Process_param, Component_properties, Fibre_Dimensions, J =  Param

        Feed = {}
        Flue_Inlet = unisim.get_spreadsheet('Flue_Inlet_2')
        Feed['Feed_Flow'] = Flue_Inlet.get_cell_value('C3') / 3.6 # feed flow rate from UNISIM in mol/s (from kmol/h)
        Feed["Feed_Composition"] = [Flue_Inlet.get_cell_value(f'C{i+4}') for i in range(J)] #feed mole fractions from UNISIM

        #----------------------------------------#
        #----------- Get spreadsheets -----------#
        #----------------------------------------#

        Membrane1 = unisim.get_spreadsheet('Membrane 1')
        Membrane2 = unisim.get_spreadsheet('Membrane 2')
        Membrane3 = unisim.get_spreadsheet('Membrane 3')
        Duties = unisim.get_spreadsheet('Duties')
        Mem_Inlet_1 = unisim.get_spreadsheet('Flue_Inlet_1')
        Mem_Inlet_2 = unisim.get_spreadsheet('Flue_Inlet_2')
        Mem_Inlet_3 = unisim.get_spreadsheet('Flue_Inlet_3')
        Train_inlet = unisim.get_spreadsheet('Compression_Train')


        #-----------------------------------------#
        #------------- Initial setup -------------#
        #-----------------------------------------#

        # Reset streams
        for i in range(J):
            Membrane1.set_cell_value(f'D{i+14}', 0) # Reset Membrane 1 permeate component flows
            Membrane1.set_cell_value(f'D{i+21}', 0) # Reset Membrane 1 retentate component flows
            Membrane2.set_cell_value(f'D{i+14}', 0) # Reset Membrane 2 permeate component flows
            Membrane2.set_cell_value(f'D{i+21}', 0) # Reset Membrane 2 retentate component flows
            Membrane3.set_cell_value(f'D{i+14}', 0)
            Membrane3.set_cell_value(f'D{i+21}', 0)
        unisim.wait_solution(timeout=10, check_pop_ups=2, check_consistency_error=3)

        # Setup temperatures and pressures
        def to_bar(p):
            """Ensure pressure is always sent in bar"""
            return p * 1e-5 if p > 100 else p

        # Setup temperatures and pressures
        Membrane1.set_cell_value('D3', Membrane_1["Temperature"])  
        Membrane1.set_cell_value('D4', to_bar(Membrane_1["Pressure_Feed"]))  
        Membrane1.set_cell_value('D6', to_bar(Membrane_1["Pressure_Permeate"]))  

        Membrane2.set_cell_value('D3', Membrane_2["Temperature"]) 
        Membrane2.set_cell_value('D4', to_bar(Membrane_2["Pressure_Feed"])) 
        Membrane2.set_cell_value('D6', to_bar(Membrane_2["Pressure_Permeate"]))  

        Membrane3.set_cell_value('D3', Membrane_3["Temperature"]) 
        Membrane3.set_cell_value('D4', to_bar(Membrane_3["Pressure_Feed"])) 
        Membrane3.set_cell_value('D6', to_bar(Membrane_3["Pressure_Permeate"]))  

        unisim.wait_solution(timeout=10, check_pop_ups=5, check_consistency_error=5)

        #------------------------------------------------#
        #--------- Get correct feed from Unisim ---------#
        #------------------------------------------------#

        #Function to select the correct membrane compression train - Determined to be the one with the lowest non null compressor duty
    
        def Mem_Train_Choice(Membrane):
            
            unisim.wait_solution(timeout=10, check_pop_ups=2, check_consistency_error=3)
            Train_data = []
            for i in range(3):  # three potential trains for each membrane
                if Membrane == Membrane_1:
                    Train_data.append([  
                        Duties.get_cell_value(f'H{i+9}'),  # Compressor Duty (kW)
                        Duties.get_cell_value(f'I{i+9}'),  # Hex Area (m2)
                        Duties.get_cell_value(f'J{i+9}'),  # Water Flowrate (kg/hr)
                        Duties.get_cell_value(f'K{i+9}')/1e6 if Duties.get_cell_value(f'K{i+9}') is not None and Duties.get_cell_value(f'K{i+9}') > 0 else 0   # Cryogenic Cooler Duty (GJ/hr)
                    ])
                elif Membrane == Membrane_2:
                    Train_data.append([  
                        Duties.get_cell_value(f'H{i+15}'),  # Compressor Duty (kW)
                        Duties.get_cell_value(f'I{i+15}'),  # Hex Area (m2)
                        Duties.get_cell_value(f'J{i+15}'),   # Water Flowrate (kg/hr)
                        Duties.get_cell_value(f'K{i+15}')/1e6  if Duties.get_cell_value(f'K{i+15}') is not None and Duties.get_cell_value(f'K{i+15}') > 0 else 0 # Cryogenic Cooler Duty (GJ/hr)
                    ])
                elif Membrane == Membrane_3:
                    Train_data.append([     
                        Duties.get_cell_value(f'H{i+21}'),  # Compressor Duty (kW)
                        Duties.get_cell_value(f'I{i+21}'),  # Hex Area (m2)
                        Duties.get_cell_value(f'J{i+21}'),  # Water Flowrate (kg/hr)
                        Duties.get_cell_value(f'K{i+21}')/1e6 if Duties.get_cell_value(f'K{i+21}') is not None and Duties.get_cell_value(f'K{i+21}') > 0 else 0  # Cryogenic Cooler Duty (GJ/hr)
                    ])

                else: raise ValueError ("Incorrect membrane denomination")

            # Filter out trains with None or non-positive compressor duty

            valid_train_indices = [i for i, train in enumerate(Train_data) if train[0] is not None and train[0] > 0 and train[1] is not None]
            if not valid_train_indices and Membrane_2["Pressure_Feed"] < Membrane_1["Pressure_Feed"]:
                print(valid_train_indices)
                raise ValueError(f"No valid trains found with positive compressor duty in {Membrane["Name"]}.")
        
            # Find the index of the train with the lowest compressor duty in Train_data
            if valid_train_indices:
                lowest_duty_train_index = min(valid_train_indices, key=lambda i: Train_data[i][0])

            # Read the spreadsheet of the corresponding train          
            if Membrane == Membrane_1:
                if Membrane_2["Pressure_Feed"] >= Membrane_1["Pressure_Feed"]: # If membrane 2 retentate pressure is higher than mem 1 feed pressure, the stream needs to be the one with the expander
                    Mem_train = unisim.get_spreadsheet(f'Train 104')
                else:
                    Mem_train = unisim.get_spreadsheet(f'Train 10{lowest_duty_train_index + 1}')  

                Membrane_1['Feed_Flow'] = Mem_train.get_cell_value('C3') / 3.6 # feed flow rate from UNISIM in mol/s (from kmol/h)
                Membrane_1["Feed_Composition"] = [Mem_train.get_cell_value(f'C{i+4}') for i in range(J)] #feed mole fractions from UNISIM

            elif Membrane == Membrane_2:
                Mem_train = unisim.get_spreadsheet(f'Train 20{lowest_duty_train_index + 1}')
                Membrane_2['Feed_Flow'] = Mem_train.get_cell_value('C3') / 3.6 # feed flow rate from UNISIM in mol/s (from kmol/h)
                Membrane_2["Feed_Composition"] = [Mem_train.get_cell_value(f'C{i+4}') for i in range(J)] #feed mole fractions from UNISIM

            elif Membrane == Membrane_3:            
                Mem_train = unisim.get_spreadsheet(f'Train 30{lowest_duty_train_index + 1}') # Get the train with the expander
                Membrane_3['Feed_Flow'] = Mem_train.get_cell_value('C3') / 3.6 # feed flow rate from UNISIM in mol/s (from kmol/h)
                Membrane_3["Feed_Composition"] = [Mem_train.get_cell_value(f'C{i+4}') for i in range(J)] #feed mole fractions from UNISIM


        #------------------------------------------#
        #--------- Function to run module ---------#
        #------------------------------------------#
        #------------------------------------------#

        def Run(Membrane):

            unisim.wait_solution(timeout=10, check_pop_ups=5, check_consistency_error=5)


            # Set membrane Area based on its feed flow and Q_A_ratio:
            Membrane["Area"] = (Membrane["Feed_Flow"] * 0.0224  * 3600) / Membrane["Q_A_ratio"] # (0.0224 is the molar volume of an ideal gas at STP in m3/mol)
            #print(f'{Membrane["Name"]} Area: {Membrane["Area"]:.0f}')

            Membrane["Sweep_Flow"] = 0 # No sweep in this configuration
            Membrane["Sweep_Composition"] = [0] * len(Membrane_1["Permeance"])

            Export_to_mass_balance = Membrane, Component_properties, Fibre_Dimensions

            J = len(Membrane["Permeance"]) #number of components

            # Obtain Permeance with temperature:
            for i in range(J):
                Membrane["Permeance"][i] = Component_properties["Activation_Energy_Aged"][i][1] * np.exp(-Component_properties["Activation_Energy_Aged"][i][0] / (8.314 * Membrane["Temperature"]))

            # No sweep in any of the membranes in this configuration
   
            results, profile = Hub_Connector(Export_to_mass_balance)
            Membrane["Retentate_Composition"],Membrane["Permeate_Composition"],Membrane["Retentate_Flow"],Membrane["Permeate_Flow"] = results

            #print(f"Overall mass balance error of membrane {Membrane["Name"]}: Feed + Sweep  - Retentate - Permeate = {abs(Membrane["Feed_Flow"] + Membrane["Sweep_Flow"] - Membrane["Retentate_Flow"] - Membrane["Permeate_Flow"]):.3e}")
    
            if np.any(profile<-1e-5):
                raise ConvergenceError  # Return a scalar and an empty dictionary as a placeholder

            #Reformat Permeance and Pressure values to the initial units
            Membrane["Permeance"] = [p / ( 3.348 * 1e-10 ) for p in Membrane["Permeance"]]  # convert from mol/m2.s.Pa to GPU
            Membrane["Pressure_Feed"] *= 1e-5  #convert to bar
            Membrane["Pressure_Permeate"] *= 1e-5  

            #print(profile)
            return results, profile 



        ### Run iterations for process recycling loop - Specific to this configuration!

        max_iter = 150
        tolerance = 5e-5

        Placeholder_2={ #Intermediade data storage for the recycling loop entering the first membrane used to check for convergence
            "Feed_Composition": [0] * J,
            "Feed_Flow": 0,              
            } 

        for j in range(max_iter):

            #print(f"Iteration {j+1}")

            for i in range(J+3): #Obtain stream from unisim for membrane 2 and put it through pre-conditioning (buffer added to avoid convergence issues)
                Mem_Inlet_2.set_cell_value(f'C{i+10}', Mem_Inlet_2.get_cell_value(f'B{i+10}'))
            Mem_Train_Choice(Membrane_2) #Obtain stream after pre-conditioning

            try: #Run the second membrane
                results_2 , profile_2 = Run(Membrane_2)
            except ConvergenceError:
                return 5e8
            for i in range(J): #results "[0]: x", "[1]: y", "[2]: Q_ret", "[3]: Q_perm"
                Membrane2.set_cell_value(f'D{i+14}', results_2[1][i] * results_2[3] * 3.6) # convert from mol/s to kmol/h
                Membrane2.set_cell_value(f'D{i+21}', results_2[0][i] * results_2[2] * 3.6)
            unisim.wait_solution(timeout=10, check_pop_ups=5, check_consistency_error=5)

            for i in range(J+3): #Obtain stream from unisim for membrane 1 and put it through pre-conditioning (buffer added to avoid convergence issues)
                Mem_Inlet_1.set_cell_value(f'C{i+3}', Mem_Inlet_1.get_cell_value(f'B{i+3}'))
            Mem_Train_Choice(Membrane_1)

            try: #Run the first membrane
                results_1 , profile_1 = Run(Membrane_1)
            except ConvergenceError:
                return 5e8
            for i in range(J): #results "[0]: x", "[1]: y", "[2]: Q_ret", "[3]: Q_perm"
                Membrane1.set_cell_value(f'D{i+14}', results_1[1][i] * results_1[3] * 3.6) # convert from mol/s to kmol/h
                Membrane1.set_cell_value(f'D{i+21}', results_1[0][i] * results_1[2] * 3.6)
            unisim.wait_solution(timeout=10, check_pop_ups=5, check_consistency_error=5)

            for i in range(J+3): #Obtain stream from unisim for membrane 3 and put it through pre-conditioning (buffer added to avoid convergence issues)
                Mem_Inlet_3.set_cell_value(f'C{i+3}', Mem_Inlet_3.get_cell_value(f'B{i+3}'))
            Mem_Train_Choice(Membrane_3)
            
            try: #Run the third membrane
                results_3 , profile_3 = Run(Membrane_3)
            except ConvergenceError:
                return 5e8
            for i in range(J): #results "[0]: x", "[1]: y", "[2]: Q_ret", "[3]: Q_perm"
                Membrane3.set_cell_value(f'D{i+14}', results_3[1][i] * results_3[3] * 3.6) # convert from mol/s to kmol/h
                Membrane3.set_cell_value(f'D{i+21}', results_3[0][i] * results_3[2] * 3.6)
            unisim.wait_solution(timeout=10, check_pop_ups=5, check_consistency_error=5)

            Convergence_Composition = sum(abs(np.array(Placeholder_2["Feed_Composition"]) - np.array(Membrane_2["Feed_Composition"])))
            Convergence_Flowrate = abs( ( (Placeholder_2["Feed_Flow"]) - (Membrane_2["Feed_Flow"] ) ) / (Membrane_2["Feed_Flow"] ) / 100 )
            #print(f'Convergence Composition: {Convergence_Composition:.3e}, Convergence Flowrate: {Convergence_Flowrate*100:.3e}')    
            
            CO2_in = Feed["Feed_Composition"][0]*Feed["Feed_Flow"]
            CO2_out = results_3[1][0]*results_3[3] + results_1[0][0]*results_1[2] # out = permeate of mem 3 + retentate of mem 1
            mass_balance_error_CO2 = abs(CO2_in - CO2_out)/CO2_in

            if mass_balance_error_CO2 > 1:
                print(f"Warning: CO2 mass balance error is especially high - skipping")
                return 5e7

            #check for convergence
            if j > 0 and Convergence_Composition < tolerance and Convergence_Flowrate < tolerance:  

                ''' co2 mass balance check - should be very close to 0 '''

                if mass_balance_error_CO2 > 5e-3:
                    print(f"Warning: CO2 mass balance error is {mass_balance_error_CO2:.3%}")
                    return 5e7 


                for i in range(J+3):
                    Train_inlet.set_cell_value(f'C{i+3}', Mem_Inlet_2.get_cell_value(f'C{i+3}')) # Final update of the stream going into the compression train fo raccurate costing
                
                break

            #Ready for next iteration
            Placeholder_2["Feed_Composition"] = Membrane_2["Feed_Composition"]
            Placeholder_2["Feed_Flow"] = Membrane_2["Feed_Flow"]

    

        else: 
            print("Max iterations for recycling reached")
            return 2e9

                                        

            #-------------------------------------------#
            #----------- Export/Import UniSim ----------#
            #-------------------------------------------#

        def Duty_Gather():  # Gather Duties of the trains from the solved process
            def get_lowest_duty_train(train_data):
                # Filter out trains with None or non-positive compressor duty
                valid_trains = [i for i, train in enumerate(train_data) if train[0] is not None and train[0] > 0 and train[1] is not None]
                if not valid_trains:
                    raise ValueError(f"No valid trains found with positive compressor duty for train {i+1}.")
            
                # Find the train with the lowest compressor duty
                lowest_duty_train_index = min(valid_trains, key=lambda i: train_data[i][0])
                lowest_duty_train = train_data[lowest_duty_train_index]
                lowest_duty_train.append(lowest_duty_train_index) #append the index of the train with the lowest duty to know the equipment count
                return lowest_duty_train

            def gather_train_data(start_row):
                return [
                    [
                        Duties.get_cell_value(f'H{i+start_row}'),  # Compressor Duty (kW)
                        Duties.get_cell_value(f'I{i+start_row}'),  # Hex Area (m2)
                        Duties.get_cell_value(f'J{i+start_row}'),  # Water Flowrate (kg/hr)
                        Duties.get_cell_value(f'K{i+start_row}') / 1e6 
                        if Duties.get_cell_value(f'K{i+start_row}') is not None and Duties.get_cell_value(f'K{i+start_row}') > 0 
                        else 0  # Cryogenic Cooler Duty (MJ/hr)
                    ]
                    for i in range(3)
                ]

            Train1 = gather_train_data(9)
            Train2 = gather_train_data(15)
            Train3 = gather_train_data(21)
            Liquefaction = gather_train_data(3)

            # Get the train with the lowest compressor duty for each category

            if to_bar(Membrane_2["Pressure_Feed"]) < to_bar(Membrane_1["Pressure_Feed"]):
                Train1_lowest = [0]*5 #no duties needed as there is an expander instead of the compression train
            else: Train1_lowest = get_lowest_duty_train(Train1)
            Train2_lowest = get_lowest_duty_train(Train2)
            Train3_lowest = get_lowest_duty_train(Train3)
            Liquefaction_lowest = get_lowest_duty_train(Liquefaction)

            return Train1_lowest, Train2_lowest, Train3_lowest, Liquefaction_lowest


        Train1, Train2, Train3, Liquefaction = Duty_Gather() # Gather the duties from the solved process

        # Gather the energy recovery form the retentate. Assume flue gas at 1 bar and a maximum temperature of 120 C to match original flue gas.
        Expanders = (Duties.get_cell_value('H27'), Duties.get_cell_value('H33'), Duties.get_cell_value('H30') ) # Get the retentate expanders duties (kW)    
        Heaters = (Duties.get_cell_value('I27'), Duties.get_cell_value('I30')) # Get the retentate heaters duties (kJ/hr)

        # Gather the cryogenic cooler duties - if any
        Cryogenics = ( (Train1[3], Membrane_1["Temperature"]) , (Train2[3], Membrane_2["Temperature"]), (Train3[3], Membrane_3["Temperature"]) ) # Get the cryogenic cooler duties (MJ/hr) for each membrane train

        if to_bar(Membrane_2["Pressure_Feed"]) < to_bar(Membrane_1["Pressure_Feed"]): #if expansion not needed, need to remove expander and heater for mem 1 pre conditioning
            Expanders = Expanders[:-1]
            Heaters = Heaters[:-1]
            Cryogenics = Cryogenics[1:]

        # Add information to the compression trains about their number of compressors and coolers
        if not to_bar(Membrane_2["Pressure_Feed"]) < to_bar(Membrane_1["Pressure_Feed"]):
            Train1.append(Train1[4]+1) # Append the number of compressors in the train
            Train1.append(Train1[4]+2)  # Extra heat exchanger for retentate heat recovery

        Train2.append(Train2[4]+1)
        Train2.append(Train2[4]+2)  # Extra heat exchanger for feed cooling

        Train3.append(Train2[4]+1)
        Train3.append(Train2[4]+1)

        Liquefaction.append(Liquefaction[4]+3)  # Append the number of compressors and heat exchangers in the liquefaction train
        Liquefaction.append(Liquefaction[4]+3)

        Process_specs = {
            "Feed": Feed,
            "Purity": (results_3[1][0]/(1-results_3[1][-1])),
            "Recovery": results_3[1][0]*results_3[3]/(Feed["Feed_Composition"][0]*Feed["Feed_Flow"]),
            "Compressor_trains": ( (Train1[0], Train1[-2]), (Train2[0],Train2[-2]), (Liquefaction[0], Liquefaction[-2]) ),  # Compressor trains data
            "Cooler_trains": ( (Train1[1], Train2[2], Train1[-1]), (Train2[1],Train2[2], Train2[-1]), (Liquefaction[1], Liquefaction[2], Liquefaction[-1]) ),  # Cooler trains data"
            "Membranes": (Membrane_1, Membrane_2),
            "Expanders": Expanders,  # Expander data
            "Heaters": Heaters,  # Heater data
            "Cryogenics": Cryogenics,
        }

        #print(Process_specs)

        if to_bar(Membrane_2["Pressure_Feed"]) < to_bar(Membrane_1["Pressure_Feed"]): #if not compression train for membrane 1 needed, we need to remove its duties
            Process_specs["Compressor_trains"] = Process_specs["Compressor_trains"][1:]
            Process_specs["Cooler_trains"] = Process_specs["Cooler_trains"][1:]

        from Costing import Costing
        Economics = Costing(Process_specs, Process_param)
        
        return Economics

    #-------------------------------#
    #----- Optimisation Method -----#
    #-------------------------------#


    def Opti_algorithm():
        
        checkpoint_file = "de_checkpoint_laptop_9param_091025.pkl" # Checkpoint file name
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file) # Use the savepoints directory

        popsize = 20  # Population size multiplier
        
                # ----------------- Load checkpoint or midpoint guess -----------------
        def load_checkpoint():
            """Ask user whether to reload from checkpoint, otherwise use midpoint guess."""

            if os.path.exists(checkpoint_path):
                choice = input("A checkpoint was found. Do you want to resume? (y/n): ").strip().lower()
                if choice == "y":
                    with open(checkpoint_path, "rb") as f:
                        checkpoint_data = pickle.load(f)
                    if len(checkpoint_data["best_solution"]) == len(bounds):
                        print(f"Resuming from iteration {checkpoint_data['iteration']} with best solution {checkpoint_data["best_solution"]}")
                        x0 = checkpoint_data["best_solution"]

                        # --- bounds-scaled Gaussian population around checkpoint ---
                        widths = np.array([b[1] - b[0] for b in bounds], float)
                        sigma  = 0.20 * widths
                        N, D   = popsize * len(bounds), len(bounds)
                        init_pop = x0 + np.random.normal(0.0, sigma, size=(N, D))
                        init_pop = np.clip(init_pop, [b[0] for b in bounds], [b[1] for b in bounds])
                        init_pop[0] = x0
                        return init_pop
                    else:
                        print("Checkpoint incompatible with current problem, starting fresh.")

            # --- No checkpoint: ask user for initialisation method ---
            
            user_choice = input("Use initial guess? (y for initial guess, n for random): ").strip().lower()
            use_initial_guess = (user_choice == "y")

            if use_initial_guess:
                # Use custom guess
                first_guess = np.array([9.11362129, 13.40702624, 10, 3.71798611, 4.59653773, 3, -9.26197132, -34.23333671, 15])
                print(f"Starting with guess: {first_guess}")

                widths = np.array([b[1] - b[0] for b in bounds], float)
                sigma  = 0.20 * widths
                N, D   = popsize * len(bounds), len(bounds)
                init_pop = first_guess + np.random.normal(0.0, sigma, size=(N, D))
                init_pop = np.clip(init_pop, [b[0] for b in bounds], [b[1] for b in bounds])
                init_pop[0] = first_guess
                return np.array(init_pop)
            else:
                # Use random initialization
                print("Starting with random initialization.")
                N, D = popsize * len(bounds), len(bounds)
                init_pop = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(N, D))
                return np.array(init_pop)


        # Define the bounds for the parameters to be optimised
        bounds = [  
            Opti_Param["Q_A_ratio_1"],  # Flow/Area ratio for the second stage
            Opti_Param["P_up_1"],  # Upper pressure range for the second stage in bar    
            Opti_Param["Q_A_ratio_2"],  # Flow/Area ratio for the first stage
            Opti_Param["P_up_2"],  # Upper pressure range for the first stage in bar
            Opti_Param["Q_A_ratio_3"],  # Flow/Area ratio for the third stage
            Opti_Param["P_up_3"],  # Upper pressure range for the third stage in bar
            Opti_Param["Temperature_1"],  # Temperature range in Celcius
            Opti_Param["Temperature_2"],  # Temperature range in Celcius   
            Opti_Param["Temperature_3"],  # Temperature range in Celcius
        ]
    
        # Define the objective function to be minimised
        def objective_function(params):
            Membrane_1["Q_A_ratio"] = params[0]
            Membrane_1["Pressure_Feed"] = params[1]
            Membrane_2["Q_A_ratio"] = params[2]
            Membrane_2["Pressure_Feed"] = params[3]
            Membrane_3["Q_A_ratio"] = params[4]
            Membrane_3["Pressure_Feed"] = params[5]
            Membrane_1["Temperature"] = params[6] + 273.15
            Membrane_2["Temperature"] = params[7] + 273.15
            Membrane_3["Temperature"] = params[8] + 273.15
            Parameters = Membrane_1, Membrane_2,Membrane_3, Process_param, Component_properties, Fibre_Dimensions, J
            Economics = CMRC_3_Main(Parameters)
            
            global debug_number #debug to make sure te sript is running
            debug_number += 1
            if debug_number == 10:
                print("debug: simulation has compiled 10 sets of parameters")
                debug_number = 0

            if isinstance(Economics, dict): #checks that the function return is valid - this is used to eliminate bugs or unsuccessful simulation 
                if Economics["Recovery"]>1 or Economics["Purity"]>1 or isinstance(Economics["Evaluation"], complex):
                    return 1e10
                else:
                    return Economics['Evaluation']
            else:
                return 1e10  # Large penalty if simulation fails

        # Callback function to track progress
        def callback(xk, convergence):
            elapsed_time = time.time() - callback.start_time
            print(f"Iteration: {callback.n_iter}, Elapsed Time: {elapsed_time/3600:.2f} hr, "
                  f"Best Solution: {xk}, Convergence: {convergence:.4f}")
            print()

            # Save checkpoint
            checkpoint_data = {
                "iteration": callback.n_iter,
                "best_solution": xk,
                "convergence": convergence
            }
            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint_data, f)

            callback.n_iter += 1
            return False  # return True to stop optimization

        callback.n_iter = 1
        callback.start_time = time.time()

        # Run the optimisation algorithm
        init_setting = load_checkpoint()

        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=500,  
            popsize=20,  
            tol=5e-3,
            callback=callback,
            polish=True,
            init=init_setting
        )


        def Solved_process():
            Membrane_1["Q_A_ratio"] = result.x[0]
            Membrane_1["Pressure_Feed"] = result.x[1]
            Membrane_2["Q_A_ratio"] = result.x[2]
            Membrane_2["Pressure_Feed"] = result.x[3]
            Membrane_3["Q_A_ratio"] = result.x[4]
            Membrane_3["Pressure_Feed"] = result.x[5]
            Membrane_1["Temperature"] = result.x[6] + 273.15
            Membrane_2["Temperature"] = result.x[7] + 273.15
            Membrane_3["Temperature"] = result.x[8] + 273.15
            Parameters = Membrane_1, Membrane_2, Membrane_3, Process_param, Component_properties, Fibre_Dimensions, J
            Economics = CMRC_3_Main(Parameters)
            if isinstance(Economics, dict):
                return Economics
            else: 
                return "Simulation failed to converge"

        Economics = Solved_process()

        print("Optimisation Result:")
        print("Optimal Parameters: ", [f"{res:.5f}" for res in result.x])
        print(f"Objective Function Value: {result.fun:.5e}")
        print(Economics)
        def save_results(results, Economics, filepath):
            with open(filepath, 'w') as f:
                f.write("Best Parameters: {}\n".format(results.x))
                f.write("Objective Value: {}\n".format(results.fun))
                f.write("Iterations: {}\n".format(results.nit))
                f.write("Convergence Message: {}\n".format(results.message))
                f.write("Economics: {}\n".format(Economics))

        def format_economics(Economics):
            return "\n".join(f"{key}: {value}" for key, value in Economics.items())

        # Save the results
        filename = 'CRMC3_laptop_7param_opti.txt'
        economics_str = format_economics(Economics)
        res_filepath = os.path.join(results_dir, filename)
        save_results(result, economics_str, res_filepath)

        print("Results saved to", filename)




    #------------------------------#
    #----- Brute Force Method -----#
    #------------------------------#


    def Brute_Force():

        
        # In case of brute force
        number_evaluation = 1000 * len(Opti_Param)  # Number of evaluations for the brute force method

        Brute_Force_Param = [  
            {  
                "Q_A_ratio_1": np.random.uniform(*Opti_Param["Q_A_ratio_2"]),  
                "P_up_1": np.random.uniform(*Opti_Param["P_up_2"]),
                "Q_A_ratio_2": np.random.uniform(*Opti_Param["Q_A_ratio_1"]),
                "P_up_2": np.random.uniform(*Opti_Param["P_up_1"]),
                "Q_A_ratio_3": np.random.uniform(*Opti_Param["Q_A_ratio_3"]),
                "P_up_3": np.random.uniform(*Opti_Param["P_up_3"]),
                "Temperature_1": np.random.uniform(*Opti_Param["Temperature_1"]),
                "Temperature_2": np.random.uniform(*Opti_Param["Temperature_2"]),
                'Temperature_3': np.random.uniform(*Opti_Param["Temperature_3"]),
            }  
            for _ in range(number_evaluation)  
        ]



        # Define the columns based on the parameters being changed and all variables in the Economics dictionary  
        columns = [  
          f'Param_{j+1}' for j in range(len(Brute_Force_Param[0]))
        ] + [  
          "Evaluation",  # Evaluation metric  
          "Purity",  # Purity of the product  
          "Recovery",  # Recovery of the product  
          "TAC_CC",  # Total Annualised Cost of Carbon Capture  
          "Capex_tot",  # Total Capital Expenditure in 2014 money
          "Total_Plant_Cost",  # Total Plant Cost
          "Opex_tot",  # Total Operational Expenditure per year  
          "Variable_Opex",  # Variable Operational Expenditure per year  
          "Fixed_Opex",  # Fixed Operational Expenditure per year  
          "TAC_Cryo",  # Total Annualised Cost of Cryogenic cooling  
          "Direct_CO2_emission",  # CO2 emissions from the process in tonnes per year
          "Indirect_CO2_emission",  # CO2 emissions from electricity consumption in tonnes per year"
          "C_compressor",  # Capital cost of compressors  
          "C_cooler",  # Capital cost of coolers  
          "C_membrane",  # Capital cost of membranes  
          "C_expander",  # Capital cost of expanders  
          "O_compressor",  # Operational cost of compressors per year  
          "O_cooler",  # Operational cost of coolers per year  
          "O_membrane",  # Operational cost of membranes per year  
          "O_expander",  # Operational cost of expanders per year  
          "O_heater",  # Operational cost of heat integration from retentate per year  
          "Penalty_purity",  # Penalty for purity under target  
          "Penalty_CO2_emission",  # Penalty for CO2 emissions under target  
        ]
        # Preallocate for the range of data sets
        rows = []
        Invalid = 0 #Count of invalid datasets due to simulation not converging.

        for i in tqdm.tqdm(range(number_evaluation)):
            Membrane_1["Q_A_ratio"] = Brute_Force_Param[i]["Q_A_ratio_2"]
            Membrane_1["Pressure_Feed"] = Brute_Force_Param[i]["P_up_2"]
            Membrane_2["Q_A_ratio"] = Brute_Force_Param[i]["Q_A_ratio_1"]
            Membrane_2["Pressure_Feed"] = Brute_Force_Param[i]["P_up_1"]
            Membrane_3["Q_A_ratio"] = Brute_Force_Param[i]["Q_A_ratio_3"]
            Membrane_3["Pressure_Feed"] = Brute_Force_Param[i]["P_up_3"]
            Membrane_1["Temperature"] = Brute_Force_Param[i]["Temperature_1"] + 273.15
            Membrane_2["Temperature"] = Brute_Force_Param[i]["Temperature_2"] + 273.15
            Membrane_3["Temperature"] = Brute_Force_Param[i]["Temperature_3"] + 273.15

            Parameters = Membrane_1, Membrane_2,Membrane_3, Process_param, Component_properties, Fibre_Dimensions, J

            # Run the optimisation algorithm
            Economics = CMRC_3_Main(Parameters)

            if isinstance(Economics, dict):
                row = list(Brute_Force_Param[i].values()) + list(Economics.values())
                rows.append(row)
            else: Invalid += 1

        # Convert the list of rows to a DataFrame
        df = pd.DataFrame(rows, columns=columns)

        filename_bruteforce = "CRMC3_laptop_7param_bruteforce.csv"

        file_path = os.path.join(results_dir, filename_bruteforce)
        df.to_csv(file_path, index=False)

        # Save the DataFrame to a CSV file on the desktop
        df.to_csv(file_path, index=False)


        print(f"{number_evaluation - Invalid} sets of data collected in {file_path} over {number_evaluation} evaluations total ({(number_evaluation - Invalid)/number_evaluation:.2%}). ")


    
    if Method == "Brute_Force":
        Brute_Force()
    elif Method == "Optimisation":
        Opti_algorithm()
    elif Method == "Both":
        Brute_Force()
        Opti_algorithm()
    else: raise ValueError ("Incorrect method chosen - it should be either Brute_Force or Optimisation")
    
    print("Done - probably")





   
