
import numpy as np
import os
from PIL import Image
from Hub import Hub_Connector
from UNISIMConnect import UNISIMConnector
import math  
from scipy.optimize import differential_evolution  
import pandas as pd  
import tqdm
import os
import time
import sys
"""
This module is used to run the optimisation for the process simulation, would it be by brute force or using an optimisation algorithm.

A set of parameters from the process are chosen and given bounds for the optimisation algorithm to work with.

The aim is to minimise a scaler from the process simulation function, which usually will be associated with the cost of the process.

The brute force method can be used for up to three parameters and will be mainly used to validate the optimisation algorithm.

Most debugging and test messages are removed from this solution ; manual checks can be done on another project.

"""

if len(sys.argv) > 1:
    mode = sys.argv[1]  # First argument after script name
    if mode == "laptop":
        filename = 'Cement_4Comp_FerrariPaper_Flash.usc'
        directory = 'C:\\Users\\s1854031\\OneDrive - University of Edinburgh\\Python\\Cement_Plant_2021\\' #laptop
    elif mode == "desktop":
        filename = 'Cement_4Comp_FerrariPaper_Flash_Copy.usc'
        directory = 'C:\\Users\\Simulation Machine\\OneDrive - University of Edinburgh\\Python\\Cement_Plant_2021\\' #desktop.
    else:
        raise ValueError(f"Unknown mode: {mode}")

else:    
    input_filename = input("Filename: Original or Copy or Copy2?") # For manual testing, this will allow to choose the file to use.")
    if input_filename == "Original":
        filename = 'Cement_4Comp_FerrariPaper_Flash.usc'
        directory = 'C:\\Users\\s1854031\\OneDrive - University of Edinburgh\\Python\\Cement_Plant_2021\\' #idefault path on laptop.

    elif input_filename == "Copy":
        filename = 'Cement_4Comp_FerrariPaper_Flash_Copy.usc'
        directory = 'C:\\Users\\Simulation Machine\\OneDrive - University of Edinburgh\\Python\\Cement_Plant_2021\\' #default path on desktop.
    elif input_filename == "Copy2":
        filename = 'Cement_4Comp_FerrariPaper_Flash_Copy2.usc'
        directory = 'C:\\Users\\Simulation Machine\\OneDrive - University of Edinburgh\\Python\\Cement_Plant_2021\\'

#File paths:

unisim_path = os.path.join(directory, filename)
#-------------------------------#
#--- Optimisation Parameters ---#
#-------------------------------#
Method = "Optimisation" # Method is either by Brute_Force or Optimisation or Both
if Method == "Both":
   print(f"Using software path: {filename}; Running both Bruteforce and Optimisation methods")
else:
   print(f"Running the {Method} method for path {filename}")

# Generate brute force parameters
Opti_Param = {
    "Recycling_Ratio" : [0, 1],  # Recycling ratio range for the process
    "Q_A_ratio_1" : [0.5, 20], # Flow/Area ratio for the second stage
    "P_up_1" : [2, 20],  # Upper pressure range for the second stage in bar    
    "Q_A_ratio_2" : [1, 50], # Flow/Area ratio for the first stage"
    "P_up_2" : [2, 20],  # Upper pressure range for the first stage in bar"
    #"Temperature_1" : [-40, 70],  # Temperature range in Celcius
    #"Temperature_2" : [-40, 70],  # Temperature range in Celcius
    }


# In case of brute force
number_evaluation = 1000 * len(Opti_Param)  # Number of evaluations for the brute force method

Brute_Force_Param = [  
    {  
        "Recycling_Ratio": np.random.uniform(*Opti_Param["Recycling_Ratio"]),  
        "Q_A_ratio_1": np.random.uniform(*Opti_Param["Q_A_ratio_1"]),  
        "P_up_1": np.random.uniform(*Opti_Param["P_up_1"]),
        "Q_A_ratio_2": np.random.uniform(*Opti_Param["Q_A_ratio_2"]),
        "P_up_2": np.random.uniform(*Opti_Param["P_up_2"]),
        #"Temperature_1": np.random.uniform(*Opti_Param["Temperature_1"]),
        #"Temperature_2": np.random.uniform(*Opti_Param["Temperature_2"]),
    }  
    for _ in range(number_evaluation)  
]

#--------------------------#
#--- Default Parameters ---#
#--------------------------#

Membrane_1 = {
    "Name": 'Membrane_1',
    "Solving_Method": 'CC',                     # 'CC' or 'CO' - CC is for counter-current, CO is for co-current
    "Temperature": 35+273.15,                   # Kelvin
    "Pressure_Feed": 5.80605279885049,                         # bar
    "Pressure_Permeate": 1,                   # bar
    "Q_A_ratio": 1.92046790858698,                           # ratio of the membrane feed flowrate to its area (in m3(stp)/m2.hr)
    "Permeance": [360, 13, 60, 360],        # GPU at 35C  - aged - https://doi.org/10.1016/j.memsci.2017.02.012
    "Pressure_Drop": True,
    }

Membrane_2 = {
    "Name": 'Membrane_2',
    "Solving_Method": 'CC',                   
    "Temperature": 35+273.15,                   
    "Pressure_Feed": 3.34562438356631,                       
    "Pressure_Permeate": 1,                  
    "Q_A_ratio": 8.81807746737551,                           # ratio of the membrane feed flowrate to its area (in m3(stp)/m2.hr)
    "Permeance": [360, 13, 60, 360],        # GPU
    "Pressure_Drop": True,
    }

Process_param = {
"Recycling_Ratio" : 0.9, # Ratio of the retentate flow from Membrane 2 that is recycled back to Membrane 1 feed    
"Target_Purity" : 0.95, # Target purity of the dry permeate from Membrane 2
"Target_Recovery" : 0.9, # Target recovery from Membrane 2 - for now not a hard limit, but a target to be achieved
"Replacement_rate": 4, # Replacement rate of the membranes (in yr)
"Operating_hours": 8000, # Operating hours per year
"Lifetime": 20, # Lifetime of the plant (in yr)
"Base Plant Cost": 149.8 * 1e6, # Total direct cost of plant (no CCS) in 2014 money
"Contingency": 0.3, # or 0.4 (30% or 40% contingency for process design - based on TRL)
}

Component_properties = {
    "Viscosity_param": ([0.0479,0.6112],[0.0466,3.8874],[0.0558,3.8970], [0.03333, -0.23498]), # Viscosity parameters for each component: slope and intercept for the viscosity correlation wiht temperature (in K) - from NIST
    "Molar_mass": [44.009, 28.0134, 31.999,18.01528], # Molar mass of each component in g/mol"
    "Activation_Energy_Aged": ([12750,321019],[25310,2186946],[15770,196980],[12750,321019]), # ([Activation energy - J/mol],[pre-exponential factor - GPU])
    "Activation_Energy_Fresh": ([2880,16806],[16520,226481],[3770,3599],[2880,16806]), #Valid only if temperature is -20 C or under - not considered for now
    }


Fibre_Dimensions = {
"D_in" : 150 * 1e-6, # Inner diameter in m (from um)
"D_out" : 300 * 1e-6, # Outer diameter in m (from um)
}
  
J = len(Membrane_1["Permeance"]) #number of components



with UNISIMConnector(unisim_path, close_on_completion=False) as unisim:

    class ConvergenceError(Exception):
        pass

    def Ferrari_Paper_Main(Param):
    
        Membrane_1 , Membrane_2, Process_param, Component_properties, Fibre_Dimensions, J =  Param


        Feed = {}
        Flue_Inlet = unisim.get_spreadsheet('Flue_Inlet')
        Feed['Feed_Flow'] = Flue_Inlet.get_cell_value('C3') / 3.6 # feed flow rate from UNISIM in mol/s (from kmol/h)
        Feed["Feed_Composition"] = [Flue_Inlet.get_cell_value(f'C{i+4}') for i in range(J)] #feed mole fractions from UNISIM

        #----------------------------------------#
        #----------- Get spreadsheets -----------#
        #----------------------------------------#

        Membrane1 = unisim.get_spreadsheet('Membrane 1')
        Membrane2 = unisim.get_spreadsheet('Membrane 2')
        Recycle_Membrane_2 = unisim.get_spreadsheet('Recycle Membrane 2')
        Duties = unisim.get_spreadsheet('Duties')


        #-----------------------------------------#
        #------------- Initial setup -------------#
        #-----------------------------------------#

        # Reset streams
        for i in range(J):
            Membrane1.set_cell_value(f'D{i+14}', 0) # Reset Membrane 1 permeate component flows
            Membrane1.set_cell_value(f'D{i+21}', 0) # Reset Membrane 1 retentate component flows
            Membrane2.set_cell_value(f'D{i+14}', 0) # Reset Membrane 2 permeate component flows
            Membrane2.set_cell_value(f'D{i+21}', 0) # Reset Membrane 2 retentate component flows
    
        unisim.wait_solution()

        # Setup Recycling Ratio
        Recycle_Membrane_2.set_cell_value('C3', Process_param["Recycling_Ratio"] ) # Set recycling ratio in the spreadsheet

        # Setup temperatures and pressures
        Membrane1.set_cell_value('D3', Membrane_1["Temperature"])  # Set temperature in Kelvin
        Membrane1.set_cell_value('D4', Membrane_1["Pressure_Feed"])  # Set feed pressure in bar
        Membrane1.set_cell_value('D6', Membrane_1["Pressure_Permeate"])  # Set permeate pressure in bar
        Membrane2.set_cell_value('D3', Membrane_2["Temperature"]) 
        Membrane2.set_cell_value('D4', Membrane_2["Pressure_Feed"]) 
        Membrane2.set_cell_value('D6', Membrane_2["Pressure_Permeate"])  # Set permeate pressure in bar

        unisim.wait_solution()
        #------------------------------------------------#
        #--------- Get correct feed from Unisim ---------#
        #------------------------------------------------#

        #Function to select the correct membrane compression train - Determined to be the one with the lowest compressor duty
    

        def Mem_Train_Choice(Membrane):
        
            unisim.wait_solution()

            Train_data = []
            for i in range(3):  
                if Membrane == Membrane_1:
                    Train_data.append([  
                        Duties.get_cell_value(f'H{i+9}'),  # Compressor Duty (kW)
                        Duties.get_cell_value(f'I{i+9}'),  # Hex Area (m2)
                        Duties.get_cell_value(f'J{i+9}'),  # Water Flowrate (kg/hr)
                        Duties.get_cell_value(f'K{i+9}')/1e6   # Cryogenic Cooler Duty (GJ/hr)
                    ])
                elif Membrane == Membrane_2:
                    Train_data.append([  
                        Duties.get_cell_value(f'H{i+15}'),  # Compressor Duty (kW)
                        Duties.get_cell_value(f'I{i+15}'),  # Hex Area (m2)
                        Duties.get_cell_value(f'J{i+15}'),   # Water Flowrate (kg/hr)
                        Duties.get_cell_value(f'K{i+15}')/1e6   # Cryogenic Cooler Duty (GJ/hr)
                    ])

                else: raise ValueError ("Incorrect membrane denomination")

            # Filter out trains with None or non-positive compressor duty
            valid_train_indices = [i for i, train in enumerate(Train_data) if train[0] is not None and train[0] > 0 and train[1] is not None and train[1] >0]
            if not valid_train_indices:
                raise ValueError("No valid trains found with positive compressor duty.")
        
            # Find the index of the train with the lowest compressor duty in Train_data
            lowest_duty_train_index = min(valid_train_indices, key=lambda i: Train_data[i][0])
            lowest_duty_train = Train_data[lowest_duty_train_index]
            # Read the spreadsheet of the corresponding train  
        
            if Membrane == Membrane_1:
                Mem_train = unisim.get_spreadsheet(f'Train 10{lowest_duty_train_index + 1}')  
                Membrane_1['Feed_Flow'] = Mem_train.get_cell_value('C3') / 3.6 # feed flow rate from UNISIM in mol/s (from kmol/h)
                Membrane_1["Feed_Composition"] = [Mem_train.get_cell_value(f'C{i+4}') for i in range(J)] #feed mole fractions from UNISIM
                Membrane_1["Train_Data"] = lowest_duty_train # Store the train data in the membrane dictionary
            elif Membrane == Membrane_2:
                Mem_train = unisim.get_spreadsheet(f'Train 20{lowest_duty_train_index + 1}')
                Membrane_2['Feed_Flow'] = Mem_train.get_cell_value('C3') / 3.6 # feed flow rate from UNISIM in mol/s (from kmol/h)
                Membrane_2["Feed_Composition"] = [Mem_train.get_cell_value(f'C{i+4}') for i in range(J)] #feed mole fractions from UNISIM
                Membrane_2["Train_Data"] = lowest_duty_train # Store the train data in the membrane dictionary

        #------------------------------------------#
        #--------- Function to run module ---------#
        #------------------------------------------#

        def Run(Membrane):

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
        
            
            #Reformat Permeance and Pressure values to the initial units
            Membrane["Permeance"] = [p / ( 3.348 * 1e-10 ) for p in Membrane["Permeance"]]  # convert from mol/m2.s.Pa to GPU
            Membrane["Pressure_Feed"] *= 1e-5  #convert to bar
            Membrane["Pressure_Permeate"] *= 1e-5  

            if np.any(profile<-1e-5):
                #print(profile)
                #print("Negative values in the membrane profile") #check for negative values in the profile
                raise ConvergenceError  # Return a scalar and an empty dictionary as a placeholder


            #print(profile)
            return results, profile 


        ### Run iterations for process recycling loop - Specific to this configuration!

        max_iter = 150
        tolerance = 5e-5

        Placeholder_1={ #Intermediade data storage for the recycling loop entering the first membrane used to check for convergence
            "Feed_Composition": [0] * J,
            "Feed_Flow": 0,              
            } 

        for j in range(max_iter):

            #print(f"Iteration {j+1}")

            Mem_Train_Choice(Membrane_1)

            try:
                results_1 , profile_1 = Run(Membrane_1) # Run the first membrane
            except ConvergenceError:
                return 5e8

            for i in range(J): #results "[0]: x", "[1]: y", "[2]: Q_ret", "[3]: Q_perm"
                Membrane1.set_cell_value(f'D{i+14}', results_1[1][i] * results_1[3] * 3.6) # convert from mol/s to kmol/h
                Membrane1.set_cell_value(f'D{i+21}', results_1[0][i] * results_1[2] * 3.6)

            Mem_Train_Choice(Membrane_2) # Get the correct membrane compression train for Membrane 2
            
            try: 
                results_2, profile_2 = Run(Membrane_2) # Run the second membrane - #results "[0]: x", "[1]: y", "[2]: Q_ret", "[3]: Q_perm"
            except ConvergenceError:
                return 5e8


            for i in range(J): #results "[0]: x", "[1]: y", "[2]: Q_ret", "[3]: Q_perm"
                Membrane2.set_cell_value(f'D{i+14}', results_2[1][i] * results_2[3] * 3.6)
                Membrane2.set_cell_value(f'D{i+21}', results_2[0][i] * results_2[2] * 3.6) 

            Convergence_Composition = sum(abs(np.array(Placeholder_1["Feed_Composition"]) - np.array(Membrane_1["Feed_Composition"])))
            Convergence_Flowrate = abs( ( (Placeholder_1["Feed_Flow"]) - (Membrane_1["Feed_Flow"] ) ) / (Membrane_1["Feed_Flow"] ) / 100 )
            #print(f'Convergence Composition: {Convergence_Composition:.3e}, Convergence Flowrate: {Convergence_Flowrate*100:.3e}')
            #check for convergence

            if j > 0 and Convergence_Composition < tolerance and Convergence_Flowrate < tolerance:  
                '''
                print(f"Converged after {j} iterations")
                print()
                print("checking convergence for debugging:")
                print(f'Membrane 2 Feed Composition before iteration is {[f"{comp:.4g}" for comp in Placeholder_1["Feed_Composition"]]} with flow {Placeholder_1["Feed_Flow"]:.4g} mol/s')
                print(f'Membrane 2 Feed Composition after iteration is {[f"{comp:.4g}" for comp in Membrane_1["Feed_Composition"]]} with flow {Membrane_1["Feed_Flow"]:.4g} mol/s')
                print()
                print(f'Component 1 final Recovery is {results_2[1][0]*results_2[3]/(Feed["Feed_Composition"][0]*Feed["Feed_Flow"])*100:.4f}%')
                print(f'Component 1 final Purity before liquefaction is {results_2[1][0]*100:.4f}%')
                print()
                print(f'Approximate final Purity after liquefaction is {(results_2[1][0]/(1-results_2[1][-1])):.2%}')
                print(f'Membrane 1 Profile')
                print(profile_1)
                print()
                print(f'Membrane 2 Profile')
                print(profile_2)
                print()
                '''


                break

            #Ready for next iteration
            Placeholder_1["Feed_Composition"] = Membrane_1["Feed_Composition"]
            Placeholder_1["Feed_Flow"] = Membrane_1["Feed_Flow"]

        
            unisim.wait_solution()


        else: 
            print("Max iterations for recycling reached")
            return 5e8  # Return a scalar and an empty dictionary as a placeholder
                                        

        #-------------------------------------------#
        #----------- Export/Import UniSim ----------#
        #-------------------------------------------#

        def Duty_Gather():  # Gather Duties of the trains from the solved process
            def get_lowest_duty_train(train_data):
                # Filter out trains with None or non-positive compressor duty
                valid_trains = [i for i, train in enumerate(train_data) if train[0] is not None and train[0] > 0 and train[1] is not None and train[1] >0]
                if not valid_trains:
                    raise ValueError("No valid trains found with positive compressor duty.")
            
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
                        Duties.get_cell_value(f'K{i+start_row}')/1e6 if start_row != 3 else 0  # Cryogenic Cooler Duty (MJ/hr)
                    ]
                    for i in range(3)
                ]

            Train1 = gather_train_data(9)
            Train2 = gather_train_data(15)
            Liquefaction = gather_train_data(3)

            # Get the train with the lowest compressor duty for each category
            Train1_lowest = get_lowest_duty_train(Train1)
            Train2_lowest = get_lowest_duty_train(Train2)
            Liquefaction_lowest = get_lowest_duty_train(Liquefaction)

            return Train1_lowest, Train2_lowest, Liquefaction_lowest

        Train1, Train2, Liquefaction = Duty_Gather() # Gather the duties from the solved process

        # Gather the energy recovery form the retentate. Assume flue gas at 1 bar and a maximum temperature of 120 C to match original flue gas.
        Expanders = (Duties.get_cell_value('H21'), Duties.get_cell_value('H24'), Duties.get_cell_value('H27')) # Get the retentate expanders duties (kW)
        Heaters = (Duties.get_cell_value('I21'), Duties.get_cell_value('I24')) # Get the retentate heaters duties (kJ/hr)

        # Gather the cryogenic cooler duties - if any
        Cryogenics = ( (Train1[3], Membrane_1["Temperature"]) , (Train2[3], Membrane_2["Temperature"]) ) # Get the cryogenic cooler duties (MJ/hr) for each membrane train

        # Add information to the compression trains about their number of compressors and coolers
        Train1.append(Train1[4]+1) # Append the number of compressors in the train
        Train1.append(Train1[4]+2)  # Extra heat exchanger for retentate heat recovery

        Train2.append(Train2[4]+1)
        if Process_param["Recycling_Ratio"] == 1: # If the recycling ratio is 1, no extra heat exchanger is needed for retentate heat recovery
            Train2.append(Train2[4]+1)
        else: 
            Train2.append(Train2[4]+2)  # Extra heat exchange for retentate heat recovery

        Liquefaction.append(Liquefaction[4]+3)  # Append the number of compressors and heat exchangers in the liquefaction train
        Liquefaction.append(Liquefaction[4]+3)

        #print(Train1) # compressor duty - heat exchanger area - water flowrate - cryogenic cooler duty - index - number of compressors - number of coolers


        '''
        Process_specs = {
        ...
        "Compressor_trains" : ([duty1, number_of_compressors1], ... , [dutyi, number_of_compressorsi]), # Compressor trains data]
        "Cooler_trains" : ([area1, waterflow1, number_of_coolers1], ... , [areai, waterflowi, number_of_coolersi]), # Cooler trains data
        "Membranes" : (Membrane_1, ..., Membrane_i), # Membrane data)
        "Expanders" : ([expander1_duty], ...[expanderi_duty]), # Expander data
        "Heaters" : ([heater1_duty], ...[heateri_duty]), # Heater data
        "Cryogenics" = ([cooling_power1, temperature1], ... [cooling_poweri, temperaturei]), # Cryogenic cooling data
        }
        '''

        Process_specs = {
            "Feed": Feed,
            "Purity": (results_2[1][0]/(1-results_2[1][-1])),
            "Recovery": results_2[1][0]*results_2[3]/(Feed["Feed_Composition"][0]*Feed["Feed_Flow"]),
            "Compressor_trains": ( (Train1[0], Train1[-2]), (Train2[0],Train2[-2]), (Liquefaction[0], Liquefaction[-2]) ),  # Compressor trains data
            "Cooler_trains": ( (Train1[1], Train2[2], Train1[-1]), (Train2[1],Train2[2], Train2[-1]), (Liquefaction[1], Liquefaction[2], Liquefaction[-1]) ),  # Cooler trains data"
            "Membranes": (Membrane_1, Membrane_2),
            "Expanders": Expanders,  # Expander data
            "Heaters": Heaters,  # Heater data
            "Cryogenics": Cryogenics,
        }


        from Costing_General import Costing
        Economics = Costing(Process_specs, Process_param)


        return(Economics)

    #-------------------------------#
    #----- Optimisation Method -----#
    #-------------------------------#


    def Opti_algorithm():
    
        # Define the bounds for the parameters to be optimised
        bounds = [  
            Opti_Param["Recycling_Ratio"],  # Recycling ratio range for the process
            Opti_Param["Q_A_ratio_1"],  # Flow/Area ratio for the second stage
            Opti_Param["P_up_1"],  # Upper pressure range for the second stage in bar    
            Opti_Param["Q_A_ratio_2"],  # Flow/Area ratio for the first stage
            Opti_Param["P_up_2"],  # Upper pressure range for the first stage in bar
            #Opti_Param["Temperature_1"],  # Temperature range in Celcius
            #Opti_Param["Temperature_2"],  # Temperature range in Celcius
        ]
    
        # Define the objective function to be minimised
        def objective_function(params):
            Process_param["Recycling_Ratio"] = params[0]
            Membrane_1["Q_A_ratio"] = params[1]
            Membrane_1["Pressure_Feed"] = params[2]
            Membrane_2["Q_A_ratio"] = params[3]
            Membrane_2["Pressure_Feed"] = params[4]
            #Membrane_1["Temperature"] = params[5] + 273.15
            #Membrane_2["Temperature"] = params[6] + 273.15
            Parameters = Membrane_1, Membrane_2, Process_param, Component_properties, Fibre_Dimensions, J
            Economics = Ferrari_Paper_Main(Parameters)
            if isinstance(Economics, dict):
                #print(f"Debug - objective function called with Evaluation: {Economics["Evaluation"]:.3e}")
                return Economics['Evaluation']  # Return the evaluation metric to be minimised
                
            else:
                return 2e9  # If simulation fails, return a large number to avoid this solution


        # Callback function to track progress
        def callback(xk, convergence):
            elapsed_time = time.time() - callback.start_time
            print(f"Iteration: {callback.n_iter}, Elapsed Time: {elapsed_time:.2f}s, Best Solution: {xk} Objective Function Value, Convergence: {convergence:.4f}")
            print()
            callback.n_iter += 1

        callback.n_iter = 1
        callback.start_time = time.time()

        # Run the optimisation algorithm
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter = 500,  
            popsize = 22, # WARNING popsize is the factor applied to the number of parameters, so 20 means 100 individuals for 5 parameters - not a population of 20
            tol = 5e-3,
            callback=callback,
            polish=True
        )

        def Solved_process():
            Process_param["Recycling_Ratio"] = result.x[0]
            Membrane_1["Q_A_ratio"] = result.x[1]
            Membrane_1["Pressure_Feed"] = result.x[2]
            Membrane_2["Q_A_ratio"] = result.x[3]
            Membrane_2["Pressure_Feed"] = result.x[4]
            #Membrane_1["Temperature"] = result.x[5] + 273.15
            #Membrane_2["Temperature"] = result.x[6] + 273.15
            Parameters = Membrane_1, Membrane_2, Process_param, Component_properties, Fibre_Dimensions, J
            Economics = Ferrari_Paper_Main(Parameters)
            if isinstance(Economics, dict):
                return Economics
            else: 
                return "Simulation failed to converge"

        Economics = Solved_process()

        print("Optimisation Result:")
        print("Optimal Parameters: ", [f"{res:.5f}" for res in result.x])
        print(f"Objective Function Value: {result.fun:.5e}")
        print(Economics)
        def save_results(results, Economics, filename):
            with open(filename, 'w') as f:
                f.write("Best Parameters: {}\n".format(results.x))
                f.write("Objective Value: {}\n".format(results.fun))
                f.write("Iterations: {}\n".format(results.nit))
                f.write("Convergence Message: {}\n".format(results.message))
                f.write("Economics: {}\n".format(Economics))

        def format_economics(Economics):
            return "\n".join(f"{key}: {value}" for key, value in Economics.items())

        # Save the results
        filename = 'Ferrari_laptop_5param_opti.txt'
        economics_str = format_economics(Economics)
        save_results(result, economics_str, filename)

        print("Results saved to", filename)




    #------------------------------#
    #----- Brute Force Method -----#
    #------------------------------#


    def Brute_Force():

        # Define the columns based on the parameters being changed and all variables in the Economics dictionary  
        columns = [  
          f'Param_{j+1}' for j in range(len(Brute_Force_Param[0]))
        ] + [  
          "Evaluation",  # Evaluation metric  
          "Purity",  # Purity of the product  
          "Recovery",  # Recovery of the product  
          "TAC_CC",  # Total Annualised Cost of Carbon Capture  
          "Capex_tot",  # Total Capital Expenditure in 2014 money  
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
            Process_param["Recycling_Ratio"] = Brute_Force_Param[i]["Recycling_Ratio"]
            Membrane_1["Q_A_ratio"] = Brute_Force_Param[i]["Q_A_ratio_2"]
            Membrane_1["Pressure_Feed"] = Brute_Force_Param[i]["P_up_2"]
            Membrane_2["Q_A_ratio"] = Brute_Force_Param[i]["Q_A_ratio_1"]
            Membrane_2["Pressure_Feed"] = Brute_Force_Param[i]["P_up_1"]
            #Membrane_1["Temperature"] = Brute_Force_Param[i]["Temperature_1"] + 273.15
            #Membrane_2["Temperature"] = Brute_Force_Param[i]["Temperature_2"] + 273.15

            Parameters = Membrane_1, Membrane_2, Process_param, Component_properties, Fibre_Dimensions, J

            # Run the optimisation algorithm
            Economics = Ferrari_Paper_Main(Parameters)

            if isinstance(Economics, dict):
                row = list(Brute_Force_Param[i].values()) + list(Economics.values())
                rows.append(row)
            else: Invalid += 1

        # Convert the list of rows to a DataFrame
        df = pd.DataFrame(rows, columns=columns)

        filename_df = "Ferrari_desktop_7param_bruteforce.csv"

        # Define the file path correctly using the variable
        file_path = filename_df

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








   
