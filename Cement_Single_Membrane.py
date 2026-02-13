
from multiprocessing import Process
import numpy as np
import pickle
from Hub import Hub_Connector
from UNISIMConnect import UNISIMConnector
from scipy.optimize import differential_evolution  
import pandas as pd  
import tqdm
import os
import time
from Optimisation_logger import OptimisationLogger

#the ode solver sometimes returns these warning.Since we already handle incorrect mass balances we can safely ignore them.
import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings("ignore", category=LinAlgWarning) 

"""

This module is used to run the optimisation for the process simulation, would it be by brute force or using an optimisation algorithm.

A set of parameters from the process are chosen and given bounds for the optimisation algorithm to work with.

The aim is to minimise a scaler from the process simulation function, which usually will be associated with the cost of the process.

The brute force method can be used for up to three parameters and will be mainly used to validate the optimisation algorithm.

Most debugging and test messages are removed from this solution ; manual checks can be done on another project.

"""


Filename_input = input("Enter the version of the file: Original, Copy, Copy2, or Copy3: ")
if Filename_input.lower() == "original":
    filename = 'Cement_Single_Membrane_Config.usc' #Unisim file name
    directory = 'C:\\Users\\s1854031\\OneDrive - University of Edinburgh\\Python\\Cement_Plant_2021\\' #Directory of the unisim file
    results_dir = 'C:\\Users\\s1854031\\OneDrive - University of Edinburgh\\Python\\Opti_results_Graveyard\\' #Directory to save results files
    checkpoint_dir = 'C:\\Users\\s1854031\\OneDrive - University of Edinburgh\\Python\\Checkpoint_Files' #Directory to save checkpoint files
elif Filename_input.lower() == "copy":
    filename = 'Cement_Single_Membrane_Config_Copy.usc'
    directory = 'C:\\Users\\Simulation Machine\\OneDrive - University of Edinburgh\\Python\\Cement_Plant_2021\\'
    results_dir = 'C:\\Users\\Simulation Machine\\OneDrive - University of Edinburgh\\Python\\Opti_results_Graveyard\\'
    checkpoint_dir = 'C:\\Users\\Simulation Machine\\OneDrive - University of Edinburgh\\Python\\Checkpoint_Files\\'
elif Filename_input.lower() == "copy2":
    filename = 'Cement_Single_Membrane_Config_Copy2.usc'
    directory = 'C:\\Users\\Simulation Machine\\OneDrive - University of Edinburgh\\Python\\Cement_Plant_2021\\'
    results_dir = 'C:\\Users\\Simulation Machine\\OneDrive - University of Edinburgh\\Python\\Opti_results_Graveyard\\'
    checkpoint_dir = 'C:\\Users\\Simulation Machine\\OneDrive - University of Edinburgh\\Python\\Checkpoint_Files\\'
elif Filename_input.lower() == "copy3":
    filename = 'Cement_Single_Membrane_Config_Copy3.usc'
    directory = 'C:\\Users\\Simulation Machine\\OneDrive - University of Edinburgh\\Python\\Cement_Plant_2021\\'
    results_dir = 'C:\\Users\\Simulation Machine\\OneDrive - University of Edinburgh\\Python\\Opti_results_Graveyard\\'
    checkpoint_dir = 'C:\\Users\\Simulation Machine\\OneDrive - University of Edinburgh\\Python\\Checkpoint_Files' 

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
unisim_path = os.path.join(directory, filename)
logger = OptimisationLogger(
    log_dir=results_dir,
)

output_filename = 'SingleMem_3param_forchiara_mem5_220126.txt' # Output file name
checkpoint_file = "SingleMem_3param_forchiara_mem5_20126.pkl" # Checkpoint file name


#-------------------------------#
#--- Optimisation Parameters ---#
#-------------------------------#

Options = { 
    "Method": "Optimisation",  # Method is either by Brute_Force or Optimisation or Both
    "Permeance_From_Activation_Energy": True, # True will use the activation energies from the component_properties dictionary - False will use the permeances defined in the membranes dictionaries.
    "Extra_Recovery_Penalty": False,  # If true, adds a penalty to the objective function to encourage higher recoveries
    "Recovery_Soft_Cap": (True, 0.90),  # (Activate limit, value) - If true, sets a soft limit on recovery: recovery above the soft cap will not decrease the primary emission cost further 
    }    
print(Options) 
if Options["Method"] == "Both":print(f"Using software path: {filename}; Running both Bruteforce and Optimisation methods") 
else: print(f"Running the { Options["Method"]} method for path {filename}")


# Set bounds of optimisation parameters - comment unused parameters
Opti_Param = {
    "Q_A_ratio_1" : [0.5, 10], # Flow/Area ratio for the second stage
    "P_up_1" : [1.3, 15],  # Upper pressure range for the second stage in bar
    "Recycling_Ratio" : [0, 0.95],  # Ratio of the retentate flow from the membrane that is recycled back to the feed
    #"Temperature_1" : [-40, 70],  # Temperature range in Celcius
    }

#--------------------------#
#--- Default Parameters ---#
#--------------------------#
'''These include parameters that will be modified through the optimisation function'''

Membrane_1 = {
    "Name": 'Membrane_1',
    "Solving_Method": 'CC_ODE',                 # 'CC' or 'CO' - CC is for counter-current, CO is for co-current
    "Temperature": 25+273.15,               # Kelvin
    "Pressure_Feed": 5.80,                  # bar
    "Pressure_Permeate": 0.22,                 # bar
    "Q_A_ratio": 1.92,                      # ratio of the membrane feed flowrate to its area (in m3(stp)/m2.hr)
    "Permeance": [1000, 1000/200, 1000/80, 1000], # in GPU [CO2, N2, O2, H2O]
    "Pressure_Drop": False,
    }

Sweep = {
    "Option" : False, #True or False - False is no sweep
    "Source" : "Retentate", # Retentate or User. Defines source of sweep to be fom a retentate recycling or driectly defined from the user
    ### if the is a sweep ###
    # if source == retentate:
    "Ratio" : 0.1, #Define the fraction of the retentate used as the sweep
    # if source == user:
    "Temperature" : 35+273.15, #Temperature and pressure of the sweep are conditioned later in unisim to fit the membrane properties
    "Pressure": 1,
    "Flowrate": 1e3, #mol/h
    "Composition": np.array([0, 0.2, 0, 8]),

    }

Process_param = {
    "Recycling_Ratio" : 1,      # Ratio of the retentate flow from Membrane 2 that is recycled back to Membrane 1 feed    
    "Target_Purity" : 0.95,     # Target purity of the dry permeate from Membrane 2
    "Target_Recovery" : 0.9,    # Target recovery from Membrane 2 - for now not a hard limit, but a target to be achieved
    "Replacement_rate": 4,      # Replacement rate of the membranes (in yr)
    "Operating_hours": 8000,    # Operating hours per year
    "Lifetime": 25,             # Lifetime of the plant (in yr)
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
"D_in" : 600 * 1e-6,    # Inner diameter in m (from um)
"D_out" : 800 * 1e-6,   # Outer diameter in m (from um)
}
  
J = len(Membrane_1["Permeance"]) #number of components



with UNISIMConnector(unisim_path, close_on_completion=False) as unisim:

    class ConvergenceError(Exception): #allowing to skip iteration if convergence error appears in one of the mass balance
        pass

    def Single_Membrane_Main(Param):
    
        Membrane_1 , Process_param, Component_properties, Fibre_Dimensions, J =  Param

        Feed = {}
        Flue_Inlet = unisim.get_spreadsheet('Flue_Inlet')
        Feed["Feed_Flow"] = Flue_Inlet.get_cell_value('C3') / 3.6 # feed flow rate from UNISIM in mol/s (from kmol/h)
        Feed["Feed_Composition"] = [Flue_Inlet.get_cell_value(f'C{i+4}') for i in range(J)] #feed mole fractions from UNISIM

        #----------------------------------------#
        #----------- Get spreadsheets -----------#
        #----------------------------------------#

        Membrane1 = unisim.get_spreadsheet('Membrane 1')
        Recycle_Membrane_1 = unisim.get_spreadsheet('Recycle Membrane 1')
        Sweep1 = unisim.get_spreadsheet('Sweep 1')
        Duties = unisim.get_spreadsheet('Duties')
        Sweep_Check = unisim.get_spreadsheet('Sweep_Check')


        #-----------------------------------------#
        #------------- Initial setup -------------#
        #-----------------------------------------#

        # Reset streams
        for i in range(J):
            Membrane1.set_cell_value(f'D{i+14}', 0) # Reset Membrane 1 permeate component flows
            Membrane1.set_cell_value(f'D{i+21}', 0) # Reset Membrane 1 retentate component flows
    
        unisim.wait_solution(timeout=10, check_pop_ups=2, check_consistency_error=3)

        # Setup Recycling Ratio
        Recycle_Membrane_1.set_cell_value('C3', Process_param["Recycling_Ratio"] ) # Set recycling ratio in the spreadsheet
        if Sweep["Option"] and Sweep["Source"]=="Retentate":
            Recycle_Membrane_1.set_cell_value('C4', Sweep["Ratio"] ) # Set retentate recycling as sweep ratio in the spreadsheet
        else :
            Recycle_Membrane_1.set_cell_value('C4', 0 ) # No sweep from retentate recycling

        # Setup temperatures and pressures
        Membrane1.set_cell_value('D3', Membrane_1["Temperature"])  # Set temperature in Kelvin
        Membrane1.set_cell_value('D4', Membrane_1["Pressure_Feed"])  # Set feed pressure in bar
        Membrane1.set_cell_value('D6', Membrane_1["Pressure_Permeate"])  # Set permeate pressure in bar

        if Sweep["Option"] :
            Sweep1.set_cell_value('D3', Sweep["Temperature"]) 
            Sweep1.set_cell_value('D4', Sweep["Pressure"]) 
            if Sweep["Source"]=="User":
                for i in range(J):
                    Sweep1.set_cell_value(f'C{9+i}', Sweep["Flowrate"]*Sweep["Composition"][i])


        unisim.wait_solution(timeout=10, check_pop_ups=2, check_consistency_error=3)
        
        
        #------------------------------------------------#
        #--------- Get correct feed from Unisim ---------#
        #------------------------------------------------#
        '''Function to select the correct membrane compression train - Determined to be the one with the lowest non-null compressor duty'''

        def Mem_Train_Choice(Membrane):
        
            unisim.wait_solution(timeout=10, check_pop_ups=2, check_consistency_error=3)

            Train_data = []
            for i in range(3):  
                if Membrane == Membrane_1:
                    Train_data.append([  
                        Duties.get_cell_value(f'H{i+9}'),  # Compressor Duty (kW)
                        Duties.get_cell_value(f'I{i+9}'),  # Hex Area (m2)
                        Duties.get_cell_value(f'J{i+9}'),  # Water Flowrate (kg/hr)
                        Duties.get_cell_value(f'K{i+9}') / 1e6 if Duties.get_cell_value(f'K{i+9}') is not None and Duties.get_cell_value(f'K{i+9}') > 0 else 0  # Cryogenic Cooler Duty (MJ/hr)
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
                Membrane_1["Feed_Flow"] = Mem_train.get_cell_value('C3') / 3.6 # feed flow rate from UNISIM in mol/s (from kmol/h)
                Membrane_1["Feed_Composition"] = [Mem_train.get_cell_value(f'C{i+4}') for i in range(J)] #feed mole fractions from UNISIM
                Membrane_1["Train_Data"] = lowest_duty_train # Store the train data in the membrane dictionary


        #------------------------------------------#
        #--------- Function to run module ---------#
        #------------------------------------------#

        def Run(Membrane):

            J = len(Membrane["Permeance"])
            # Set membrane Area based on its feed flow and Q_A_ratio:
            Membrane["Area"] = (Membrane["Feed_Flow"] * 0.0224  * 3600) / Membrane["Q_A_ratio"] # (0.0224 is the molar volume of an ideal gas at STP in m3/mol)
            if not Sweep["Option"]:
                Membrane["Sweep_Flow"] = 0 # No sweep
                Membrane["Sweep_Composition"] = [0] * len(Membrane_1["Permeance"])

            elif Sweep["Source"]=="User": # Constant Sweep from User dictionary
                Membrane["Sweep_Flow"] = Sweep1.get_cell_value('C15') #need to obtain the sweep flowrates after conditioning to membrane temperature and pressure
                Membrane["Sweep_Composition"] = [0] * J
                for i in range(J):
                    Membrane["Sweep_Composition"][i] = Sweep1.get_cell_value(f'C{16+i}') / Membrane["Sweep_Flow"]

            elif Sweep["Source"]=="Retentate": #Sweep from Retentate Recycling
                Membrane["Sweep_Flow"] = Sweep1.get_cell_value('G8') #need to obtain the sweep flowrates after conditioning to membrane temperature and pressure
                Membrane["Sweep_Composition"] = [0] * J
                if not Membrane["Sweep_Flow"]:
                    Membrane["Sweep_Flow"] = 0  # Handles no sweep on the first iteration (no recycling yet)
                else:
                    for i in range(J):
                        Membrane["Sweep_Composition"][i] = Sweep1.get_cell_value(f'G{9+i}') / (1e-12 + Membrane["Sweep_Flow"])

            Export_to_mass_balance = Membrane, Component_properties, Fibre_Dimensions

            J = len(Membrane["Permeance"]) #number of components
            
            if Options["Permeance_From_Activation_Energy"]:
                # Obtain Permeance with temperature:
                for i in range(J):
                    Membrane["Permeance"][i] = Component_properties["Activation_Energy_Aged"][i][1] * np.exp(-Component_properties["Activation_Energy_Aged"][i][0] / (8.314 * Membrane["Temperature"]))


            results, profile = Hub_Connector(Export_to_mass_balance)
            Membrane["Retentate_Composition"],Membrane["Permeate_Composition"],Membrane["Retentate_Flow"],Membrane["Permeate_Flow"] = results

            #Reformat Permeance and Pressure values to the initial units - will find a smarter way to do this later
            Membrane["Permeance"] = [p / ( 3.348 * 1e-10 ) for p in Membrane["Permeance"]]  # convert from mol/m2.s.Pa to GPU
            Membrane["Pressure_Feed"] *= 1e-5  #convert to bar
            Membrane["Pressure_Permeate"] *= 1e-5  
        
            errors = []
            for i in range(J):    
                # Calculate comp molar flows
                Feed_Sweep_Mol = Membrane["Feed_Flow"] * Membrane["Feed_Composition"][i] + Membrane["Sweep_Flow"] * Membrane["Sweep_Composition"][i]
                Retentate_Mol = Membrane["Retentate_Flow"] * Membrane["Retentate_Composition"][i]
                Permeate_Mol = Membrane["Permeate_Flow"] * Membrane["Permeate_Composition"][i]
    
                # Calculate and store the error
                error = abs((Feed_Sweep_Mol - Retentate_Mol - Permeate_Mol)/Feed_Sweep_Mol)
                errors.append(error)

            # Calculate the cumulated error
            cumulated_error = sum(errors) - errors[-1] # Remove water because its relative error is large at low temperature (1e-4). Its absolute error however is negligible due to its very low concentration
            if np.any(profile<-1e-5) or cumulated_error>1e-5 or errors[-1]>1e-3:
                raise ConvergenceError 
                
            return results, profile

        #-----------------------------------------------------------------------------#
        # Run iterations for process recycling loop - Specific to this configuration! #
        #-----------------------------------------------------------------------------#

        max_iter = 150      # maximum number of iterations for the recycling loop
        tolerance = 5e-5    # convergence tolerance for the recycling loop

        Placeholder_1={ #Intermediade data storage for the recycling loop entering the first membrane used to check for convergence
            "Feed_Composition": [0] * J,
            "Feed_Flow": 0,              
            } 

        for j in range(max_iter):

            unisim.wait_solution(timeout=10, check_pop_ups=2, check_consistency_error=3)

            try:
                Mem_Train_Choice(Membrane_1)
                results_1 , profile_1 = Run(Membrane_1) # Run the first membrane
            except ConvergenceError:
                return 5e8

            for i in range(J): #results "[0]: x", "[1]: y", "[2]: Q_ret", "[3]: Q_perm"
                Membrane1.set_cell_value(f'D{i+14}', results_1[1][i] * results_1[3] * 3.6) # convert from mol/s to kmol/h and send to unisim
                Membrane1.set_cell_value(f'D{i+21}', results_1[0][i] * results_1[2] * 3.6)
            unisim.wait_solution(timeout=10, check_pop_ups=2, check_consistency_error=3)

            Convergence_Composition = sum(abs(np.array(Placeholder_1["Feed_Composition"]) - np.array(Membrane_1["Feed_Composition"])))
            Convergence_Flowrate = abs( ( (Placeholder_1["Feed_Flow"]) - (Membrane_1["Feed_Flow"] ) ) / (Membrane_1["Feed_Flow"] ) / 100 )

            #check for convergence
            if j > 0 and Convergence_Composition < tolerance and Convergence_Flowrate < tolerance:  
                break

            #Ready for next iteration
            Placeholder_1["Feed_Composition"] = Membrane_1["Feed_Composition"]
            Placeholder_1["Feed_Flow"] = Membrane_1["Feed_Flow"]

            unisim.wait_solution(timeout=10, check_pop_ups=2, check_consistency_error=3)

        else: 
            print("Max iterations for recycling reached")
            return 5e9  # Return a scalar and an empty dictionary as a placeholder
                                        

        #-------------------------------------------#
        #----------- Export/Import UniSim ----------#
        #-------------------------------------------#

        def Duty_Gather():  # Gather Duties of the trains from the solved process

            def get_lowest_duty_train(train_data):

                # Filter out trains with None or non-positive compressor duty
                valid_trains = [i for i, train in enumerate(train_data) if train[0] is not None and train[0] > 0 and train[1] is not None and train[1] >0]
                if not valid_trains:
                    raise ConvergenceError
            
                # Find the train with the lowest compressor duty
                lowest_duty_train_index = min(valid_trains, key=lambda i: train_data[i][0])
                lowest_duty_train = train_data[lowest_duty_train_index]
                lowest_duty_train.append(lowest_duty_train_index) #append the index of the train with the lowest duty to know the equipment count
                return lowest_duty_train

            def gather_train_data(start_row):
                data = []
                for i in range(3):
                    k_val = Duties.get_cell_value(f'K{i + start_row}')

                    cryo = 0
                    if k_val is not None:
                        try:
                            cryo = (k_val / 1e6) if k_val > 0 else 0
                        except TypeError:
                            v = float(k_val)  # if needed
                            cryo = (v / 1e6) if v > 0 else 0

                    data.append([
                        Duties.get_cell_value(f'H{i + start_row}'),
                        Duties.get_cell_value(f'I{i + start_row}'),
                        Duties.get_cell_value(f'J{i + start_row}'),
                        cryo
                    ])
                return data


            Train1 = gather_train_data(9)
            Liquefaction = gather_train_data(3)

            # Get the train with the lowest compressor duty for each category
            Train1_lowest = get_lowest_duty_train(Train1)
            Liquefaction_lowest = get_lowest_duty_train(Liquefaction)

            return Train1_lowest, Liquefaction_lowest
        try: 
            Train1, Liquefaction = Duty_Gather() # Gather the duties from the solved process
        except ConvergenceError:
            return 5e8

        Train1.append(Train1[4]+1) # Append the number of compressors in the train
        Train1.append(Train1[4]+2)  # Extra heat exchanger for retentate heat recovery
        Liquefaction.append(Liquefaction[4]+3)  # Append the number of compressors and heat exchangers in the liquefaction train
        Liquefaction.append(Liquefaction[4]+3)

        # Gather the energy recovery form the retentate. Assume flue gas at 1 bar and a maximum temperature of 120 C to match original flue gas.
        Expanders = [Duties.get_cell_value('H15')]  
        Heaters = [Duties.get_cell_value('I15')]  
        Cooler_trains = [[Train1[1], Train1[2], Train1[-1]], [Liquefaction[1], Liquefaction[2], Liquefaction[-1]]]
        Compressor_trains = [[Train1[0], Train1[-2]], [Liquefaction[0], Liquefaction[-2]]]

        if Process_param["Recycling_Ratio"] > 0:  # Add energy recovery of recycling loop
            Expanders.append(Duties.get_cell_value('H18'))

        if Sweep["Option"]:  # Add energy recovery from sweep conditioning
            if Sweep["Source"] == "Retentate":  # We have an expander and heater
                Expanders.append(Duties.get_cell_value('H22'))
                Heaters.append(Duties.get_cell_value('I22'))
                Cooler_trains[0][-1] += 1  # Add one heat exchanger to the mem1 pre-conditioning for heat recovery
            elif Sweep["Source"] == "User":
                Expanders.append(Duties.get_cell_value('H21'))
                if Sweep_Check.get_cell_value('E8') >= Sweep_Check.get_cell_value('D8'):  # Need heater for sweep conditioning
                    Heaters.append(Duties.get_cell_value('I21'))
                    Cooler_trains[0][-1] += 1
                else:  # Need cooler for sweep conditioning
                    Cooler_trains.append([Sweep_Check.get_cell_value('G9'), Sweep_Check.get_cell_value('H9'), 1])  # Add cooler to equipment list

        # Gather the cryogenic cooler duties - if any
        Cryogenics = [ (Train1[3], Membrane_1["Temperature"])] # Get the cryogenic cooler duties (MJ/hr) for each membrane train

        #Obtain water content in the compression train to dehydrate
        H2O_train = []
        for k in range(3):
            H2O_train.append(Duties.get_cell_value(f'H{k+30}'))

        if H2O_train:
            valid_water = []
            for water in H2O_train:
                if water is not None:  # Check if the element is not None
                    valid_water.append(water)
            H2O_to_remove = max(min(valid_water), 0) if valid_water else 0
           
        else: H2O_to_remove=0
    
        #Obtain vacuum pump duty and resulting cooling duty from each membrane:
        Vacuum_1 = unisim.get_spreadsheet("Vacuum_1")
        Vacuum_Duty1 = [Vacuum_1.get_cell_value("B10")] # kW
        Vacuum_Cooling1 = [Vacuum_1.get_cell_value("G10"),Vacuum_1.get_cell_value("H10")]  # Area, WaterFlow
        #PS: logic is implemented in unisim for coolers. If output of the vacuum pump is not hot (<35 C), the cooler will not be active and will return 0 duty.


        '''
        Process_specs = {
        ...
        "Compressor_trains" : ([duty1, number_of_compressors1], ... , [dutyi, number_of_compressorsi]), # Compressor trains data]
        "Cooler_trains" : ([area1, waterflow1, number_of_coolers1], ... , [areai, waterflowi, number_of_coolersi]), # Cooler trains data
        "Membranes" : (Membrane_1, ..., Membrane_i), # Membrane data)
        "Expanders" : ([expander1_duty], ...[expanderi_duty]), # Expander data
        "Heaters" : ([heater1_duty], ...[heateri_duty]), # Heater data
        "Cryogenics" : ([cooling_power1, temperature1], ... [cooling_poweri, temperaturei]), # Cryogenic cooling data
        "Dehydration" : ([Mass_flow_H2O]) #mass flow of H2O at 30 bar in the compression train
        "Vacuum_Pump": ([Pump_Duty_1],[Duty2],...,[Dutyi])
        "Vacuum_Cooling": ([area1, waterflow1], ... , [areai, waterflowi]) #required when vacuum pump outlet is hot
        }
        '''

        Process_specs = {
            "Feed": Feed,
            "Purity": (results_1[1][0]/(1-results_1[1][-1])),
            "Recovery": results_1[1][0]*results_1[3]/(Feed["Feed_Composition"][0]*Feed["Feed_Flow"]),
            "Compressor_trains": Compressor_trains ,  # Compressor trains data
            "Cooler_trains": Cooler_trains,  # Cooler trains data"
            "Membranes": ([Membrane_1]),
            "Expanders": Expanders,  # Expander data
            "Heaters": Heaters,  # Heater data
            "Cryogenics": Cryogenics,
            "Dehydration":(H2O_to_remove),
            "Vacuum_Pump":Vacuum_Duty1,
            "Vacuum_Cooling": (Vacuum_Cooling1,),
        }

        def replace_none_with_zero(obj):
            if obj is None:
                return 0
            if isinstance(obj, dict):
                return {k: replace_none_with_zero(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [replace_none_with_zero(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(replace_none_with_zero(v) for v in obj)
            if isinstance(obj, set):
                return {replace_none_with_zero(v) for v in obj}

            return obj       
        Process_specs = replace_none_with_zero(Process_specs)    

        #print(Process_specs)

        from Costing import Costing
        Economics = Costing(Process_specs, Process_param, Component_properties, Options)
        
        return(Economics)


    #-------------------------------#
    #----- Optimisation Method -----#
    #-------------------------------#

    def Opti_algorithm():
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file) # Use the savepoints directory

        popsize = 25 #Larger popsize for single membrane due to less parameters to optimise
        
                # ----------------- Load checkpoint or midpoint guess -----------------
        def load_checkpoint():
            """Ask user whether to reload from checkpoint, otherwise use midpoint guess."""
    
            if os.path.exists(checkpoint_path):
                choice = input("A checkpoint was found. Do you want to resume? (y/n): ").strip().lower()
                if choice == "y":
                    with open(checkpoint_path, "rb") as f:
                        checkpoint_data = pickle.load(f)
                    if len(checkpoint_data["best_solution"]) == len(bounds):
                        print(f"Resuming from iteration {checkpoint_data['iteration']} with best solution {checkpoint_data['best_solution']}")
                        x0 = checkpoint_data["best_solution"]

                        # --- bounds-scaled Gaussian population around checkpoint ---
                        widths = np.array([b[1] - b[0] for b in bounds], float)
                        sigma  = 0.1 * widths
                        N, D   = popsize * len(bounds), len(bounds)
                        init_pop = x0 + np.random.normal(0.0, sigma, size=(N, D))
                        init_pop = np.clip(init_pop, [b[0] for b in bounds], [b[1] for b in bounds])
                        init_pop[0] = x0
                        '''
                        # Inject some random individuals
                        num_random_individuals = int(0.1 * N)  # 10% of the population
                        random_individuals = np.random.uniform(
                            low=[b[0] for b in bounds], 
                            high=[b[1] for b in bounds], 
                            size=(num_random_individuals, D)
                        )
                        init_pop[-num_random_individuals:] = random_individuals
                        '''
                        # Ensure the best solution from checkpoint is included
                        init_pop[0] = x0
                        return init_pop
                    else:
                        print(checkpoint_data["best_solution"])
                        print("Checkpoint incompatible with current problem, starting fresh.")

            # --- No checkpoint: ask user for initialisation method ---
            
            user_choice = input("Use initial guess? (y for initial guess, n for random): ").strip().lower()
            use_initial_guess = (user_choice == "y")

            if use_initial_guess:

                # Use custom guess
                first_guess = np.array([3,1.5,8,1.3])
                print(f"Starting with guess: {first_guess}")

                widths = np.array([b[1] - b[0] for b in bounds], float)
                sigma  = 0.05 * widths
                N, D   = popsize * len(bounds), len(bounds)
                init_pop = first_guess + np.random.normal(0.0, sigma, size=(N, D))
                init_pop = np.clip(init_pop, [b[0] for b in bounds], [b[1] for b in bounds])
                init_pop[0] = first_guess
                return np.array(init_pop)
            else:
                # Use random initialisation
                print("Starting with random initialisation.")
                N, D = popsize * len(bounds), len(bounds)
                init_pop = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(N, D))
                return np.array(init_pop)

        # ----------------- Define bounds -----------------
        bounds = [ 
            Opti_Param["Recycling_Ratio"],
            Opti_Param["Q_A_ratio_1"],  
            Opti_Param["P_up_1"],  
            #Opti_Param["Temperature_1"],  
  
        ]
        
        # ----------------- Objective function -----------------
        def objective_function(params):
            # Update Membrane parameters
            Process_param["Recycling_Ratio"] = params[0]
            Membrane_1["Q_A_ratio"] = params[1]
            Membrane_1["Pressure_Feed"] = params[2]
            #Membrane_1["Temperature"] = params[-1] + 273.15

            Parameters = (Membrane_1,Process_param, Component_properties, Fibre_Dimensions, J)

            Economics = Single_Membrane_Main(Parameters)

            evaluation_value = logger.log(params, Economics)
            
            # Print progress every 200 evaluations
            if logger.attempted_run % 200 == 0:
                success_percentage = logger.success / logger.attempted_run * 100
                print(f"[{logger.attempted_run} Total Runs] "
                      f"Success: {logger.success} "
                      f"({success_percentage:.1f}%), "
                      f"Failed: {logger.failed}")
            return evaluation_value

        # ----------------- Callback with checkpoint save -----------------
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
            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint_data, f)

            callback.n_iter += 1
            return False  # return True to stop optimization

        callback.n_iter = 1
        callback.start_time = time.time()

        # ----------------- Run optimization -----------------
        init_setting = load_checkpoint()

        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=1000,  
            popsize=popsize,  
            tol=5e-3,
            callback=callback,
            mutation=(0.5,1.0),
            recombination=0.7,
            polish=True,
            init=init_setting
        )

        # ----------------- Extract solution -----------------
        def Solved_process():
            Process_param["Recycling_Ratio"] = result.x[0]
            Membrane_1["Q_A_ratio"] = result.x[1]
            Membrane_1["Pressure_Feed"] = result.x[2]
            #Membrane_1["Temperature"] = result.x[-1] + 273.15
            Parameters = Membrane_1, Process_param, Component_properties, Fibre_Dimensions, J
            Economics = Single_Membrane_Main(Parameters)

            if isinstance(Economics, dict):
                return Economics
            else: 
                return "Simulation failed to converge"

        Economics = Solved_process()

        # ----------------- Save final results -----------------
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


        res_filepath = os.path.join(results_dir, output_filename)
        economics_str = format_economics(Economics)
        save_results(result, economics_str, res_filepath)

        print("Results saved to", output_filename)


    #------------------------------#
    #----- Brute Force Method -----#
    #------------------------------#


    def Brute_Force():
            
        # In case of brute force
        number_evaluation = 1000 * len(Opti_Param)  # Number of evaluations for the brute force method

        Brute_Force_Param = [  #random sets of bounded parameters
            {  
                "Recycling_Ratio": np.random.uniform(*Opti_Param["Recycling_Ratio"]),  
                "Q_A_ratio_1": np.random.uniform(*Opti_Param["Q_A_ratio_1"]),  
                "P_up_1": np.random.uniform(*Opti_Param["P_up_1"]),
                "Q_A_ratio_2": np.random.uniform(*Opti_Param["Q_A_ratio_2"]),
                "P_up_2": np.random.uniform(*Opti_Param["P_up_2"]),
                "P_perm_1": np.random.uniform(*Opti_Param["P_perm_1"]),
                "P_perm_2": np.random.uniform(*Opti_Param["P_perm_2"]),
                "Temperature_1": np.random.uniform(*Opti_Param["Temperature_1"]),
                "Temperature_2": np.random.uniform(*Opti_Param["Temperature_2"]),
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
          "TAC_Dehydration",
          "Direct_CO2_emission",  # CO2 emissions from the process in tonnes per year
          "Indirect_CO2_emission",  # CO2 emissions from electricity consumption in tonnes per year"
          "C_compressor",  # Capital cost of compressors  
          "C_vacuum_pump",
          "C_vacuum_cooler",
          "C_cooler",  # Capital cost of coolers  
          "C_membrane",  # Capital cost of membranes  
          "C_expander",  # Capital cost of expanders  
          "O_compressor",  # Operational cost of compressors per year  
          "O_vacuum_pump",
          "O_vacuum_cooler",
          "O_cooler",  # Operational cost of coolers per year  
          "O_membrane",  # Operational cost of membranes per year  
          "O_expander",  # Operational cost of expanders per year  
          "O_heater",  # Operational cost of heat integration from retentate per year  
          "Penalty_purity",  # Penalty for purity under target  
          "Penalty_CO2_emission",  # Penalty for CO2 emissions under target  
          "Cost_of_Capture",
        ]
        # Preallocate for the range of data sets
        rows = []
        Invalid = 0 #Count of invalid datasets due to simulation not converging.

        for i in tqdm.tqdm(range(number_evaluation)):
            Process_param["Recycling_Ratio"] = Brute_Force_Param[i]["Recycling_Ratio"]
            Membrane_1["Q_A_ratio"] = Brute_Force_Param[i]["Q_A_ratio_2"]
            Membrane_1["Pressure_Feed"] = Brute_Force_Param[i]["P_up_2"]
            Membrane_1["Temperature"] = Brute_Force_Param[i]["Temperature_1"] + 273.15

            Parameters = Membrane_1, Process_param, Component_properties, Fibre_Dimensions, J

            # Run the optimisation algorithm
            Economics = Single_Membrane_Main(Parameters)

            if isinstance(Economics, dict):
                row = list(Brute_Force_Param[i].values()) + list(Economics.values())
                rows.append(row)
            else: Invalid += 1

        # Convert the list of rows to a DataFrame
        df = pd.DataFrame(rows, columns=columns)

        filename_bruteforce = "Ferrari_desktop_7param_bruteforce.csv"
        
        # Define the file path correctly using the variable
        file_path = os.path.join(results_dir, filename_bruteforce)

        # Save the DataFrame to a CSV file on the desktop
        df.to_csv(file_path, index=False)


        print(f"{number_evaluation - Invalid} sets of data collected in {file_path} over {number_evaluation} evaluations total ({(number_evaluation - Invalid)/number_evaluation:.2%}). ")


    
    if Options["Method"] == "Brute_Force":
        Brute_Force()
    elif Options["Method"] == "Optimisation":
        Opti_algorithm()        
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

    elif Options["Method"] == "Both":
        Brute_Force()
        Opti_algorithm()        
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

    else: raise ValueError ("Incorrect method chosen - it should be either Brute_Force or Optimisation")

    print("Done - probably")







   
