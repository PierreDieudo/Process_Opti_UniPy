
from math import e
import numpy as np
import pandas as pd
import os
from PIL import Image
from Hub import Hub_Connector
from UNISIMConnect import UNISIMConnector

class ConvergenceError(Exception):
    pass

def Ferrari_Paper_Main(Param):
    
    Membrane_1 , Membrane_2, Process_param, Component_properties, Fibre_Dimensions, J =  Param

    #File paths:
    directory = 'C:\\Users\\s1854031\\OneDrive - University of Edinburgh\\Python\\Cement_Plant_2021\\' #input file path here.
    filename = 'Cement_4Comp_FerrariPaper_Flash.usc' #input file name here.
    unisim_path = os.path.join(directory, filename)

    with UNISIMConnector(unisim_path, close_on_completion=False) as unisim:
    
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
            valid_train_indices = [i for i, train in enumerate(Train_data) if train[0] is not None and train[0] > 0]
            if not valid_train_indices:
                raise ValueError("No valid trains found with positive compressor duty.")
        
            # Find the index of the train with the lowest compressor duty in Train_data
            lowest_duty_train_index = min(range(len(Train_data)), key=lambda i: Train_data[i][0] if Train_data[i][0] is not None and Train_data[i][0] > 0 else float('inf')) #return inf for 0 or undefined
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

        max_iter = 300
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
                valid_trains = [(i, train) for i, train in enumerate(train_data) if train[0] is not None and train[0] > 0]
                if not valid_trains:
                    raise ValueError("No valid trains found with positive compressor duty.")
                # Find the train with the lowest compressor duty
                lowest_duty_train_index, lowest_duty_train = min(valid_trains, key=lambda x: x[1][0])
                lowest_duty_train.append(lowest_duty_train_index)
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
        Ret1_savings = [Duties.get_cell_value('H21'), Duties.get_cell_value('I21')] # Get the retentate 1 expander (kW) and heater duty (kJ/hr)
        Ret2_savings = [Duties.get_cell_value('H24'), Duties.get_cell_value('I24')]
    
    
        Process_specs = {
            "Feed": Feed,
            "Membrane_1": Membrane_1,
            "Membrane_2": Membrane_2,
            "Pre_cond_1": Train1,
            "Pre_cond_2": Train2,
            "Comp_train": Liquefaction,
            "Ret1_Savings": Ret1_savings,
            "Ret2_Savings": Ret2_savings,
            "Process_param": Process_param,
            "Purity": (results_2[1][0]/(1-results_2[1][-1])),
            "Recovery": results_2[1][0]*results_2[3]/(Feed["Feed_Composition"][0]*Feed["Feed_Flow"])
        }


        from Costing import Costing
        Economics = Costing(Process_specs, Process_param, Membrane_1, Membrane_2)


    return(Economics)


   
