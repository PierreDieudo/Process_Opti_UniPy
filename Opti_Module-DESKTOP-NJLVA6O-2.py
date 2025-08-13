import math  
import numpy as np  
from scipy.optimize import differential_evolution  
from Ferrari_paper_opti import Ferrari_Paper_Main  
import pandas as pd  
import tqdm
import os
import time

"""
This module is used to run the optimisation for the process simulation, would it be by brute force or using an optimisation algorithm.

A set of parameters from the process are chosen and given bounds for the optimisation algorithm to work with.

The aim is to minimise a scaler from the process simulation function, which usually will be associated with the cost of the process.

The brute force method can be used for up to three parameters and will be mainly used to validate the optimisation algorithm.

Most debugging and test messages are removed from this solution ; manual checks can be done on another project.

"""



#-------------------------------#
#--- Optimisation Parameters ---#
#-------------------------------#
Method = "Optimisation" # Method is either by Brute_Force or Optimisation


# Generate brute force parameters
Opti_Param = {
    "Recycling_Ratio" : [0, 1],  # Recycling ratio range for the process
    "Q_A_ratio_2" : [0.2, 10], # Flow/Area ratio for the second stage
    "P_up_2" : [2, 15],  # Upper pressure range for the second stage in bar    
    "Q_A_ratio_1" : [0.5, 20], # Flow/Area ratio for the first stage"
    "P_up_1" : [2, 15],  # Upper pressure range for the first stage in bar"
    "Temperature_1" : [-40, 70],  # Temperature range in Celcius
    "Temperature_2" : [-40, 70],  # Temperature range in Celcius
    }


# In case of brute force
number_evaluation = 1000 * len(Opti_Param)  # Number of evaluations for the brute force method

Brute_Force_Param = [  
   {  
       "Recycling_Ratio": np.random.uniform(*Opti_Param["Recycling_Ratio"]),  
       "Q_A_ratio_2": np.random.uniform(*Opti_Param["Q_A_ratio_2"]),  
       "P_up_2": np.random.uniform(*Opti_Param["P_up_2"]),
       "Q_A_ratio_1": np.random.uniform(*Opti_Param["Q_A_ratio_1"]),
       "P_up_1": np.random.uniform(*Opti_Param["P_up_1"]),
       "Temperature_1": np.random.uniform(*Opti_Param["Temperature_1"]),
       "Temperature_2": np.random.uniform(*Opti_Param["Temperature_2"]),
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
}

Fibre_Dimensions = {
"D_in" : 150 * 1e-6, # Inner diameter in m (from um)
"D_out" : 300 * 1e-6, # Outer diameter in m (from um)
}
  
J = len(Membrane_1["Permeance"]) #number of components

#-------------------------------#
#----- Optimisation Method -----#
#-------------------------------#


def Opti_algorithm():
    
    # Define the bounds for the parameters to be optimised
    bounds = [  
        Opti_Param["Recycling_Ratio"],  # Recycling ratio range for the process
        Opti_Param["Q_A_ratio_2"],  # Flow/Area ratio for the second stage
        Opti_Param["P_up_2"],  # Upper pressure range for the second stage in bar    
        Opti_Param["Q_A_ratio_1"],  # Flow/Area ratio for the first stage
        Opti_Param["P_up_1"],  # Upper pressure range for the first stage in bar
        Opti_Param["Temperature_1"],  # Temperature range in Celcius
        Opti_Param["Temperature_2"],  # Temperature range in Celcius
    ]
    
    # Define the objective function to be minimised
    def objective_function(params):
        Process_param["Recycling_Ratio"] = params[0]
        Membrane_2["Q_A_ratio"] = params[1]
        Membrane_2["Pressure_Feed"] = params[2]
        Membrane_1["Q_A_ratio"] = params[3]
        Membrane_1["Pressure_Feed"] = params[4]
        Membrane_1["Temperature"] = params[5] + 273.15
        Membrane_2["Temperature"] = params[6] + 273.15
        Parameters = Membrane_1, Membrane_2, Process_param, Component_properties, Fibre_Dimensions, J
        Economics = Ferrari_Paper_Main(Parameters)
        if isinstance(Economics, dict):
            return Economics['Evaluation']  # Return the evaluation metric to be minimised
        else:
            return 1e9  # If simulation fails, return a large number to avoid this solution


    # Callback function to track progress
    def callback(xk, convergence):
        elapsed_time = time.time() - callback.start_time
        print(f"Iteration: {callback.n_iter}, Elapsed Time: {elapsed_time:.2f}s, Best Solution: {xk}, Convergence: {convergence:.4f}")
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
        Membrane_2["Q_A_ratio"] = result.x[1]
        Membrane_2["Pressure_Feed"] = result.x[2]
        Membrane_1["Q_A_ratio"] = result.x[3]
        Membrane_1["Pressure_Feed"] = result.x[4]
        Membrane_1["Temperature"] = result.x[5] + 273.15
        Membrane_2["Temperature"] = result.x[6] + 273.15
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
    filename = 'optimisation_results_laptop_5param_new_penalty.txt'
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
        Membrane_2["Q_A_ratio"] = Brute_Force_Param[i]["Q_A_ratio_2"]
        Membrane_2["Pressure_Feed"] = Brute_Force_Param[i]["P_up_2"]
        Membrane_1["Q_A_ratio"] = Brute_Force_Param[i]["Q_A_ratio_1"]
        Membrane_1["Pressure_Feed"] = Brute_Force_Param[i]["P_up_1"]
        Membrane_1["Temperature"] = Brute_Force_Param[i]["Temperature_1"] + 273.15
        Membrane_2["Temperature"] = Brute_Force_Param[i]["Temperature_2"] + 273.15

        Parameters = Membrane_1, Membrane_2, Process_param, Component_properties, Fibre_Dimensions, J

        # Run the optimisation algorithm
        Economics = Ferrari_Paper_Main(Parameters)

        if isinstance(Economics, dict):
            row = list(Brute_Force_Param[i].values()) + list(Economics.values())
            rows.append(row)
        else: Invalid += 1

    # Convert the list of rows to a DataFrame
    df = pd.DataFrame(rows, columns=columns)

    filename_df = "bruteforce_7param_7000sets_360GPU.csv"

    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

    # Define the file path correctly using the variable
    file_path = filename_df

    # Save the DataFrame to a CSV file on the desktop
    df.to_csv(file_path, index=False)


    print(f"{number_evaluation - Invalid} sets of data collected in {file_path} over {number_evaluation} evaluations total ({(number_evaluation - Invalid)/number_evaluation:.2%}). ")

    '''
    # ================================================
    # ====== 3D SCATTER PLOTS OF RESULTS  ============
    # ================================================
    import matplotlib.pyplot as plt

    # Choose variables to plot
    econ_vars = ["Evaluation", "Purity", "Recovery", "TAC_CC"]

    # Ensure the parameter columns exist
    if "Param_1" not in df.columns or "Param_2" not in df.columns:
        # If columns are unnamed (dict order), rename appropriately
        df.rename(columns={df.columns[0]: "Param_1", df.columns[1]: "Param_2"}, inplace=True)

    # Create 2x2 grid of 3D scatter plots
    fig = plt.figure(figsize=(16, 12))
    for i, var in enumerate(econ_vars, start=1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        sc = ax.scatter(
            df["Param_1"],
            df["Param_2"],
            df[var],
            c=df[var],
            cmap='viridis',
            s=30,
            alpha=0.8
        )
        ax.set_xlabel("Param_0")
        ax.set_ylabel("Param_1")
        ax.set_zlabel(var)
        ax.set_title(f"Param_0 vs Param_1 vs {var}")
        fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10, label=var)


    plt.tight_layout()

    filename_plot = "3D_Scatter_Plots.png"
    #file_path = os.path.join(desktop_path, filename_plot)
    file_path = filename_plot
    plt.savefig(file_path, dpi=300, bbox_inches='tight')

    plt.show()
'''

if Method == "Brute_Force":
    Brute_Force()
elif Method == "Optimisation":
    Opti_algorithm()
else: raise ValueError ("Incorrect method chosen - it should be either Brute_Force or Optimisation")

print("Done - probably")








