import math  
import numpy as np  
from scipy.optimize import differential_evolution  
from Ferrari_paper_opti import Ferrari_Paper_Main  
import pandas as pd  
import tqdm
import os

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

# Generate brute force parameters using NumPy for better performance


Opti_Param = {
    "Recycling_Ratio" : [0, 1],  # Recycling ratio range for the process
    "Q_A_ratio_2" : [1, 10], # Flow/Area ratio for the second stage
    #"P_up_2" : [3, 10]  # Upper pressure range for the second stage in bar    
    }

# In case of brute force
number_evaluation = 100  # Number of evaluations for the brute force method

Brute_Force_Param = [  
   {  
       "Recycling_Ratio": np.random.uniform(*Opti_Param["Recycling_Ratio"]),  
       "Q_A_ratio_2": np.random.uniform(*Opti_Param["Q_A_ratio_2"]),  
       #"P_up_2": np.random.uniform(*Opti_Param["P_up_2"])  
   }  
   for _ in range(number_evaluation)  
]

#--------------------------#
#--- Default Parameters ---#
#--------------------------#

Membrane_1 = {
"Name": 'Membrane_1',
"Solving_Method": 'CC',                     # 'CC' or 'CO' - CC is for counter-current, CO is for co-current
"Temperature": 30+273.15,                   # Kelvin
"Pressure_Feed": 3,                         # bar
"Pressure_Permeate": 1,                     # bar
"Q_A_ratio": 0.8,                           # ratio of the membrane feed flowrate to its area (in m3(stp)/m2.hr)
"Permeance": [1000, 23, 80, 1000],          # GPU
"Pressure_Drop": True,
}

Membrane_2 = {
"Name": 'Membrane_2',
"Solving_Method": 'CC',                   
"Temperature": 30+273.15,                   
"Pressure_Feed": 3,                       
"Pressure_Permeate": 1,                  
"Q_A_ratio": 6,                           # ratio of the membrane feed flowrate to its area (in m3(stp)/m2.hr)
"Permeance": [1000, 23, 80, 1000],        # GPU
"Pressure_Drop": True,
}

Process_param = {
"Recycling_Ratio" : 0.9, # Ratio of the retentate flow from Membrane 2 that is recycled back to Membrane 1 feed    
"Target_Purity" : 0.95, # Target purity of the permeate from Membrane 2
"Target_Recovery" : 0.9, # Target recovery from Membrane 2 - for now not a hard limit, but a target to be achieved
"Replacement_rate": 4, # Replacement rate of the membranes (in yr)
"Operating_hours": 8000, # Operating hours per year
"Lifetime": 20, # Lifetime of the membranes (in yr)
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
  "CO2_emission",  # CO2 emissions from the process in tonnes per year  
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
df = pd.DataFrame(index=range(number_evaluation), columns=columns)

for i in tqdm.tqdm(range(number_evaluation)):
    Process_param["Recycling_Ratio"] = Brute_Force_Param[i]["Recycling_Ratio"]
    Membrane_2["Q_A_ratio"] = Brute_Force_Param[i]["Q_A_ratio_2"]
    #Membrane_2["Pressure_Feed"] = Brute_Force_Param[i]["P_up_2"]

    Parameters = Membrane_1, Membrane_2, Process_param, Component_properties, Fibre_Dimensions, J

    # Run the optimisation algorithm
    Economics = Ferrari_Paper_Main(Parameters)

    if isinstance(Economics, dict):
        df.loc[i] = list(Brute_Force_Param[i].values()) + list(Economics.values())
    else:
        df.loc[i] = list(Brute_Force_Param[i].values()) + [Economics] + [0] * (len(columns) - len(Brute_Force_Param[i]) - 1)



desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

# Define the file path
file_path = os.path.join(desktop_path, 'example.csv')

# Save the DataFrame to a CSV file  
df.to_csv("optimisation_results.csv", index=False)



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
plt.show()














