# =============================================================================
# materials.py
# Central database for membrane material properties and component properties.
#
# To add a new component to the master list:
#   1. Append its name to COMPONENTS below
#   2. Append its values to each property in Component_properties
#   3. Append its [Ea, P0] (or None if unavailable) to each material entry
#
# The active component subset for a given process is defined in the main script
# and passed to validate_membrane() — not here.
# =============================================================================

# -----------------------------------------------------------------------------
# Master component list
# Defines the order of all data in this file.
# The main script selects an active subset for each process.
# -----------------------------------------------------------------------------
COMPONENTS = ["CO2", "N2", "O2", "H2O"]

# -----------------------------------------------------------------------------
# Material Database
# Each material contains:
#   - Activation_Energy_Aged:  tuple of [Ea (J/mol), P0 (GPU)] per component,
#                              following COMPONENTS order. Use None if unavailable.
#   - Activation_Energy_Fresh: same, or None if the material does not age
#                              (in which case aged properties are used as fallback)
# -----------------------------------------------------------------------------
Material_Database = {
    "PIM-1": {
        "Activation_Energy_Aged": (
            [12750, 321019],    # CO2
            [25310, 2186946],   # N2
            [15770, 196980],    # O2
            [12750, 321019],    # H2O
        ),
        "Activation_Energy_Fresh": (
            [2880,  16806],     # CO2
            [16520, 226481],    # N2
            [3770,  3599],      # O2
            [2880,  16806],     # H2O
        ),
    },
    "KIM-1": {
        "Activation_Energy_Aged": (
            [510,  629],        # CO2
            [9670, 1236],       # N2
            [1500, 193],        # O2
            [510,  629],        # H2O
        ),
        "Activation_Energy_Fresh": None,    # No fresh data — falls back to aged
    },
    "BMA-TB": {
        "Activation_Energy_Aged": (
            [27060, 22325381],  # CO2
            [33110, 3396963],   # N2
            [29000, 2409785],   # O2
            [27060, 22325381],  # H2O
        ),
        "Activation_Energy_Fresh": None,
    },
    "Matrimid": {
        # Does not age — fresh and aged are identical
        "Activation_Energy_Aged": (
            [7700,  181],       # CO2
            [20300, 1045],      # N2
            [16000, 1144],      # O2
            [7700,  181],       # H2O
        ),
        "Activation_Energy_Fresh": (
            [7700,  181],       # CO2
            [20300, 1045],      # N2
            [16000, 1144],      # O2
            [7700,  181],       # H2O
        ),
    },
}

# -----------------------------------------------------------------------------
# Component Properties
# Material-independent physical properties, ordered by COMPONENTS.
# -----------------------------------------------------------------------------
Component_properties = {
    "Molar_mass": [44.009, 28.0134, 31.999, 18.01528],     # g/mol

    # Viscosity: [slope, intercept] for linear correlation with T (K) — from NIST
    "Viscosity_param": (
        [0.0479,  0.6112],      # CO2
        [0.0466,  3.8874],      # N2
        [0.0558,  3.8970],      # O2
        [0.03333, -0.23498],    # H2O
    ),
}

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def get_material(name):
    """
    Retrieve material properties by name.
    Raises a clear error if the material is not found.
    """
    if name not in Material_Database:
        available = list(Material_Database.keys())
        raise ValueError(
            f"Material '{name}' not found in database. "
            f"Available materials: {available}"
        )
    return Material_Database[name]


def validate_membrane(Membrane, components):
    """
    Given a membrane dict and the active component list for the current process,
    validates data consistency and returns a ready-to-use properties dict
    for those components only.

    Checks performed:
      - All active components exist in the master COMPONENTS list
      - Membrane permeance list length matches the active component count
      - Material exists in Material_Database
      - No None entries in activation energy data for active components
      - Component_properties arrays are consistent with master COMPONENTS list

    Returns:
      membrane_properties (dict):
        - "Material":        the material data dict (Activation_Energy_Aged/Fresh)
        - "Molar_mass":      list of molar masses, ordered as components
        - "Viscosity_param": tuple of [slope, intercept] per component, ordered as components
                             (matches numpy slicing used in mixture_visc)
    """
    # --- Check active components are known ---
    unknown = [c for c in components if c not in COMPONENTS]
    if unknown:
        raise ValueError(
            f"Unknown components {unknown}. "
            f"Master COMPONENTS list is: {COMPONENTS}"
        )

    # --- Get indices of active components in master list ---
    indices = [COMPONENTS.index(c) for c in components]

    # --- Check permeance list length ---
    if len(Membrane["Permeance"]) != len(components):
        raise ValueError(
            f"Membrane '{Membrane['Name']}' has {len(Membrane['Permeance'])} permeance values "
            f"but {len(components)} components are active: {components}"
        )

    # --- Retrieve and validate material ---
    material = get_material(Membrane["Material"])

    for key in ["Activation_Energy_Aged", "Activation_Energy_Fresh"]:
        data = material.get(key)
        if data is None:
            continue  # None means "no fresh data, fall back to aged" — acceptable
        missing = [components[j] for j, i in enumerate(indices) if data[i] is None]
        if missing:
            raise ValueError(
                f"Material '{Membrane['Material']}' is missing {key} data "
                f"for active components: {missing}"
            )

    # --- Check Component_properties consistency with master COMPONENTS ---
    for prop_name, values in Component_properties.items():
        if len(values) != len(COMPONENTS):
            raise ValueError(
                f"Component_properties['{prop_name}'] has {len(values)} entries "
                f"but master COMPONENTS has {len(COMPONENTS)}: {COMPONENTS}. "
                f"Please update materials.py."
            )

    # --- Build and return Component_properties for active components and material ---
    return {
        "Viscosity_param":        tuple(Component_properties["Viscosity_param"][i] for i in indices),  # slope & intercept per component (K) — from NIST
        "Molar_mass":             [Component_properties["Molar_mass"][i] for i in indices],             # g/mol
        "Activation_Energy_Aged": tuple(material["Activation_Energy_Aged"][i] for i in indices),       # [Ea (J/mol), P0 (GPU)]
        "Activation_Energy_Fresh": tuple(material["Activation_Energy_Fresh"][i] for i in indices)      # valid at low temperature — None if not available
                                   if material["Activation_Energy_Fresh"] is not None else None,
    }