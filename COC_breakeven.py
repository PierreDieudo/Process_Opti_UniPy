from scipy.optimize import brentq

def find_breakeven(Process_param, Breakeven_params_base, Breakeven_params_capt):
    
    print(Breakeven_params_capt)
    """
    Computes the breakeven cost of clinker (EUR/t) for both cases,
    with and without carbon tax.

    Args:
        Process_param        : dict -- process and plant parameters
        Breakeven_params_base: dict -- keys: TPC_cem, opex, carbon_tax (EUR/yr)
        Breakeven_params_capt: dict -- keys: TPC_cem, TPC_capt, opex, carbon_tax (EUR/yr)

    Returns:
        COC_base_0 : float -- cost of clinker, base plant, no carbon tax (EUR/t)
        COC_CCS_0  : float -- cost of clinker, full plant, no carbon tax (EUR/t)
        COC_base   : float -- cost of clinker, base plant, with carbon tax (EUR/t)
        COC_CCS    : float -- cost of clinker, full plant, with carbon tax (EUR/t)
    """

    # -- Unpack process parameters --------------------------------------------
    lifetime       = Process_param["Lifetime"]
    clinker_annual = Process_param["Base_Clinker_Production"]   # t/yr

    # -- Fixed financial assumptions ------------------------------------------
    construction_time = 3
    dcf_rate          = 0.08

    def npv(selling_price, TPC_cem, TPC_capt, carbon_tax, opex):
        TPC_cem_M    = TPC_cem    / 1e6
        TPC_capt_M   = TPC_capt  / 1e6
        carbon_tax_M = carbon_tax / 1e6
        opex_M       = opex       / 1e6
        cumulative   = 0.0

        for i in range(-construction_time + 1, lifetime + 1):

            # -- Construction fractions (40/30/30 split) ----------------------
            if i <= 0:
                pct_cem  = 0.5 if i in (-1, 0) else 0.0
                pct_capt = 0.4 if i == -2 else 0.3

            # -- Expenses (MEur) ----------------------------------------------
            if i <= 0:
                expense_M = pct_cem * TPC_cem_M + pct_capt * TPC_capt_M
            else:
                expense_M = opex_M + carbon_tax_M

            # -- Revenue (MEur) -----------------------------------------------
            revenue_M = (selling_price * clinker_annual / 1e6) if i > 0 else 0.0

            # -- Net cash flow (depreciation = 0, taxes = 0) ------------------
            net_cf_M = revenue_M - expense_M

            # -- Discount factor ----------------------------------------------
            factor = (1 + dcf_rate) ** (-i)

            cumulative += net_cf_M * factor

        return cumulative

    # -- Base case, no carbon tax ---------------------------------------------
    COC_base_0 = brentq(
        npv, 0, 1e9,
        args=(
            Breakeven_params_base["TPC_cem"],
            0.0,
            0.0,
            Breakeven_params_base["opex"]
        )
    )

    # -- Full case, no carbon tax ---------------------------------------------
    COC_CCS_0 = brentq(
        npv, 0, 1e9,
        args=(
            Breakeven_params_capt["TPC_cem"],
            Breakeven_params_capt["TPC_capt"],
            0.0,
            Breakeven_params_capt["opex"]
        )
    )

    # -- Base case, with carbon tax -------------------------------------------
    COC_base = brentq(
        npv, 0, 1e9,
        args=(
            Breakeven_params_base["TPC_cem"],
            0.0,
            Breakeven_params_base["carbon_tax"],
            Breakeven_params_base["opex"]
        )
    )

    # -- Full case, with carbon tax -------------------------------------------
    COC_CCS = brentq(
        npv, 0, 1e9,
        args=(
            Breakeven_params_capt["TPC_cem"],
            Breakeven_params_capt["TPC_capt"],
            Breakeven_params_capt["carbon_tax"],
            Breakeven_params_capt["opex"]
        )
    )

    return COC_base, COC_CCS, COC_base_0, COC_CCS_0