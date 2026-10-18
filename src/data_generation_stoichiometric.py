import numpy as np
import cantera as ct
import pandas as pd
from pathlib import Path

from src.reactor import initialize_variables, initialize_reactor, get_reactor_constants
from src.state_conversion import state_to_amounts
from src.stoichiometry import amounts_to_coordinates, build_stoichiometric_matrix, compute_stoichiometric_basis


def run_reactor(input_data):
    """
    input_data: pandas dataframe 
    """
    mechanism = Path(
        __file__).parent.parent / "mechanisms" / "coOxidationRhAlumina.yaml"
    gas, solid, surface = initialize_variables(mechanism)

    gas_species_order = gas.species_names
    surface_species_order = surface.species_names

    minimum_temperature = max(gas.min_temp, solid.min_temp, surface.min_temp)
    maximum_temperature = min(gas.max_temp, solid.max_temp, surface.max_temp)

    S, species_names = build_stoichiometric_matrix(gas, surface)
    Q, rank, _ = compute_stoichiometric_basis(S)  # Q: (9, 5), rank: 5

    output_data = np.empty((0, rank + 1))  # + 1 for delta T
    output_data_frame = pd.DataFrame(
        columns=["W", "Gamma", "sizes", "Vgas", "A", "delta_T"])

    for current in input_data:
        sample = {
            "temperature": current[0],
            "pressure": current[1],
            "porosity": current[2],
            "advance_time": current[3],
            "catalytic_surface_area_per_cell_volume": current[4],
            "mass_fractions": current[5:5 + len(gas_species_order)].tolist(),
            "surface_coverages": current[5 + len(gas_species_order):].tolist(),
        }
        reactor, reactor_surface = initialize_reactor(
            gas, solid, surface, sample["temperature"], sample["pressure"],
            sample["porosity"],
            sample["catalytic_surface_area_per_cell_volume"],
            sample["mass_fractions"], sample["surface_coverages"])

        W, Gamma, sizes, Vgas, A = get_reactor_constants(
            gas, solid, surface, sample["porosity"],
            sample["catalytic_surface_area_per_cell_volume"], 1.1e-6)

        def read_amounts():
            return state_to_amounts(
                reactor.phase.Y,
                reactor.phase.density,
                Vgas,
                W,
                reactor_surface.coverages,
                A,
                Gamma,
                sizes,
            )

        n0 = read_amounts()
        T0 = reactor.T

        ct.ReactorNet([reactor]).advance(sample["advance_time"])

        n1 = read_amounts()
        targets = np.r_[amounts_to_coordinates(n0, n1, Q), reactor.T - T0]
        output_data = np.append(output_data,
                                np.reshape(targets, (1, -1)),
                                axis=0)
        output_data_frame = pd.concat([
            output_data_frame,
            pd.DataFrame([{
                "W": W,
                "Gamma": Gamma,
                "sizes": sizes,
                "Vgas": Vgas,
                "A": A,
                "delta_T": reactor.T - T0
            }])
        ],
                                      ignore_index=True)

    return output_data, output_data_frame
