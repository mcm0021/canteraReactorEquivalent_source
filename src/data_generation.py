import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import qmc

import cantera as ct

from src.reactor import LocalThermalEquilibriumReactor
from src.validation import validate_inputs


def generate_input_data(
        num_samples, 
        scalar_input_size,
        species_size,
        coverages_size,
        temp_bounds, 
        press_bounds,
        porosity_bounds,
        adv_time_log_bounds,
        cat_surf_log_bounds
): 

    # Pure states
    pure_mass_fractions = np.eye(species_size)
    pure_surface_coverages = np.eye(coverages_size)

    # Random states
    rng = np.random.default_rng()

    random_mass_fractions = rng.dirichlet(np.ones(species_size), size=num_samples - species_size)
    random_surface_coverages = rng.dirichlet(np.ones(coverages_size), size=num_samples - coverages_size)


    # Block 1: Pure gas + random surface
    # Block 2: Random gas + pure surface
    # Block 3: Random gas + random surface
    mass_fractions = np.vstack([
        pure_mass_fractions, 
        random_mass_fractions
    ])
    
    surface_coverages = np.vstack([
        random_surface_coverages[:species_size], 
        pure_surface_coverages, 
        random_surface_coverages[species_size:]
    ])

    # Sample scalar inputs using Latin Hypercube Sampling
    sampler = qmc.LatinHypercube(d=scalar_input_size)
    lhs_samples = sampler.random(n=num_samples)

    temperature = temp_bounds[0] + lhs_samples[:, 0] * (temp_bounds[1] - temp_bounds[0])
    pressure = press_bounds[0] + lhs_samples[:, 1] * (press_bounds[1] - press_bounds[0])
    porosity = porosity_bounds[0] + lhs_samples[:, 2] * (porosity_bounds[1] - porosity_bounds[0])
    
    # Logarithmic bounds (sample exponents uniformly)
    advance_time = 10 ** (adv_time_log_bounds[0] + lhs_samples[:, 3] * (adv_time_log_bounds[1] - adv_time_log_bounds[0]))
    
    catalytic_surface = 10 ** (cat_surf_log_bounds[0] + lhs_samples[:, 4] * (cat_surf_log_bounds[1] - cat_surf_log_bounds[0]))

    data = np.column_stack([
        temperature,
        pressure,
        porosity,
        advance_time,
        catalytic_surface,
        mass_fractions,
        surface_coverages
    ])

    mf_cols = [f"mass_frac_{i+1}" for i in range(species_size)]
    sc_cols = [f"surf_cov_{i+1}" for i in range(coverages_size)]

    cols = cols = ["temperature", "pressure", "porosity", "advance_time", "catalytic_surface_area"] + mf_cols + sc_cols
    df = pd.DataFrame(data, columns=cols)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  

    return data, df

def generate_output_data(
   input_data,      
): 
    """
    input_data: np.ndarray of shape (num_samples, scalar_input_size + species_size + coverages_size)
    """
    mechanism = Path(__file__).parent.parent / "mechanisms" / "coOxidationRhAlumina.yaml"
    gas = ct.Solution(mechanism, "gas")
    solid = ct.Solution(mechanism, "alumina")
    surface = ct.Interface(mechanism, "surface", adjacent=[gas])

    gas_species_order = gas.species_names
    surface_species_order = surface.species_names

    minimum_temperature = max(gas.min_temp, solid.min_temp, surface.min_temp)
    maximum_temperature = min(gas.max_temp, solid.max_temp, surface.max_temp)


    output_data = []
    output_data_frame = pd.DataFrame(columns=["temperature", "pressure"])
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
        validate_inputs(
            sample,
            minimum_temperature=minimum_temperature,
            maximum_temperature=maximum_temperature,
            gas_species_order=gas_species_order,
            surface_species_order=surface_species_order
        )

        gas.TPY = (
            sample["temperature"],
            sample["pressure"],
            sample["mass_fractions"]
        )

        surface.coverages = sample["surface_coverages"]

        cell_volume = 1.1e-6
        gas_volume = sample["porosity"] * cell_volume
        solid_volume = (1.0 - sample["porosity"]) * cell_volume
        catalytic_surface_area = (
            sample["catalytic_surface_area_per_cell_volume"] * cell_volume
        )

        reactor = LocalThermalEquilibriumReactor(
            gas,
            solid,
            solid_volume,
            clone=False,
        )
        reactor.volume = gas_volume
        reactor_surface = ct.ReactorSurface(
            surface,
            r=reactor,
            A=catalytic_surface_area,
            clone=False,
        )

        ct.ReactorNet([reactor]).advance(sample["advance_time"])

        output_state = {
            "temperature": reactor.T,
            "pressure": reactor.phase.P,
            "mass_fractions": reactor.phase.Y.tolist(),
            "surface_coverages": reactor_surface.coverages.tolist(),
        }

        output_data.append([
            output_state["temperature"],
            output_state["pressure"],
            *output_state["mass_fractions"],
            *output_state["surface_coverages"]
        ])
        output_data_frame = pd.concat([output_data_frame, pd.DataFrame([output_state])], ignore_index=True)
        
    return output_data, output_data_frame

def generate_input_output_data(
        num_samples, 
        scalar_input_size, 
        species_size, 
        coverages_size, 
        temp_bounds, 
        press_bounds,
        porosity_bounds,
        adv_time_log_bounds,
        cat_surf_log_bounds):
    
    input_data, input_data_frame = generate_input_data(
        num_samples, 
        scalar_input_size, 
        species_size, 
        coverages_size, 
        temp_bounds, 
        press_bounds,
        porosity_bounds,
        adv_time_log_bounds,
        cat_surf_log_bounds            
    )

    output_data, output_data_frame = generate_output_data(input_data)
    
    return input_data, input_data_frame, output_data, output_data_frame