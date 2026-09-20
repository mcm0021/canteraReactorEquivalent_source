from pathlib import Path
from pprint import pformat

import cantera as ct

from src.reactor import LocalThermalEquilibriumReactor
from src.validation import validate_inputs

mechanism = Path(__file__).parent / "mechanisms" / "coOxidationRhAlumina.yaml"
gas = ct.Solution(mechanism, "gas")
solid = ct.Solution(mechanism, "alumina")
surface = ct.Interface(mechanism, "surface", adjacent=[gas])


gas_species_order = gas.species_names
surface_species_order = surface.species_names

# Input features (these should be trained)
input_state = {
    "temperature": 900.0,
    "pressure": ct.one_atm,
    "mass_fractions": [0.1, 0.0, 0.1, 0.8],
    "surface_coverages": [1.0, 0.0, 0.0, 0.0, 0.0],
    "advance_time": 1.0e-4,
    "porosity": 0.3,
    "catalytic_surface_area_per_cell_volume": 1.0e3,
}

# Validation of input
minimum_temperature = max(gas.min_temp, solid.min_temp, surface.min_temp)
maximum_temperature = min(gas.max_temp, solid.max_temp, surface.max_temp)
validate_inputs(
    input_state,
    minimum_temperature=minimum_temperature,
    maximum_temperature=maximum_temperature,
    gas_species_order=gas_species_order,
    surface_species_order=surface_species_order,
)

# Calculation of needed reactor inputs
gas.TPY = (
    input_state["temperature"],
    input_state["pressure"],
    input_state["mass_fractions"],
)
surface.coverages = input_state["surface_coverages"]

cell_volume = 1.1e-6
gas_volume = input_state["porosity"] * cell_volume
solid_volume = (1.0 - input_state["porosity"]) * cell_volume
catalytic_surface_area = (
    input_state["catalytic_surface_area_per_cell_volume"] * cell_volume
)

# Reactor setup
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

# Calculate new updated state
ct.ReactorNet([reactor]).advance(input_state["advance_time"])

# Get output features (these should be trained)
output_state = {
    "temperature": reactor.T,
    "pressure": reactor.phase.P,
    "mass_fractions": reactor.phase.Y.tolist(),
    "surface_coverages": reactor_surface.coverages.tolist(),
}

print(
    "\n"
    "=== Reactor State Transition ===\n\n"
    "Species order\n"
    f"  Gas:     {gas_species_order}\n"
    f"  Surface: {surface_species_order}\n\n"
    "Input state\n"
    f"{pformat(input_state, sort_dicts=False, indent=2)}\n\n"
    "Output state\n"
    f"{pformat(output_state, sort_dicts=False, indent=2)}"
)
