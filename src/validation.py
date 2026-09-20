from math import isclose


def validate_composition(name, values, species_order):
    """Reject invalid vectors instead of letting Cantera normalize them silently."""
    if len(values) != len(species_order):
        raise ValueError(
            f"{name} must contain {len(species_order)} entries in this order: "
            f"{species_order}"
        )
    if any(value < 0.0 for value in values):
        raise ValueError(f"{name} entries must be nonnegative")
    if not isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{name} entries must sum to one")


def validate_inputs(
    input_state,
    *,
    minimum_temperature,
    maximum_temperature,
    gas_species_order,
    surface_species_order,
):
    """Validate one complete set of sampled reactor inputs."""
    validate_composition(
        "mass_fractions",
        input_state["mass_fractions"],
        gas_species_order,
    )
    validate_composition(
        "surface_coverages",
        input_state["surface_coverages"],
        surface_species_order,
    )

    # if not minimum_temperature <= input_state["temperature"] <= maximum_temperature:
    #     raise ValueError(
    #         "temperature must be within the common thermodynamic range "
    #         f"[{minimum_temperature}, {maximum_temperature}] kelvin"
    #     )
    if input_state["pressure"] <= 0.0:
        raise ValueError("pressure must be positive")
    if input_state["advance_time"] < 0.0:
        raise ValueError("advance_time must be nonnegative")
    if not 0.0 < input_state["porosity"] < 1.0:
        raise ValueError("porosity must be strictly between zero and one")
    if input_state["catalytic_surface_area_per_cell_volume"] < 0.0:
        raise ValueError(
            "catalytic_surface_area_per_cell_volume must be nonnegative"
        )
