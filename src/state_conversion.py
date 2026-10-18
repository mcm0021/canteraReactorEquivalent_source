import numpy as np


def atom_counts(amounts, gas, surface):
    """Return elemental amounts in kmol as a list in alphabetical element order.

    ``amounts`` contains species amounts in kmol, gas species first, then
    surface species (as returned by ``state_to_amounts``). ``gas`` and
    ``surface`` supply the Cantera species definitions; their current state
    is not used. The ordering is always
    ``sorted(set(gas.element_names) | set(surface.element_names))``, including
    elements with zero inventory. Values are kmol of atoms and include both
    phases.
    """
    amounts = np.asarray(amounts, dtype=float)
    species = gas.species() + surface.species()
    if amounts.shape != (len(species),):
        raise ValueError('Expected one amount per species, gas first, then surface')
    elements = sorted(set(gas.element_names) | set(surface.element_names))
    composition = np.array([
        [sp.composition.get(element, 0.0) for sp in species]
        for element in elements
    ])
    return (composition @ amounts).tolist()

# Convert to amounts as the amounts are conserved (not the mass fraction or coverage)
def state_to_amounts(
    mass_fractions,
    gas_density,
    gas_volume,
    molecular_weights,
    coverages,
    catalytic_area,
    site_density,
    site_sizes,
):
    """Return species amounts in kmol: gas first, then surface."""
    gas_amounts = (
        gas_density * gas_volume
        * np.asarray(mass_fractions)
        / np.asarray(molecular_weights)
    )

    surface_amounts = (
        catalytic_area * site_density
        * np.asarray(coverages)
        / np.asarray(site_sizes)
    )

    return np.concatenate([gas_amounts, surface_amounts])


# Convert amounts to the state of interest
def amounts_to_state(
    amounts,
    gas_volume,
    molecular_weights,
    catalytic_area,
    site_density,
    site_sizes,
):
    """Convert species amounts in kmol into Y, coverages and gas density."""
    amounts = np.asarray(amounts)
    molecular_weights = np.asarray(molecular_weights)
    site_sizes = np.asarray(site_sizes)

    n_gas_species = len(molecular_weights)
    gas_amounts = amounts[:n_gas_species]
    surface_amounts = amounts[n_gas_species:]

    gas_masses = gas_amounts * molecular_weights
    total_gas_mass = gas_masses.sum()

    if total_gas_mass <= 0:
        raise ValueError("Total gas mass must be positive.")

    mass_fractions = gas_masses / total_gas_mass
    gas_density = total_gas_mass / gas_volume

    coverages = (
        surface_amounts * site_sizes
        / (catalytic_area * site_density)
    )

    return mass_fractions, coverages, gas_density
