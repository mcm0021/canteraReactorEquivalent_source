from pathlib import Path

import cantera as ct
import numpy as np

from state_conversion import atom_counts, state_to_amounts



def build_stoichiometric_matrix(gas, surface):
    """Return the joint reaction matrix and its species ordering.

    Rows: gas species, followed by surface species.
    Columns: gas reactions, followed by surface reactions.
    """
    species_names = gas.species_names + surface.species_names
    # The existing mechanism uses unique names across both phases.
    if len(set(species_names)) != len(species_names):
        raise ValueError("Species names must be unique across phases.")
    
    # Create stoichiometric matrix
    n_species = gas.n_species + surface.n_species
    n_reactions = gas.n_reactions + surface.n_reactions
    matrix = np.zeros((n_species, n_reactions))
    # print(n_species, n_reactions)

    # Construct stoichiometric matrix
    # Gas reactions affect only gas species.
    matrix[:gas.n_species, :gas.n_reactions] = (gas.product_stoich_coeffs - gas.reactant_stoich_coeffs)

    # Surface kinetics includes both gas and surface species.
    surface_matrix = (surface.product_stoich_coeffs - surface.reactant_stoich_coeffs)
    
    for row, name in enumerate(species_names):
        kinetics_row = surface.kinetics_species_index(name)
        matrix[row, gas.n_reactions:] = surface_matrix[kinetics_row, :]

    return matrix, species_names



def compute_stoichiometric_basis(matrix):
    """Return an orthonormal basis, numerical rank and singular values."""
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)

    tolerance = (
        max(matrix.shape)
        * np.finfo(float).eps
        * singular_values[0]
    )
    rank = np.count_nonzero(singular_values > tolerance)

    basis = u[:, :rank]

    return basis, rank, singular_values




def amounts_to_coordinates(initial_amounts, final_amounts, basis):
    """Convert one amount change (kmol) into independent coordinates."""
    return basis.T @ (final_amounts - initial_amounts)


def coordinates_to_amounts(initial_amounts, coordinates, basis):
    """Reconstruct one amount vector (kmol), without clipping."""
    return initial_amounts + basis @ coordinates


if __name__ == "__main__":
    mechanism = (
        Path(__file__).resolve().parent.parent
        / "mechanisms"
        / "coOxidationRhAlumina.yaml"
    )

    gas = ct.Solution(str(mechanism), "gas")
    surface = ct.Interface(str(mechanism), "surface", adjacent=[gas])

    # Example input state with 1 m^3 of gas and 1 m^2 of catalyst.
    gas.TPX = 600.0, ct.one_atm, "CO:2, O2:1"
    surface.TP = gas.TP
    amounts = state_to_amounts(
        gas.Y, gas.density, 1.0, gas.molecular_weights,
        surface.coverages, 1.0, surface.site_density,
        [species.size for species in surface.species()],
    )
    print("Elements:", sorted(set(gas.element_names) | set(surface.element_names)))
    print("Atom counts (kmol):", atom_counts(amounts, gas, surface))

    # print("Gas species:", gas.species_names)
    # print("Surface species:", surface.species_names)

    # print("Gas reactions:", gas.n_reactions)
    # print("Surface reactions:", surface.n_reactions)

    # for equation in surface.reaction_equations():
    #     print(equation)

    stoich_matrix, ordered_species_names = build_stoichiometric_matrix(gas, surface)
    print(stoich_matrix)
    print(ordered_species_names)

    basis, rank, singular_values = compute_stoichiometric_basis(stoich_matrix)

    print("Singular values:", singular_values)
    print("Rank:", rank)
    print("Basis shape:", basis.shape)

    # Each off-diagonal entry should be zero.
    # This yields the largest absolute error.
    orthogonality_error = np.max(
        np.abs(basis.T @ basis - np.eye(rank))
    )

    # This error represents whether the basis can represent
    # all reaction columns without losing information.
    span_error = np.max(
        np.abs(stoich_matrix - basis @ (basis.T @ stoich_matrix))
    )
    # Both errors should be near zero (floating point precision)
    print("Orthogonality error:", orthogonality_error)
    print("Span error:", span_error)
