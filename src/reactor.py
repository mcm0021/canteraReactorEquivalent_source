import cantera as ct
import numpy as np


class LocalThermalEquilibriumReactor(ct.ExtensibleIdealGasReactor):
    """Constant-volume gas reactor with solid and surface thermal mass."""

    def __init__(self, gas, solid, solid_volume, **kwargs):
        super().__init__(gas, **kwargs)
        self.solid = solid
        self.solid_mass = solid.density * solid_volume

    def after_eval(self, _time, left_hand_side, right_hand_side):
        """Add solid and catalytic-surface energy to Cantera's equations."""
        temperature_equation_index = self.component_index("temperature")
        self.solid.TP = self.phase.TP

        # Add the inert solid heat capacity to the left-hand side (LHS).
        left_hand_side = np.asarray(left_hand_side)
        right_hand_side = np.asarray(right_hand_side)
        left_hand_side[temperature_equation_index] += (self.solid_mass * self.solid.cv_mass)

        surface_equation_offset = self.phase.n_species + 3
        for reactor_surface in self.surfaces:
            surface_phase = reactor_surface.phase
            surface_coverages = reactor_surface.coverages
            surface_species_count = surface_phase.n_species
            surface_equation_slice = slice(surface_equation_offset, surface_equation_offset + surface_species_count,)
            species_site_occupancies = np.array(
                [
                    surface_phase.species(species_index).size
                    for species_index in range(surface_species_count)
                ]
            )

            total_site_amount = reactor_surface.area * surface_phase.site_density
            surface_species_amounts = (
                total_site_amount * surface_coverages / species_site_occupancies
            )
            surface_species_amount_rates = (
                total_site_amount
                * right_hand_side[surface_equation_slice]
                / left_hand_side[surface_equation_slice]
                / species_site_occupancies
            )

            left_hand_side[temperature_equation_index] += np.dot(
                surface_species_amounts,
                surface_phase.partial_molar_cp,
            )
            right_hand_side[temperature_equation_index] -= np.dot(
                surface_species_amount_rates,
                surface_phase.partial_molar_enthalpies,
            )
            surface_equation_offset += surface_species_count
