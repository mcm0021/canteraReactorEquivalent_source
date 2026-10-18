import cantera as ct
import numpy as np

def initialize_variables(mechanism):
    gas = ct.Solution(mechanism, "gas")
    solid = ct.Solution(mechanism, "alumina")
    surface = ct.Interface(mechanism, "surface", adjacent=[gas])
    return gas, solid, surface


def get_reactor_constants(gas, solid, surface, porosity,
                          catalytic_surface_area_per_cell_volume, cell_volume):
    W = gas.molecular_weights
    Gamma = surface.site_density
    sizes = np.array([s.size for s in surface.species()])
    Vgas = porosity * cell_volume
    A = catalytic_surface_area_per_cell_volume * cell_volume
    return W, Gamma, sizes, Vgas, A


def initialize_reactor(gas,
                       solid,
                       surface,
                       temperature,
                       pressure,
                       porosity,
                       catalytic_surface_area_per_cell_volume,
                       mass_fractions,
                       surface_coverages,
                       cell_volume=1.1e-6):

    gas.TPY = (temperature, pressure, mass_fractions)

    surface.coverages = surface_coverages
    gas_volume = porosity * cell_volume
    solid_volume = (1.0 - porosity) * cell_volume
    catalytic_surface_area = (catalytic_surface_area_per_cell_volume *
                              cell_volume)
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
    return reactor, reactor_surface


def advance_reactor(reactor, reactor_surface, time_step):
    ct.ReactorNet([reactor]).advance(time_step)
    return {
        "temperature": reactor.T,
        "pressure": reactor.phase.P,
        "mass_fractions": reactor.phase.Y.tolist(),
        "surface_coverages": reactor_surface.coverages.tolist(),
    }


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
        left_hand_side[temperature_equation_index] += (self.solid_mass *
                                                       self.solid.cv_mass)
