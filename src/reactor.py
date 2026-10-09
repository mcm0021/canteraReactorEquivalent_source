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
