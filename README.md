# Cantera Reactor Equivalent

This project generates Cantera reactor data, trains a stoichiometric neural
network, and uses the trained network to predict the reactor state.

## 1. Generate data

The `generate_data.py` script currently generates `NUM_SAMPLES` samples with an 80/20 training/validation
split and saves the NumPy files at a given `PATH_PREFIX`. Important files are:

- `input_train.npy` and `output_train.npy`: `[temperature, pressure, porosity, advance_time, area, mass_fractions, surface_coverages]`
- `input_val.npy` and `output_val.npy`: `[stoichiometric_coordinates, delta_t]`
- `actual_output_train.npy` and `actual_output_val.npy`: the actual predictions for T, mass fractions and surface coverages 


## 2. Train the model

Before training, set the four path constants at the top of `train_model.py`:


Training uses the `StoichiometricNet` architecture, a batch size of 128, and
5,000 epochs. The trained weights are written to `model.pth`.

## 3. Use the trained model

For a direct model prediction, load `model.pth`, preprocess one raw input with
`preprocess_stoichiometry`, and call `model.predict`. The result contains the
standardized stoichiometric coordinates and temperature change:

```python
import torch

from nets.stoichiometric_net import StoichiometricNet
from utils.preprocessing import preprocess_stoichiometry

model = StoichiometricNet(input_size=14, hidden_size=128, output_size=6)
model.load_state_dict(torch.load("model.pth", map_location="cpu",
                                 weights_only=True))
model.eval()

# x must contain one raw input with 14 values:
# temperature, pressure, porosity, advance time, catalytic surface area,
# 4 gas mass fractions, and 5 surface coverages.
x = torch.tensor([...], dtype=torch.float32)
model_input = preprocess_stoichiometry(x)
prediction = model.predict(model_input.unsqueeze(0))
```

To reconstruct temperature, gas mass fractions, and surface coverages from
the predicted stoichiometric coordinates, use the reactor context and the
`predict_real`/`predict_real_from_input` methods in
`nets\stoichiometric_net.py`. Here the `reverse_scaling` method must be passed with for each model specific `OUTPUT_MEAN` and `OUTPUT_STD`. 
