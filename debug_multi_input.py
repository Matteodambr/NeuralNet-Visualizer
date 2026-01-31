#!/usr/bin/env python3
"""Debug script to understand multi-input centering issue"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from NN_DEFINITION_UTILITIES import *
from NN_PLOTTING_UTILITIES import NetworkPlotter

# Create a network with 3 image inputs like in the test
network = NeuralNetwork(name="Multi-Input Test")

# Add image inputs with different custom sizes
small_input = ImageInput(
    height=64, width=64, channels=3,
    name="Small Input",
    display_mode="image",
    image_path="src/readme_image_static/catto.jpg",
    color_mode="rgb",
    custom_size=1.5
)

medium_input = ImageInput(
    height=128, width=128, channels=3,
    name="Medium Input",
    display_mode="image",
    image_path="src/readme_image_static/catto.jpg",
    color_mode="rgb",
    custom_size=2.5
)

large_input = ImageInput(
    height=224, width=224, channels=3,
    name="Large Input",
    display_mode="image",
    image_path="src/readme_image_static/catto.jpg",
    color_mode="rgb",
    custom_size=3.5
)

network.add_layer(small_input, is_input=True)
network.add_layer(medium_input, is_input=True)
network.add_layer(large_input, is_input=True)

# Add a merge layer
fc = FullyConnectedLayer(num_neurons=16, name="Merged")
network.add_layer(fc)
network.add_connection(small_input, fc)
network.add_connection(medium_input, fc)
network.add_connection(large_input, fc)

# Add output
output = OutputLayer(num_neurons=10, name="Output")
network.add_layer(output)
network.add_connection(fc, output)

# Now plot and check positions
plotter = NetworkPlotter()

# Monkey-patch to add debug output
original_calc_branching = plotter._calculate_branching_positions

def debug_calc_branching(network):
    result = original_calc_branching(network)
    
    print("\n=== Debug Info ===")
    print(f"Neuron positions:")
    for layer_id, positions in plotter.neuron_positions.items():
        if positions:
            y_vals = [p[1] for p in positions]
            print(f"  {layer_id}: y_min={min(y_vals):.2f}, y_max={max(y_vals):.2f}, y_center={sum(y_vals)/len(y_vals):.2f}")
    
    print(f"\nLayer positions:")
    for layer_id, pos in plotter.layer_positions.items():
        print(f"  {layer_id}: {pos}")
    
    return result

plotter._calculate_branching_positions = debug_calc_branching

# Perform calculation
plotter._calculate_branching_positions(network)
