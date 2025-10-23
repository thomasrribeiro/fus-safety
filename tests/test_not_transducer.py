# %%
# Import

import matplotlib.pyplot as plt
import numpy as np

from kwave.data import Vector
from kwave.utils.dotdictionary import dotdict
from kwave.kWaveSimulation import SimulationOptions
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.ktransducer import NotATransducer, kWaveTransducerSimple
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.plot import voxel_plot
from kwave.utils.signals import tone_burst
from kwave.utils.colormap import get_color_map

# %%
# Parameters

c0 = 1540
rho0 = 1000
source_f0 = 2.7e6 # [Hz]
source_strength = 1e6 # [Pa]
source_cycles = 5 # [cycles]

# source_focus = float('inf')
source_focus = 20e-3

element_num_az = 140 # number of elements in the azimuth direction
element_num_el = 64 # number of elements in the elevation direction
element_width = 208e-6 # [m]
element_length = 208e-6 * element_num_el # [m]
element_pitch = 208e-6 # [m]
element_spacing = 0 # [m]
translation = Vector([0, 0, 0])
rotation = Vector([0, 0, 0])
grid_spacing_meters = Vector([element_pitch, element_pitch, element_pitch]) # [m]
pml_size_points = Vector([20, 10, 10])  # [grid points]
grid_size_points = Vector([256, 256, 128]) - 2 * pml_size_points  # [grid points]

cfl=0.5
t_end=35e-6

# %%
# Grid, input signal and medium

kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)
kgrid.makeTime(c0, cfl, t_end)

input_signal = tone_burst(1 / kgrid.dt, source_f0, source_cycles)
input_signal = (source_strength / (c0 * rho0)) * input_signal

medium = kWaveMedium(sound_speed=c0, density=rho0)

# %%
# Define transducer

transducer = dotdict()
transducer.number_elements = element_num_az  # total number of transducer elements
transducer.element_width = int(element_width / element_pitch) # width of each element in grid points
transducer.element_length = int(element_length / element_pitch) # length of each element in grid points
transducer.element_spacing = int(element_spacing / element_pitch) # spacing between elements in grid points
transducer.radius = float('inf') # radius of the transducer in grid points

# calculate the width of the transducer in grid points
transducer_width = transducer.number_elements * transducer.element_width + (
        transducer.number_elements - 1) * transducer.element_spacing

# use this to position the transducer in the middle of the computational grid
transducer.position = np.round([
    1,
    grid_size_points.y / 2 - transducer_width / 2,
    grid_size_points.z / 2 - transducer.element_length / 2
])
transducer = kWaveTransducerSimple(kgrid, **transducer)

not_transducer = dotdict()
not_transducer.sound_speed = c0  # sound speed [m/s]
not_transducer.focus_distance = float('inf')  # focus distance [m]
not_transducer.elevation_focus_distance = source_focus  # focus distance in the elevation plane [m]
not_transducer.steering_angle = 0  # steering angle [degrees]
not_transducer.transmit_apodization = 'Hanning'
not_transducer.receive_apodization = 'Rectangular'
not_transducer.active_elements = np.ones((transducer.number_elements, 1))
not_transducer.input_signal = input_signal

not_transducer = NotATransducer(transducer, kgrid, **not_transducer)

# Simulation
simulation_options = SimulationOptions(
    pml_inside=False,
    pml_size=pml_size_points,
    data_cast="single",
    data_recast = True,
    save_to_disk=True,
)
execution_options = SimulationExecutionOptions(is_gpu_simulation=False)

sensor_data = kspaceFirstOrder3DC(
    kgrid=kgrid, medium=medium,
    source=not_transducer, sensor=not_transducer, 
    simulation_options=simulation_options, 
    execution_options=execution_options
)

# %%

# p_max = np.reshape(sensor_data["p_max"], (Nx, Nz), order="F")
p_max = np.reshape(sensor_data["p_max"], (grid_size_points.x, grid_size_points.z), order="F")

# VISUALISATION
plt.figure()
plt.imshow(
    1e-6 * p_max[grid_size_points.x // 2, :, :],
    extent=[1e3 * kgrid.x_vec[0][0], 1e3 * kgrid.x_vec[-1][0], 1e3 * kgrid.z_vec[0][0], 1e3 * kgrid.z_vec[-1][0]],
    aspect="auto",
    cmap=get_color_map(),
)
plt.xlabel("z-position [mm]")
plt.ylabel("x-position [mm]")
plt.title("Pressure Field")
plt.colorbar(label="[MPa]")
plt.show()

# %%
# Extract centerline data
centerline_index = grid_size_points.y // 2
centerline_pressure = p_max[grid_size_points.x // 2, centerline_index, :]
z_axis = 1e3 * kgrid.z_vec[0]  # Convert to mm

# 1D plot
plt.figure()
plt.plot(1e-6 * centerline_pressure)
plt.xlabel('z-position [mm]')
plt.ylabel('Pressure [MPa]')
plt.title('Pressure Along Centerline')
plt.grid(True)
plt.show()