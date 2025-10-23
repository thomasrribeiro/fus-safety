# %%
# Import

import matplotlib.pyplot as plt
import numpy as np

import kwave.data
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.kWaveSimulation import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.colormap import get_color_map
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.plot import voxel_plot
from kwave.utils.signals import tone_burst

# %%
# Define constants

c0 = 1540
rho0 = 1000
source_f0 = 2.7e6
source_amp = 1e6
source_cycles = 5
source_focus = 20e-3 # [m]
element_num = 64
element_width = 208e-6
element_length = 208e-6 * 140
element_pitch = element_width
translation = kwave.data.Vector([0, 0, 0])
rotation = kwave.data.Vector([0, 0, 0])
grid_points = kwave.data.Vector([128, 256, 128])
grid_size_x = grid_points.x * element_pitch
grid_size_y = grid_points.y * element_pitch
grid_size_z = grid_points.z * element_pitch
source_focus = -grid_size_z / 2 + source_focus
ppw = 3
t_end = 35e-6
cfl = 0.5

# %%
# GRID
dx = c0 / (ppw * source_f0)
Nx = round(grid_size_x / dx)
Ny = round(grid_size_y / dx)
Nz = round(grid_size_z / dx)
kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])
kgrid.makeTime(c0, cfl, t_end)

# %%
# Source

if element_num % 2 != 0:
    ids = np.arange(1, element_num + 1) - np.ceil(element_num / 2)
else:
    ids = np.arange(1, element_num + 1) - (element_num + 1) / 2

if not np.isinf(source_focus):
    time_delays = -(np.sqrt((ids * element_pitch) ** 2 + source_focus**2) - source_focus) / c0
    time_delays = time_delays - min(time_delays)
else:
    time_delays = np.zeros(element_num)

source_sig = source_amp * tone_burst(1 / kgrid.dt, source_f0, source_cycles,
                                      signal_offset=np.round(time_delays / kgrid.dt).astype(int))
karray = kWaveArray()#bli_tolerance=0.05, upsampling_rate=10)

for ind in range(element_num):
    x_pos = 0 - (element_num * element_pitch / 2 - element_pitch / 2) + ind * element_pitch
    karray.add_rect_element([x_pos, 0, kgrid.z_vec[0][0]], element_width, element_length, rotation)

karray.set_array_position(translation, rotation)
source = kSource()
source.p_mask = karray.get_array_binary_mask(kgrid)
voxel_plot(np.single(source.p_mask))
source.p = karray.get_distributed_source_signal(kgrid, source_sig)

# %%
# Medium

medium = kWaveMedium(sound_speed=c0, density=rho0)

# %%
# Sensor

sensor_mask = np.ones((Nx, Ny, Nz))
sensor = kSensor(sensor_mask, record=['p_max'])

# %%
# Simulation
simulation_options = SimulationOptions(
    pml_auto=True,
    pml_inside=False,
    save_to_disk=True,
    data_cast="single",
)

execution_options = SimulationExecutionOptions(is_gpu_simulation=False)

sensor_data = kspaceFirstOrder3DC(
    kgrid=kgrid, medium=medium, source=source, sensor=sensor, simulation_options=simulation_options, execution_options=execution_options
)

# %%
# p_max = np.reshape(sensor_data["p_max"], (Nx, Nz), order="F")
p_max = np.reshape(sensor_data["p_max"], (Nx, Ny, Nz), order="F")

# %%
# VISUALISATION
plt.figure()
plt.imshow(
    1e-6 * p_max[:, Ny // 2, :],
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
centerline_index = Ny // 2
centerline_pressure = p_max[Nx // 2, centerline_index, :]
z_axis = 1e3 * kgrid.z_vec[0]  # Convert to mm

# 1D plot
plt.figure()
plt.plot(1e-6 * centerline_pressure)
plt.xlabel('z-position [mm]')
plt.ylabel('Pressure [MPa]')
plt.title('Pressure Along Centerline')
plt.grid(True)
plt.show()

# %%
# np.save('results/unfocused_pressure.npy', centerline_pressure)
# np.save('../results/focused_pressure.npy', centerline_pressure)
# %%
