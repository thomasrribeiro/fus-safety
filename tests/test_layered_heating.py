# %%
"""
Heating By A Focused Ultrasound Transducer

This example demonstrates how to combine acoustic and thermal simulations
in k-Wave to calculate the heating by a focused ultrasound transducer. It
builds on the Simulating Transducer Field Patterns and Using A Binary
Sensor Mask examples.

This is a Python port of the MATLAB example:
k-Wave/examples/example_diff_focused_ultrasound_heating.m

author: Python port
date: 2025
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Enable auto-reload for imported modules (only works in IPython/Jupyter)
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')
except:
    pass

# Add parent directory to path to import kWaveDiffusion
sys.path.insert(0, str(Path(__file__).parent.parent))

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.data import Vector
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.signals import tone_burst
from kwave.utils.conversion import db2neper
from kWaveDiffusion import kWaveDiffusion

# %%
# =========================================================================
# ACOUSTIC SIMULATION
# =========================================================================

print("=" * 70)
print("ACOUSTIC SIMULATION - 3D")
print("=" * 70)

# Define constants (from test_pressure.py)
c0 = 1540  # [m/s]
rho0 = 1000  # [kg/m^3]
source_f0 = 1.8e6  # [Hz]
source_amp = 1e6  # [Pa]
source_cycles = 5
source_focus = 20e-3  # [m]

# Transducer array parameters
element_num = 64
element_width = 208e-6  # [m]
element_length = 208e-6 * 140  # [m]
element_pitch = element_width

# Array positioning
translation = Vector([0, 0, 0])
rotation = Vector([0, 0, 0])

# Grid parameters
grid_points = Vector([128, 256, 128])
grid_size_x = grid_points.x * element_pitch
grid_size_y = grid_points.y * element_pitch
grid_size_z = grid_points.z * element_pitch

# Simulation parameters
ppw = 3
t_end = 35e-6  # [s]
cfl = 0.5

# Calculate grid spacing and dimensions
dx = c0 / (ppw * source_f0)
Nx = round(grid_size_x / dx)
Ny = round(grid_size_y / dx)
Nz = round(grid_size_z / dx)

print(f"Grid size: {Nx} x {Ny} x {Nz} points")
print(f"Grid spacing: {dx*1e3:.3f} mm")
print(f"Physical size: {grid_size_x*1e3:.1f} x {grid_size_y*1e3:.1f} x {grid_size_z*1e3:.1f} mm")

# Create the 3D computational grid
kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])
kgrid.makeTime(c0, cfl, t_end)

# Define the properties of the propagation medium
alpha_coeff = 0.75  # [dB/(MHz^y cm)]
alpha_power = 1.5

medium = kWaveMedium(
    sound_speed=c0,
    density=rho0,
    alpha_coeff=alpha_coeff,
    alpha_power=alpha_power
)

print(f"Medium: c={c0} m/s, rho={rho0} kg/m^3")
print(f"Absorption: alpha={alpha_coeff} dB/(MHz^{alpha_power} cm)")
print(f"Source: f={source_f0*1e-6:.1f} MHz, amp={source_amp*1e-6:.1f} MPa")
print(f"Transducer: {element_num} elements, focus={source_focus*1e3:.1f} mm")

# %%
# Create transducer array with focusing delays (from test_pressure.py)

# Calculate element positions
if element_num % 2 != 0:
    ids = np.arange(1, element_num + 1) - np.ceil(element_num / 2)
else:
    ids = np.arange(1, element_num + 1) - (element_num + 1) / 2

# Calculate time delays for geometric focusing
if not np.isinf(source_focus):
    time_delays = -(np.sqrt((ids * element_pitch) ** 2 + source_focus**2) - source_focus) / c0
    time_delays = time_delays - min(time_delays)
else:
    time_delays = np.zeros(element_num)

print(f"Time delay range: {time_delays.min()*1e6:.2f} to {time_delays.max()*1e6:.2f} us")

# Create source signal with focusing delays
source_sig = source_amp * tone_burst(
    1 / kgrid.dt,
    source_f0,
    source_cycles,
    signal_offset=np.round(time_delays / kgrid.dt).astype(int)
)

print(f"Source signal shape: {source_sig.shape}")

# Create kWaveArray for linear array transducer
karray = kWaveArray()

# Add rectangular elements to array
for ind in range(element_num):
    x_pos = 0 - (element_num * element_pitch / 2 - element_pitch / 2) + ind * element_pitch
    karray.add_rect_element(
        [x_pos, 0, kgrid.z_vec[0][0]],
        element_width,
        element_length,
        rotation
    )

print(f"Added {element_num} rectangular elements to array")

# Set array position
karray.set_array_position(translation, rotation)

# Create source object
source = kSource()
source.p_mask = karray.get_array_binary_mask(kgrid)
print(f"Source mask has {source.p_mask.sum()} active points")

source.p = karray.get_distributed_source_signal(kgrid, source_sig)
print(f"Source signal distributed, shape: {source.p.shape}")

# %%
# set the sensor mask to cover the entire 3D grid
sensor = kSensor()
sensor.mask = np.ones((Nx, Ny, Nz), dtype=bool)
sensor.record = ['p_max']  # Record maximum pressure (steady-state amplitude)

print(f"Sensor recording p_max over entire {Nx}x{Ny}x{Nz} grid")

# %%
# set the simulation options (from test_pressure.py)
simulation_options = SimulationOptions(
    pml_auto=True,  # Automatically determine PML size
    pml_inside=False,
    save_to_disk=True,
    data_cast='single'
)

execution_options = SimulationExecutionOptions(
    is_gpu_simulation=False,
    verbose_level=1
)

# run the 3D acoustic simulation
print("\nRunning 3D acoustic simulation...")
print("NOTE: This may take a while due to large 3D grid size...")
sensor_data = kspaceFirstOrder3DC(
    kgrid=kgrid,
    medium=medium,
    source=source,
    sensor=sensor,
    simulation_options=simulation_options,
    execution_options=execution_options
)

print("Acoustic simulation complete!")

# %%
# =========================================================================
# CALCULATE HEATING
# =========================================================================

print("\n" + "=" * 70)
print("CALCULATE HEATING")
print("=" * 70)

# convert the absorption coefficient to nepers/m
alpha_np = db2neper(alpha_coeff, alpha_power) * (2 * np.pi * source_f0) ** alpha_power
print(f"Absorption coefficient: {alpha_np:.2f} Np/m")

# extract the pressure amplitude from p_max (already the maximum pressure)
p_max_data = sensor_data['p_max']
print(f"Pressure p_max data shape: {p_max_data.shape}")
print(f"Expected: {Nx * Ny * Nz} total points")

# reshape the 3D pressure field
p = np.reshape(p_max_data, (Nx, Ny, Nz), order='F')  # Fortran order for MATLAB compatibility
print(f"Pressure amplitude field shape: {p.shape}")
print(f"Pressure range: {p.min()*1e-6:.3f} to {p.max()*1e-6:.3f} MPa")

# calculate the volume rate of heat deposition (3D field)
Q = alpha_np * p**2 / (rho0 * c0)

# Account for pulsed operation - scale by duty cycle
pulse_repetition_frequency = 1000  # [Hz] - how often pulses repeat
pulse_duration = source_cycles / source_f0  # [s] - duration of one tone burst
duty_cycle = pulse_duration * pulse_repetition_frequency

print(f"\nPulsing parameters:")
print(f"  Pulse duration: {pulse_duration*1e6:.2f} us ({source_cycles} cycles @ {source_f0*1e-6:.1f} MHz)")
print(f"  PRF: {pulse_repetition_frequency} Hz")
print(f"  Duty cycle: {duty_cycle*100:.3f}%")

# Scale heat deposition by duty cycle
Q = Q * duty_cycle

print(f"\nHeat deposition (instantaneous): {Q.max()/duty_cycle*1e-7:.3f} kW/cm^3")
print(f"Heat deposition (time-averaged): {Q.min()*1e-7:.3f} to {Q.max()*1e-7:.3f} kW/cm^3")
print(f"Peak heat deposition location: {np.unravel_index(Q.argmax(), Q.shape)}")

# %%
# =========================================================================
# THERMAL SIMULATION
# =========================================================================

print("\n" + "=" * 70)
print("THERMAL SIMULATION")
print("=" * 70)

# define medium properties related to diffusion
thermal_medium = {
    'density': 1020,  # [kg/m^3]
    'thermal_conductivity': 0.5,  # [W/(m.K)]
    'specific_heat': 3600  # [J/(kg.K)]
}

# define source for thermal simulation
thermal_source = {
    'Q': Q,
    'T0': 37  # [degC]
}

print(f"Thermal medium: rho={thermal_medium['density']} kg/m^3, "
      f"k={thermal_medium['thermal_conductivity']} W/(m.K), "
      f"cp={thermal_medium['specific_heat']} J/(kg.K)")
print(f"Initial temperature: {thermal_source['T0']} degC")

# create kWaveDiffusion object
kdiff = kWaveDiffusion(kgrid, thermal_medium, thermal_source, None)

# set source on time and off time
on_time = 10  # [s]
off_time = 20  # [s]

# set time step size
thermal_dt = 0.1  # [s]

print(f"Heating phase: {on_time} s (dt={thermal_dt} s)")

# %%
# take time steps for heating phase
kdiff.takeTimeStep(int(round(on_time / thermal_dt)), thermal_dt)

# store the current temperature field
T1 = kdiff.T.copy()
print(f"Temperature after heating: {T1.min():.1f} to {T1.max():.1f} degC")

# turn off heat source and take time steps
print(f"Cooling phase: {off_time} s")
kdiff.Q = 0
kdiff.takeTimeStep(int(round(off_time / thermal_dt)), thermal_dt)

# store the current temperature field
T2 = kdiff.T.copy()
print(f"Temperature after cooling: {T2.min():.1f} to {T2.max():.1f} degC")

# get thermal dose and lesion map
cem43 = kdiff.cem43
lesion_map = kdiff.lesion_map

print(f"CEM43 range: {cem43.min():.1f} to {cem43.max():.1f} min")
print(f"Lesion volume: {lesion_map.sum()} / {lesion_map.size} points "
      f"({100 * lesion_map.sum() / lesion_map.size:.2f}%)")

# %%
# =========================================================================
# VISUALISATION - 3D SLICES
# =========================================================================

print("\n" + "=" * 70)
print("VISUALISATION")
print("=" * 70)

# create figure with 3x2 subplots showing XZ slices (center Y)
fig, axes = plt.subplots(3, 2, figsize=(14, 18))

# get coordinate vectors in mm
x_vec = kgrid.x_vec.flatten() * 1e3  # [mm]
y_vec = kgrid.y_vec.flatten() * 1e3  # [mm]
z_vec = kgrid.z_vec.flatten() * 1e3  # [mm]

# Extract center XZ slice (through middle of Y dimension)
center_y = Ny // 2
p_slice = p[:, center_y, :]
Q_slice = Q[:, center_y, :]
T1_slice = T1[:, center_y, :]
T2_slice = T2[:, center_y, :]
cem43_slice = cem43[:, center_y, :]
lesion_slice = lesion_map[:, center_y, :]

print(f"Showing XZ slices at Y = {y_vec[center_y]:.1f} mm (center)")

# plot 1: acoustic pressure amplitude
ax = axes[0, 0]
im1 = ax.imshow(p_slice * 1e-6, extent=[z_vec.min(), z_vec.max(), x_vec.max(), x_vec.min()],
                aspect='auto', cmap='jet')
ax.set_xlabel('z-position [mm]')
ax.set_ylabel('x-position [mm]')
ax.set_title('Acoustic Pressure Amplitude (XZ slice)')
cbar1 = plt.colorbar(im1, ax=ax)
cbar1.set_label('[MPa]')

# plot 2: volume rate of heat deposition
ax = axes[0, 1]
im2 = ax.imshow(Q_slice * 1e-7, extent=[z_vec.min(), z_vec.max(), x_vec.max(), x_vec.min()],
                aspect='auto', cmap='jet')
ax.set_xlabel('z-position [mm]')
ax.set_ylabel('x-position [mm]')
ax.set_title('Volume Rate Of Heat Deposition (XZ slice)')
cbar2 = plt.colorbar(im2, ax=ax)
cbar2.set_label('[kW/cm^3]')

# plot 3: temperature after heating
ax = axes[1, 0]
im3 = ax.imshow(T1_slice, extent=[z_vec.min(), z_vec.max(), x_vec.max(), x_vec.min()],
                aspect='auto', cmap='jet')
ax.set_xlabel('z-position [mm]')
ax.set_ylabel('x-position [mm]')
ax.set_title('Temperature After Heating (XZ slice)')
cbar3 = plt.colorbar(im3, ax=ax)
cbar3.set_label('[degC]')

# plot 4: temperature after cooling
ax = axes[1, 1]
im4 = ax.imshow(T2_slice, extent=[z_vec.min(), z_vec.max(), x_vec.max(), x_vec.min()],
                aspect='auto', cmap='jet')
ax.set_xlabel('z-position [mm]')
ax.set_ylabel('x-position [mm]')
ax.set_title('Temperature After Cooling (XZ slice)')
cbar4 = plt.colorbar(im4, ax=ax)
cbar4.set_label('[degC]')

# plot 5: thermal dose
ax = axes[2, 0]
im5 = ax.imshow(cem43_slice, extent=[z_vec.min(), z_vec.max(), x_vec.max(), x_vec.min()],
                aspect='auto', cmap='jet', vmin=0, vmax=1000)
ax.set_xlabel('z-position [mm]')
ax.set_ylabel('x-position [mm]')
ax.set_title('Thermal Dose (XZ slice)')
cbar5 = plt.colorbar(im5, ax=ax)
cbar5.set_label('[CEM43]')

# plot 6: lesion map
ax = axes[2, 1]
im6 = ax.imshow(lesion_slice.astype(float), extent=[z_vec.min(), z_vec.max(), x_vec.max(), x_vec.min()],
                aspect='auto', cmap='jet', vmin=0, vmax=1)
ax.set_xlabel('z-position [mm]')
ax.set_ylabel('x-position [mm]')
ax.set_title('Ablated Tissue (XZ slice)')
cbar6 = plt.colorbar(im6, ax=ax)

plt.tight_layout()

# Save to results directory
results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(exist_ok=True)
output_file = results_dir / 'focused_ultrasound_heating_3d.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")
plt.show()

print("\n" + "=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)

# %%
