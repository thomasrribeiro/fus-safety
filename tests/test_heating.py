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
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.data import Vector
from kwave.utils.mapgen import make_arc
from kwave.utils.filters import extract_amp_phase
from kwave.utils.conversion import db2neper
from kWaveDiffusion import kWaveDiffusion

# %%
# =========================================================================
# ACOUSTIC SIMULATION
# =========================================================================

print("=" * 70)
print("ACOUSTIC SIMULATION")
print("=" * 70)

# define the PML size
pml_size = 20  # [grid points]

# define the grid parameters
Nx = 256 - 2 * pml_size  # [grid points]
Ny = 256 - 2 * pml_size  # [grid points]
dx = 0.25e-3  # [m]
dy = 0.25e-3  # [m]

print(f"Grid size: {Nx} x {Ny} points")
print(f"Grid spacing: {dx*1e3:.3f} x {dy*1e3:.3f} mm")

# create the computational grid
kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

# define the properties of the propagation medium
sound_speed = 1510  # [m/s]
density = 1020  # [kg/m^3]
alpha_coeff = 0.75  # [dB/(MHz^y cm)]
alpha_power = 1.5

medium = kWaveMedium(
    sound_speed=sound_speed,
    density=density,
    alpha_coeff=alpha_coeff,
    alpha_power=alpha_power
)

print(f"Medium: c={sound_speed} m/s, rho={density} kg/m^3")
print(f"Absorption: alpha={alpha_coeff} dB/(MHz^{alpha_power} cm)")

# define the source parameters
diameter = 45e-3  # [m]
radius = 35e-3  # [m]
freq = 1e6  # [Hz]
amp = 0.5e6  # [Pa]

print(f"Transducer: diameter={diameter*1e3:.1f} mm, radius of curvature={radius*1e3:.1f} mm")
print(f"Source: f={freq*1e-6:.1f} MHz, p={amp*1e-6:.1f} MPa")

# define a focused ultrasound transducer
source = kSource()
source.p_mask = make_arc(
    Vector([Nx, Ny]),
    np.array([1, Ny // 2 + 1]),  # arc_pos (1-indexed for MATLAB compatibility)
    int(round(radius / dx)),
    int(round(diameter / dx)) + 1,
    Vector([Nx // 2 + 1, Ny // 2 + 1])  # focus_pos (1-indexed)
)

# %%
# calculate the time step using an integer number of points per period
ppw = sound_speed / (freq * dx)  # points per wavelength
cfl = 0.3  # cfl number
ppp = int(np.ceil(ppw / cfl))  # points per period
T = 1 / freq  # period [s]
dt = T / ppp  # time step [s]

print(f"Time parameters: ppw={ppw:.1f}, ppp={ppp}, dt={dt*1e9:.2f} ns")

# calculate the number of time steps to reach steady state
t_end = np.sqrt(kgrid.x_size**2 + kgrid.y_size**2) / sound_speed
Nt = int(round(t_end / dt))

print(f"Simulation time: t_end={t_end*1e6:.2f} us, Nt={Nt} steps")

# create the time array
kgrid.setTime(Nt, dt)

# define the input signal (CW)
# Create CW signal manually (matching MATLAB createCWSignals behavior)
t_array = np.arange(0, Nt) * dt
period = 1.0 / freq
ramp_length = 4  # periods

# Create base sinusoid
cw_signal = amp * np.sin(2 * np.pi * freq * t_array)

# Apply cosine taper ramp to avoid startup transients
ramp_length_points = int(round(ramp_length * period / dt))
ramp_axis = np.linspace(0, np.pi, ramp_length_points)
ramp = (-np.cos(ramp_axis) + 1) * 0.5
cw_signal[:ramp_length_points] *= ramp

# Reshape to 2D array (1 time series, Nt time points)
source.p = cw_signal.reshape(1, -1)

# set the sensor mask to cover the entire grid
sensor = kSensor()
sensor.mask = np.ones((Nx, Ny), dtype=bool)
sensor.record = ['p']

# record the last 3 cycles in steady state
num_periods = 3
T_points = int(round(num_periods * T / kgrid.dt))
sensor.record_start_index = Nt - T_points + 1

print(f"Recording last {num_periods} periods ({T_points} time points)")

# %%
# set the simulation options
simulation_options = SimulationOptions(
    pml_inside=False,
    pml_size=Vector([pml_size, pml_size]),
    data_cast='single',
    save_to_disk=True
)

execution_options = SimulationExecutionOptions(
    is_gpu_simulation=False,
    verbose_level=1
)

# run the acoustic simulation
print("\nRunning acoustic simulation...")
sensor_data = kspaceFirstOrder2D(
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
alpha_np = db2neper(alpha_coeff, alpha_power) * (2 * np.pi * freq) ** alpha_power
print(f"Absorption coefficient: {alpha_np:.2f} Np/m")

# extract the pressure amplitude at each position
p_data = sensor_data['p']  # Shape: (num_time_steps, num_sensors) or (num_sensors, num_time_steps)
print(f"Pressure data shape: {p_data.shape}")
print(f"Expected: ({Nx*Ny} sensors, {T_points} time points) or vice versa")

# Determine which dimension is time and which is space
if p_data.shape[0] == Nx * Ny:
    # Shape is (num_sensors, num_time_steps) - already correct
    p_data_formatted = p_data
    print("Format: (sensors, time)")
elif p_data.shape[1] == Nx * Ny:
    # Shape is (num_time_steps, num_sensors) - transpose to (sensors, time)
    p_data_formatted = p_data.T
    print("Format: (time, sensors) - transposing to (sensors, time)")
else:
    # Try to determine based on which dimension is smaller (usually time)
    if p_data.shape[0] < p_data.shape[1]:
        # Assume shape is (num_time_steps, num_sensors) - transpose
        p_data_formatted = p_data.T
        print("Guessing: (time, sensors) - transposing to (sensors, time)")
    else:
        # Assume shape is (num_sensors, num_time_steps) - already correct
        p_data_formatted = p_data
        print("Guessing: (sensors, time)")

print(f"Formatted shape: {p_data_formatted.shape}")

# Extract amplitude for each sensor by processing time series
# For large number of sensors, process in batches or loop
num_sensors = p_data_formatted.shape[0]
amp_extracted = np.zeros(num_sensors)

print("Extracting amplitude for each sensor...")
# Process each sensor individually to avoid broadcasting issues
for i in range(num_sensors):
    time_series = p_data_formatted[i, :]  # Get time series for this sensor
    # Extract amplitude using FFT-based method
    amp, _, _ = extract_amp_phase(
        time_series.reshape(1, -1),  # Shape: (1, num_time_steps)
        1.0 / kgrid.dt,
        freq,
        dim=1  # time is the second dimension
    )
    # Handle both scalar and array returns
    if np.isscalar(amp) or amp.ndim == 0:
        amp_extracted[i] = float(amp)
    else:
        amp_extracted[i] = amp.flat[0]  # Use flat to handle any shape

    # Progress indicator
    if (i + 1) % 5000 == 0 or i == num_sensors - 1:
        print(f"  Processed {i+1}/{num_sensors} sensors...")

print(f"Amplitude extracted shape: {amp_extracted.shape}")

# reshape the data
p = np.reshape(amp_extracted, (Nx, Ny), order='F')  # Fortran order for MATLAB compatibility
print(f"Pressure amplitude field shape: {p.shape}")
print(f"Pressure range: {p.min()*1e-6:.3f} to {p.max()*1e-6:.3f} MPa")

# calculate the volume rate of heat deposition
Q = alpha_np * p**2 / (density * sound_speed)
print(f"Heat deposition range: {Q.min()*1e-7:.3f} to {Q.max()*1e-7:.3f} kW/cm^3")

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
# VISUALISATION
# =========================================================================

print("\n" + "=" * 70)
print("VISUALISATION")
print("=" * 70)

# create figure with 2x3 subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# get coordinate vectors in mm
x_vec = kgrid.x_vec * 1e3  # [mm]
y_vec = kgrid.y_vec * 1e3  # [mm]

# plot 1: acoustic pressure amplitude
ax = axes[0, 0]
im1 = ax.imshow(p * 1e-6, extent=[y_vec.min(), y_vec.max(), x_vec.max(), x_vec.min()],
                aspect='equal', cmap='jet')
ax.set_xlabel('y-position [mm]')
ax.set_ylabel('x-position [mm]')
ax.set_title('Acoustic Pressure Amplitude')
cbar1 = plt.colorbar(im1, ax=ax)
cbar1.set_label('[MPa]')

# plot 2: volume rate of heat deposition
ax = axes[0, 1]
im2 = ax.imshow(Q * 1e-7, extent=[y_vec.min(), y_vec.max(), x_vec.max(), x_vec.min()],
                aspect='equal', cmap='jet')
ax.set_xlabel('y-position [mm]')
ax.set_ylabel('x-position [mm]')
ax.set_title('Volume Rate Of Heat Deposition')
cbar2 = plt.colorbar(im2, ax=ax)
cbar2.set_label('[kW/cm^3]')

# plot 3: temperature after heating
ax = axes[0, 2]
im3 = ax.imshow(T1, extent=[y_vec.min(), y_vec.max(), x_vec.max(), x_vec.min()],
                aspect='equal', cmap='jet')
ax.set_xlabel('y-position [mm]')
ax.set_ylabel('x-position [mm]')
ax.set_title('Temperature After Heating')
cbar3 = plt.colorbar(im3, ax=ax)
cbar3.set_label('[degC]')

# plot 4: temperature after cooling
ax = axes[1, 0]
im4 = ax.imshow(T2, extent=[y_vec.min(), y_vec.max(), x_vec.max(), x_vec.min()],
                aspect='equal', cmap='jet')
ax.set_xlabel('y-position [mm]')
ax.set_ylabel('x-position [mm]')
ax.set_title('Temperature After Cooling')
cbar4 = plt.colorbar(im4, ax=ax)
cbar4.set_label('[degC]')

# plot 5: thermal dose
ax = axes[1, 1]
im5 = ax.imshow(cem43, extent=[y_vec.min(), y_vec.max(), x_vec.max(), x_vec.min()],
                aspect='equal', cmap='jet', vmin=0, vmax=1000)
ax.set_xlabel('y-position [mm]')
ax.set_ylabel('x-position [mm]')
ax.set_title('Thermal Dose')
cbar5 = plt.colorbar(im5, ax=ax)
cbar5.set_label('[CEM43]')

# plot 6: lesion map
ax = axes[1, 2]
im6 = ax.imshow(lesion_map.astype(float), extent=[y_vec.min(), y_vec.max(), x_vec.max(), x_vec.min()],
                aspect='equal', cmap='jet', vmin=0, vmax=1)
ax.set_xlabel('y-position [mm]')
ax.set_ylabel('x-position [mm]')
ax.set_title('Ablated Tissue')
cbar6 = plt.colorbar(im6, ax=ax)

plt.tight_layout()

# Save to results directory
results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(exist_ok=True)
output_file = results_dir / 'focused_ultrasound_heating.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")
plt.show()

print("\n" + "=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)

# %%
