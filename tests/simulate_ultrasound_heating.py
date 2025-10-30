# %% [markdown]
"""
Focused ultrasound heating workflow driven entirely from Python using k-Wave.

The script first launches a 2D acoustic simulation with `kspaceFirstOrder2D` to obtain
the steady-state intensity distribution, converts the absorption into a volumetric
heat source, and then evolves tissue temperature using the Python port of
`kWaveDiffusion`.

Two cases are modelled:
1. Homogeneous soft-tissue-like medium.
2. Layered air–skull–brain stack.

Run from the repository root:
    python tests/simulate_ultrasound_heating.py
"""

# %%
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
KWAVE_PYTHON_ROOT = REPO_ROOT / "k-wave-python"
if KWAVE_PYTHON_ROOT.exists():
    sys.path.append(str(KWAVE_PYTHON_ROOT))

# %%
from kwave.data import Vector
from kwave.kWaveDiffusion import kWaveDiffusion
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.ksource import kSource
from kwave.kWaveSimulation import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.signals import tone_burst


# %%
@dataclass
class CaseConfig:
    name: str
    layers: Iterable[Dict[str, float]]
    medium_alpha_power: float
    source_frequency: float
    source_cycles: int
    source_pressure: float
    focus_depth: float
    aperture_width: float
    sonication_time: float
    cooling_time: float
    thermal_dt: float
    initial_temperature: float = 37.0


# %%
def build_layered_fields(
    grid_points: Tuple[int, int], grid_spacing: Tuple[float, float], layers: Iterable[Dict[str, float]]
) -> Dict[str, np.ndarray]:
    Nx, Ny = grid_points
    _, dy = grid_spacing
    layer_list = list(layers)

    shape = (Nx, Ny)
    sound_speed = np.empty(shape, dtype=np.float64)
    density = np.empty_like(sound_speed)
    alpha_coeff = np.empty_like(sound_speed)
    thermal_conductivity = np.empty_like(sound_speed)
    heat_capacity = np.empty_like(sound_speed)

    depth_index = 0
    for idx, layer in enumerate(layer_list):
        thickness = layer["thickness"]
        cells = int(round(thickness / dy))
        if idx == len(layer_list) - 1:
            cells = Ny - depth_index
        end_index = min(depth_index + cells, Ny)
        sl = slice(depth_index, end_index)
        for arr, key in (
            (sound_speed, "sound_speed"),
            (density, "density"),
            (alpha_coeff, "alpha_coeff"),
            (thermal_conductivity, "thermal_conductivity"),
            (heat_capacity, "heat_capacity"),
        ):
            arr[:, sl] = layer[key]
        depth_index = end_index
        if depth_index >= Ny:
            break

    if depth_index < Ny:
        tail = layer_list[-1]
        sound_speed[:, depth_index:] = tail["sound_speed"]
        density[:, depth_index:] = tail["density"]
        alpha_coeff[:, depth_index:] = tail["alpha_coeff"]
        thermal_conductivity[:, depth_index:] = tail["thermal_conductivity"]
        heat_capacity[:, depth_index:] = tail["heat_capacity"]

    return {
        "sound_speed": sound_speed,
        "density": density,
        "alpha_coeff": alpha_coeff,
        "thermal_conductivity": thermal_conductivity,
        "heat_capacity": heat_capacity,
    }

# %%
def power_law_alpha_to_neper(alpha_coeff: np.ndarray, alpha_power: float, frequency: float) -> np.ndarray:
    freq_mhz = frequency / 1e6
    alpha_db_per_cm = alpha_coeff * (freq_mhz**alpha_power)
    alpha_db_per_m = alpha_db_per_cm * 100.0
    return alpha_db_per_m * np.log(10.0) / 20.0


# %%
def configure_source(
    kgrid: kWaveGrid,
    grid_points: Tuple[int, int],
    dx: float,
    source_depth_index: int,
    aperture_width: float,
    focus_depth: float,
    frequency: float,
    cycles: int,
    amplitude: float,
) -> kSource:
    Nx, _ = grid_points
    source = kSource()
    mask = np.zeros(grid_points, dtype=bool)

    aperture_cells = max(1, int(round(aperture_width / dx)))
    x_start = max(0, Nx // 2 - aperture_cells // 2)
    x_end = min(Nx, x_start + aperture_cells)
    active_x = np.arange(x_start, x_end)
    mask[active_x, source_depth_index] = True
    source.p_mask = mask

    x_positions = (active_x - active_x.mean()) * dx
    c_ref = 1540.0
    delays = (np.sqrt(x_positions**2 + focus_depth**2) - focus_depth) / c_ref
    delays -= delays.min()
    delay_samples = np.round(delays / kgrid.dt).astype(int)

    signal_matrix = tone_burst(
        sample_freq=1.0 / kgrid.dt,
        signal_freq=frequency,
        num_cycles=cycles,
        signal_length=kgrid.Nt,
        signal_offset=delay_samples,
    )
    source.p = (amplitude * signal_matrix).T
    return source


# %%
def run_acoustic_simulation(
    case: CaseConfig,
    grid_points: Tuple[int, int],
    spacing: Tuple[float, float],
    fields: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    dx, dy = spacing
    medium = kWaveMedium(
        sound_speed=fields["sound_speed"],
        density=fields["density"],
        alpha_coeff=fields["alpha_coeff"],
        alpha_power=case.medium_alpha_power,
    )

    kgrid = kWaveGrid(Vector(list(grid_points)), Vector([dx, dy]))
    c_max = float(fields["sound_speed"].max())
    tone_length = case.source_cycles / case.source_frequency
    focus_travel = case.focus_depth / min(float(fields["sound_speed"].min()), c_max)
    kgrid.makeTime(c=c_max, cfl=0.3, t_end=tone_length + focus_travel + 40e-6)

    source_depth_index = 2
    source = configure_source(
        kgrid=kgrid,
        grid_points=grid_points,
        dx=dx,
        source_depth_index=source_depth_index,
        aperture_width=case.aperture_width,
        focus_depth=case.focus_depth,
        frequency=case.source_frequency,
        cycles=case.source_cycles,
        amplitude=case.source_pressure,
    )

    sensor_mask = np.ones(grid_points, dtype=bool)
    sensor = kSensor(sensor_mask, record=["I_avg"])

    simulation_options = SimulationOptions(
        pml_inside=False,
        pml_size=Vector([16, 16]),
        data_cast="single",
        save_to_disk=True,
    )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=False)

    sensor_data = kspaceFirstOrder2DC(
        kgrid=kgrid,
        medium=medium,
        source=source,
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=execution_options,
    )

    intensity = sensor_data["I_avg"]
    intensity = np.reshape(intensity, grid_points, order="F")

    return {
        "kgrid": kgrid,
        "intensity": intensity,
    }


# %%
def simulate_case(case: CaseConfig, grid_points: Tuple[int, int], spacing: Tuple[float, float], output_dir: Path) -> None:
    fields = build_layered_fields(grid_points, spacing, case.layers)
    acoustic = run_acoustic_simulation(case, grid_points, spacing, fields)
    kgrid = acoustic["kgrid"]
    intensity = acoustic["intensity"]

    alpha_np = power_law_alpha_to_neper(fields["alpha_coeff"], case.medium_alpha_power, case.source_frequency)
    heat_source = 2.0 * alpha_np * intensity

    diffusion_medium = {
        "density": fields["density"],
        "thermal_conductivity": fields["thermal_conductivity"],
        "specific_heat": fields["heat_capacity"],
    }
    diffusion_source = {"T0": case.initial_temperature, "Q": heat_source}

    kdiff = kWaveDiffusion(kgrid, diffusion_medium, diffusion_source)

    heat_steps = int(np.ceil(case.sonication_time / case.thermal_dt))
    cool_steps = int(np.ceil(case.cooling_time / case.thermal_dt))

    kdiff.takeTimeStep(heat_steps, case.thermal_dt)
    temperature_after_heating = np.copy(kdiff.T)

    kdiff.Q = 0.0
    kdiff.takeTimeStep(cool_steps, case.thermal_dt)
    temperature_final = np.copy(kdiff.T)
    dose = np.copy(kdiff.cem43)

    result_path = output_dir / f"{case.name}_heating.npz"
    np.savez_compressed(
        result_path,
        intensity=intensity,
        heat_source=heat_source,
        temperature_after_heating=temperature_after_heating,
        temperature_final=temperature_final,
        cem43=dose,
        dx=spacing[0],
        dy=spacing[1],
        thermal_dt=case.thermal_dt,
        sonication_time=case.sonication_time,
        cooling_time=case.cooling_time,
    )

    delta_temp = temperature_after_heating - case.initial_temperature
    peak_rise = float(delta_temp.max())
    focus_index = np.unravel_index(np.argmax(delta_temp), delta_temp.shape)
    depth_mm = focus_index[1] * spacing[1] * 1e3
    lateral_mm = focus_index[0] * spacing[0] * 1e3

    print(
        f"[{case.name}] peak ΔT = {peak_rise:.2f} °C at "
        f"{lateral_mm:.1f} mm lateral, {depth_mm:.1f} mm depth (saved to {result_path.name})"
    )

# %%
def main() -> None:
    grid_points = (160, 224)
    dx = 0.0003125
    dy = 0.0003125
    spacing = (dx, dy)

    output_dir = REPO_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    homogeneous_layers = [
        {
            "name": "tissue",
            "thickness": grid_points[1] * dy,
            "sound_speed": 1540.0,
            "density": 1040.0,
            "alpha_coeff": 0.75,
            "thermal_conductivity": 0.52,
            "heat_capacity": 3600.0,
        }
    ]

    layered_layers = [
        {
            "name": "air",
            "thickness": 4e-3,
            "sound_speed": 343.0,
            "density": 1.2,
            "alpha_coeff": 4.0,
            "thermal_conductivity": 0.024,
            "heat_capacity": 1005.0,
        },
        {
            "name": "skull",
            "thickness": 6e-3,
            "sound_speed": 2800.0,
            "density": 1850.0,
            "alpha_coeff": 12.0,
            "thermal_conductivity": 0.6,
            "heat_capacity": 1313.0,
        },
        {
            "name": "brain",
            "thickness": grid_points[1] * dy - 10e-3,
            "sound_speed": 1540.0,
            "density": 1040.0,
            "alpha_coeff": 0.75,
            "thermal_conductivity": 0.52,
            "heat_capacity": 3600.0,
        },
    ]

    homogeneous_case = CaseConfig(
        name="homogeneous",
        layers=homogeneous_layers,
        medium_alpha_power=1.5,
        source_frequency=0.7e6,
        source_cycles=8,
        source_pressure=1.2e6,
        focus_depth=0.03,
        aperture_width=0.018,
        sonication_time=0.2,
        cooling_time=0.3,
        thermal_dt=0.05,
    )

    layered_case = CaseConfig(
        name="layered",
        layers=layered_layers,
        medium_alpha_power=1.3,
        source_frequency=0.7e6,
        source_cycles=8,
        source_pressure=1.2e6,
        focus_depth=0.035,
        aperture_width=0.020,
        sonication_time=0.2,
        cooling_time=0.3,
        thermal_dt=0.05,
    )

    simulate_case(homogeneous_case, grid_points, spacing, output_dir)
    simulate_case(layered_case, grid_points, spacing, output_dir)


# %%
if __name__ == "__main__":
    main()
