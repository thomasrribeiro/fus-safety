from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.fft import fftn, ifftn

from kwave.kgrid import kWaveGrid


def _as_mapping(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}
    raise TypeError(f"Unsupported container type {type(obj)!r}")


def _has_field(obj: Dict[str, Any], name: str) -> bool:
    return name in obj and obj[name] is not None


def _require_fields(obj: Dict[str, Any], names: Sequence[str]) -> None:
    missing = [name for name in names if not _has_field(obj, name)]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")


def _to_numpy(data: Any, shape: Tuple[int, ...]) -> Union[float, np.ndarray]:
    if data is None:
        return 0.0
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    arr = np.broadcast_to(arr, shape).astype(np.float64, copy=False)
    return arr


def _is_scalar(val: Union[float, np.ndarray]) -> bool:
    return np.isscalar(val) or (isinstance(val, np.ndarray) and val.ndim == 0)


class kWaveDiffusion:
    """
    Python port of the MATLAB kWaveDiffusion class. Implements a pseudospectral
    solver for the diffusion (bioheat) equation using the same interface style
    as the k-Wave Toolbox.

    Current limitations compared to MATLAB:
        * Only periodic boundary conditions are supported.
        * GPU/data casting, movie recording, and live plotting are omitted.
        * Axisymmetric grids are not implemented.
    """

    highest_prime_factor_warning = 7

    def __init__(
        self,
        kgrid: kWaveGrid,
        medium: Union[Dict[str, Any], Any],
        source: Union[Dict[str, Any], Any],
        sensor: Optional[Union[Dict[str, Any], Any]] = None,
        *,
        use_kspace: bool = True,
        display_updates: bool = True,
    ):
        self.kgrid = kgrid
        self.dim = kgrid.dim
        self.Nx = kgrid.Nx
        self.Ny = kgrid.Ny if self.dim >= 2 else 1
        self.Nz = kgrid.Nz if self.dim == 3 else 1
        self.dx = kgrid.dx
        self.dy = kgrid.dy if self.dim >= 2 else 0.0
        self.dz = kgrid.dz if self.dim == 3 else 0.0

        grid_shape = (self.Nx,) if self.dim == 1 else (self.Nx, self.Ny) if self.dim == 2 else (self.Nx, self.Ny, self.Nz)

        medium_map = _as_mapping(medium)
        source_map = _as_mapping(source)
        sensor_map = _as_mapping(sensor)

        self.use_kspace = use_kspace
        self.display_updates = display_updates

        boundary_condition = medium_map.get("boundary_condition", "periodic")
        if boundary_condition != "periodic":
            raise NotImplementedError("Only periodic boundary conditions are currently supported in the Python port.")
        self.boundary_condition = boundary_condition

        if _has_field(medium_map, "diffusion_coeff"):
            diffusion_coeff = _to_numpy(medium_map["diffusion_coeff"], grid_shape)
            self.diffusion_p1 = 1.0
            self.diffusion_p2 = diffusion_coeff
        else:
            _require_fields(medium_map, ["density", "thermal_conductivity", "specific_heat"])
            density = _to_numpy(medium_map["density"], grid_shape)
            thermal_conductivity = _to_numpy(medium_map["thermal_conductivity"], grid_shape)
            specific_heat = _to_numpy(medium_map["specific_heat"], grid_shape)
            self.diffusion_p1 = 1.0 / (density * specific_heat)
            self.diffusion_p2 = thermal_conductivity

        if any(_has_field(medium_map, key) for key in ("blood_density", "blood_specific_heat", "blood_perfusion_rate")):
            _require_fields(
                medium_map,
                ["blood_density", "blood_specific_heat", "blood_perfusion_rate", "blood_ambient_temperature", "density", "specific_heat"],
            )
            blood_density = _to_numpy(medium_map["blood_density"], grid_shape)
            blood_specific_heat = _to_numpy(medium_map["blood_specific_heat"], grid_shape)
            blood_perfusion_rate = _to_numpy(medium_map["blood_perfusion_rate"], grid_shape)
            density = _to_numpy(medium_map["density"], grid_shape)
            specific_heat = _to_numpy(medium_map["specific_heat"], grid_shape)
            self.perfusion_coeff = blood_density * blood_perfusion_rate * blood_specific_heat / (density * specific_heat)
            self.blood_ambient_temperature = _to_numpy(medium_map["blood_ambient_temperature"], grid_shape)
        elif _has_field(medium_map, "perfusion_coeff"):
            _require_fields(medium_map, ["perfusion_coeff", "blood_ambient_temperature"])
            self.perfusion_coeff = _to_numpy(medium_map["perfusion_coeff"], grid_shape)
            self.blood_ambient_temperature = _to_numpy(medium_map["blood_ambient_temperature"], grid_shape)
        else:
            self.perfusion_coeff = 0.0
            self.blood_ambient_temperature = 0.0

        if _has_field(source_map, "Q"):
            if not (_has_field(medium_map, "density") and _has_field(medium_map, "specific_heat")) and _is_scalar(self.diffusion_p1):
                raise ValueError("medium.density and medium.specific_heat must be specified when source.Q is defined.")
            self.Q = _to_numpy(source_map["Q"], grid_shape)

            if _is_scalar(self.diffusion_p1) and self.diffusion_p1 == 1.0:
                density = _to_numpy(medium_map["density"], grid_shape)
                specific_heat = _to_numpy(medium_map["specific_heat"], grid_shape)
                self.q_scale_factor = 1.0 / (density * specific_heat)
            else:
                self.q_scale_factor = 0.0
        else:
            self.Q = 0.0
            self.q_scale_factor = 0.0

        diffusion_fields = [self.diffusion_p1, self.diffusion_p2, self.perfusion_coeff, self.blood_ambient_temperature]
        self.flag_homogeneous = all(_is_scalar(field) for field in diffusion_fields)

        diff_coeff = self.diffusion_p1 * self.diffusion_p2

        diffusion_coeff_ref = medium_map.get("diffusion_coeff_ref", "max")
        self.diffusion_coeff_ref = self._resolve_reference(diffusion_coeff_ref, diff_coeff)

        if np.allclose(self.perfusion_coeff, 0.0):
            self.perfusion_coeff_ref = 0.0
        else:
            perfusion_coeff_ref = medium_map.get("perfusion_coeff_ref", "max")
            self.perfusion_coeff_ref = self._resolve_reference(perfusion_coeff_ref, self.perfusion_coeff)

        _require_fields(source_map, ["T0"])
        initial_temperature = _to_numpy(source_map["T0"], grid_shape)
        if _is_scalar(initial_temperature):
            self.T = np.full(grid_shape, float(initial_temperature), dtype=np.float64)
        else:
            self.T = np.array(initial_temperature, dtype=np.float64, copy=True)

        self.cem43 = np.zeros_like(self.T, dtype=np.float64)

        if sensor_map and _has_field(sensor_map, "mask"):
            mask = np.array(sensor_map["mask"], dtype=bool)
            if mask.shape != grid_shape:
                raise ValueError("sensor.mask must match the computational grid shape.")
            if not np.logical_or(mask, ~mask).all():
                raise ValueError("sensor.mask must be binary.")
            self.sensor_mask_flat = mask.reshape(-1, order="F")
            self.num_sensor_points = int(self.sensor_mask_flat.sum())
            self.sensor_data = np.zeros((self.num_sensor_points, 0), dtype=np.float64)
        else:
            self.sensor_mask_flat = None
            self.num_sensor_points = 0
            self.sensor_data = np.zeros((0, 0), dtype=np.float64)

        self.time_steps_taken = 0

        self.k = np.fft.ifftshift(kgrid.k)
        self.k_squared = self.k ** 2
        self.kx_vec = np.fft.ifftshift(np.asarray(kgrid.k_vec.x)).reshape(self.Nx, 1, 1 if self.dim == 3 else 1)
        if self.dim >= 2:
            self.ky_vec = np.fft.ifftshift(np.asarray(kgrid.k_vec.y)).reshape(1, self.Ny, 1 if self.dim == 3 else 1)
        else:
            self.ky_vec = None
        if self.dim == 3:
            self.kz_vec = np.fft.ifftshift(np.asarray(kgrid.k_vec.z)).reshape(1, 1, self.Nz)
        else:
            self.kz_vec = None

    @staticmethod
    def _resolve_reference(reference: Union[str, float, np.ndarray], data: Union[float, np.ndarray]) -> float:
        arr = np.asarray(data, dtype=np.float64)
        if isinstance(reference, (int, float)):
            return float(reference)
        reference = str(reference).lower()
        if reference == "min":
            return float(np.min(arr))
        if reference == "mean":
            return float(np.mean(arr))
        if reference == "max":
            return float(np.max(arr))
        raise ValueError(f"Unknown reference mode {reference!r}")

    @property
    def dt_limit(self) -> float:
        diffusion_coeff = np.asarray(self.diffusion_p1) * np.asarray(self.diffusion_p2)
        diffusion_coeff = np.asarray(diffusion_coeff, dtype=np.float64)
        D_max = float(np.max(diffusion_coeff))
        k_max = float(np.max(self.k))
        if np.allclose(self.perfusion_coeff, 0.0):
            if self.diffusion_coeff_ref >= D_max / 2.0:
                return math.inf
            if k_max == 0:
                return math.inf
            return -math.log(1.0 - 2.0 * self.diffusion_coeff_ref / D_max) / (self.diffusion_coeff_ref * k_max**2)

        reference = self.diffusion_coeff_ref * k_max**2 + self.perfusion_coeff_ref
        condition = diffusion_coeff * k_max**2 + np.asarray(self.perfusion_coeff, dtype=np.float64)
        condition_max = float(np.max(condition))
        if reference >= 0.5 * condition_max:
            return math.inf
        return -math.log(1.0 - 2.0 * reference / condition_max) / reference

    @property
    def lesion_map(self) -> np.ndarray:
        """
        Binary lesion map based on thermal dose threshold.
        Returns True where CEM43 >= 240 minutes (tissue ablation threshold).
        """
        return self.cem43 >= 240

    def _fft(self, field: np.ndarray) -> np.ndarray:
        axes = tuple(range(self.dim))
        return fftn(field, axes=axes)

    def _ifft(self, field: np.ndarray) -> np.ndarray:
        axes = tuple(range(self.dim))
        result = ifftn(field, axes=axes)
        return result.real

    def _derivative_matrices(self, kappa: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        sqrt_kappa = np.sqrt(kappa)
        deriv_x = 1j * self.kx_vec * sqrt_kappa
        deriv_y = None
        deriv_z = None
        if self.dim >= 2:
            deriv_y = 1j * self.ky_vec * sqrt_kappa
        if self.dim == 3:
            deriv_z = 1j * self.kz_vec * sqrt_kappa
        return deriv_x, deriv_y, deriv_z

    def takeTimeStep(self, Nt: int, dt: float) -> None:
        if Nt <= 0:
            raise ValueError("Nt must be positive.")
        if dt <= 0:
            raise ValueError("dt must be positive.")

        if self.use_kspace:
            op = self.diffusion_coeff_ref * self.k_squared + self.perfusion_coeff_ref
            kappa = np.ones_like(self.k_squared, dtype=np.float64)
            non_zero = op != 0
            theta = dt * op[non_zero]
            kappa[non_zero] = (1.0 - np.exp(-theta)) / theta
        else:
            kappa = np.ones_like(self.k_squared, dtype=np.float64)

        deriv_x, deriv_y, deriv_z = self._derivative_matrices(kappa)

        if np.allclose(self.Q, 0.0):
            q_term = np.zeros_like(self.T)
        else:
            Q_fft = self._fft(np.asarray(self.Q, dtype=np.float64))
            if _is_scalar(self.q_scale_factor) and self.q_scale_factor == 0.0:
                q_term = self.diffusion_p1 * self._ifft(kappa * Q_fft)
            else:
                q_term = self.q_scale_factor * self._ifft(kappa * Q_fft)

        if np.allclose(self.perfusion_coeff, 0.0):
            use_perfusion = False
            p_term = np.zeros_like(self.T)
        else:
            use_perfusion = True
            delta = self.T - np.asarray(self.blood_ambient_temperature, dtype=np.float64)
            p_term = -self.perfusion_coeff * self._ifft(kappa * self._fft(delta))

        if self.num_sensor_points > 0:
            zeros = np.zeros((self.num_sensor_points, Nt), dtype=np.float64)
            self.sensor_data = np.hstack([self.sensor_data, zeros])

        for t_index in range(Nt):
            if self.flag_homogeneous:
                T_fft = self._fft(self.T)
                diff_term = self.diffusion_p1 * self.diffusion_p2 * self._ifft(-self.k_squared * kappa * T_fft)
            else:
                T_fft = self._fft(self.T)
                term = self._ifft(deriv_x * self._fft(self.diffusion_p2 * self._ifft(deriv_x * T_fft)))
                if self.dim >= 2 and deriv_y is not None:
                    term += self._ifft(deriv_y * self._fft(self.diffusion_p2 * self._ifft(deriv_y * T_fft)))
                if self.dim == 3 and deriv_z is not None:
                    term += self._ifft(deriv_z * self._fft(self.diffusion_p2 * self._ifft(deriv_z * T_fft)))
                diff_term = self.diffusion_p1 * term

            perf_term = p_term if use_perfusion else 0.0
            update = diff_term + perf_term + q_term
            self.T = self.T + dt * update

            base = np.zeros_like(self.T, dtype=np.float64)
            base[self.T >= 43.0] = 0.5
            between = (self.T >= 37.0) & (self.T < 43.0)
            base[between] = 0.25
            self.cem43 += (base ** (43.0 - self.T)) * (dt / 60.0)

            if self.num_sensor_points > 0:
                values = self.T.reshape(-1, order="F")[self.sensor_mask_flat]
                column_index = self.time_steps_taken + t_index
                self.sensor_data[:, column_index] = values

        self.time_steps_taken += Nt

    def plotTemp(self) -> None:
        raise NotImplementedError("plotTemp is not implemented in the Python port.")

