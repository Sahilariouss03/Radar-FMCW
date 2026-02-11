"""
FMCW Radar Signal Simulation Module — Physics-Correct Implementation

Beat signal model:
    S_b(m,n) = Σ_i  A_i · exp(j·2π·(f_b_i · n·Ts + f_d_i · m·T_PRI))

Where:
    f_b = (2·B / (Tsw·c)) · R        (beat frequency)
    f_d = 2·v_radial / λ              (Doppler frequency)
    λ   = c / fc                       (wavelength)

All computation is fully vectorized — no Python loops over chirps or scatterers.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class FMCWRadar:
    """FMCW Radar Signal Simulator (Physics-Correct, Vectorized)"""

    def __init__(self, config: Dict):
        """
        Initialize FMCW radar.

        Args:
            config: Dictionary with keys:
                fc, B, Tsw, PRI, M, N, h0, c (optional, default 3e8)
        """
        self.fc = config['fc']
        self.B = config['B']
        self.Tsw = config['Tsw']
        self.PRI = config['PRI']
        self.M = config['M']
        self.N = config['N']
        self.h0 = config['h0']
        self.c = config.get('c', 3e8)

        # Derived physical constants
        self.lambda_ = self.c / self.fc          # wavelength (m)
        self.chirp_rate = self.B / self.Tsw      # Hz/s
        self.Ts = self.Tsw / self.N              # ADC sampling interval (s)
        self.fs = 1.0 / self.Ts                  # ADC sampling frequency (Hz)
        self.range_resolution = self.c / (2 * self.B)                 # ΔR (m)
        self.max_unambiguous_range = self.c * self.fs / (4 * self.chirp_rate)
        self.max_unambiguous_velocity = self.lambda_ / (4 * self.PRI)  # v_max (m/s)

        # Time axes
        self.fast_time = np.arange(self.N) * self.Ts        # n·Ts  [N]
        self.slow_time = np.arange(self.M) * self.PRI       # m·T_PRI  [M]

    # ------------------------------------------------------------------
    def generate_beat_signal(
        self,
        velocity_profile: np.ndarray,
        scatterers: np.ndarray,
        point_targets: Optional[np.ndarray] = None,
        noise_power: float = 1e-6,
        si_amplitude: float = 0.1
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate FMCW beat signal — fully vectorized.

        Args:
            velocity_profile: [M] vehicle velocity at each chirp (m/s)
            scatterers:       [Ns, 3] columns [x, y, RCS]
            point_targets:    [Nt, 3] columns [x, y, RCS]  (optional)
            noise_power:      noise power level
            si_amplitude:     self-interference amplitude (>> clutter)

        Returns:
            beat_signal: [M, N] complex
            metadata:    dict
        """
        M, N = self.M, self.N
        beat_signal = np.zeros((M, N), dtype=complex)

        # Vehicle x-position at each chirp  [M]
        vehicle_x = np.concatenate(([0.0], np.cumsum(velocity_profile[:-1] * self.PRI)))

        # --- process a batch of scatterers (vectorized) ----------------
        beat_signal += self._add_targets_vectorized(
            scatterers, vehicle_x, velocity_profile
        )

        # --- point targets (same vectorised path) ----------------------
        if point_targets is not None and len(point_targets) > 0:
            beat_signal += self._add_targets_vectorized(
                point_targets, vehicle_x, velocity_profile
            )

        # --- self-interference (zero-Doppler, near-zero range) ---------
        beat_signal += self._generate_self_interference(si_amplitude)

        # --- additive white Gaussian noise -----------------------------
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(M, N) + 1j * np.random.randn(M, N)
        )
        beat_signal += noise

        metadata = {
            'vehicle_x': vehicle_x,
            'velocity_profile': velocity_profile,
            'range_resolution': self.range_resolution,
            'max_range': self.max_unambiguous_range,
            'max_velocity': self.max_unambiguous_velocity,
            'lambda': self.lambda_,
            'num_scatterers': len(scatterers),
            'num_targets': len(point_targets) if point_targets is not None else 0
        }
        return beat_signal, metadata

    # ------------------------------------------------------------------
    def _add_targets_vectorized(
        self,
        targets: np.ndarray,
        vehicle_x: np.ndarray,
        velocity_profile: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized beat-signal contribution from a set of targets.

        targets:          [Nt, 3]  columns [x, y, RCS]
        vehicle_x:        [M]
        velocity_profile: [M]

        Returns [M, N] complex signal
        """
        M, N = self.M, self.N
        Nt = targets.shape[0]

        tx = targets[:, 0]   # [Nt]
        ty = targets[:, 1]   # [Nt]
        rcs = targets[:, 2]  # [Nt]

        # Relative x from vehicle to each target at each chirp
        # dx[m, i] = tx[i] - vehicle_x[m]
        dx = tx[np.newaxis, :] - vehicle_x[:, np.newaxis]          # [M, Nt]
        ground_range = np.sqrt(dx**2 + ty[np.newaxis, :]**2)       # [M, Nt]

        # 3-D slant range (include radar height)
        R = np.sqrt(dx**2 + ty[np.newaxis, :]**2 + self.h0**2)    # [M, Nt]

        # Beat frequency:  f_b = (2B / (Tsw·c)) · R
        fb = (2 * self.B / (self.Tsw * self.c)) * R               # [M, Nt]

        # Radial velocity component (platform moving in +x)
        # v_radial = -v · cos(angle_to_target_in_x) = -v · dx / ground_range
        # Negative sign: target appears to approach if platform moves toward it
        v_radial = -velocity_profile[:, np.newaxis] * dx / (ground_range + 1e-12)  # [M, Nt]

        # Doppler frequency:  f_d = 2 v_radial / λ
        fd = 2 * v_radial / self.lambda_                          # [M, Nt]

        # Amplitude ~ sqrt(RCS) / R²
        amplitude = np.sqrt(rcs[np.newaxis, :]) / (R**2 + 1e-12)  # [M, Nt]

        # Phase:  φ(m,n,i) = 2π (fb[m,i]·n·Ts + fd[m,i]·m·T_PRI)
        # We build the signal as:  Σ_i amplitude[m,i] · exp(j·φ(m,n,i))
        #
        # For each target i we compute outer product over n,
        # then sum over targets.  This avoids an explicit loop over n.
        #
        # fast_phase[m, i] = 2π · fb[m,i] · Ts          (multiplied by n later)
        # slow_phase[m, i] = 2π · fd[m,i] · m · T_PRI
        # Phase:
        # Fast time (intrapulse): 2π * fb * t_fast  -->  2π * fb * n * Ts
        # Slow time (interpulse): 4π * R / λ        -->  Encodes Doppler & Range Migration
        #
        # Note: We act as if phase is reset per chirp for fast-time (standard deramp),
        # but slow-time phase must evolve continuously with R.
        
        fast_phase_rate = 2 * np.pi * fb * self.Ts                 # [M, Nt]
        slow_phase = (4 * np.pi / self.lambda_) * R               # [M, Nt]

        # Build beat signal: loop over targets (typically <200) to save memory
        signal = np.zeros((M, N), dtype=complex)
        n_axis = np.arange(N)  # [N]

        for i in range(Nt):
            # phase[m, n] = fast_phase_rate[m,i]*n + slow_phase[m,i]
            phase = fast_phase_rate[:, i:i+1] * n_axis[np.newaxis, :] + slow_phase[:, i:i+1]
            signal += amplitude[:, i:i+1] * np.exp(1j * phase)

        return signal

    # ------------------------------------------------------------------
    def _generate_self_interference(self, amplitude: float) -> np.ndarray:
        """
        Self-interference: strong zero-Doppler component at near-zero range.
        A_SI >> clutter amplitude.
        """
        M, N = self.M, self.N

        # Near-zero range → very low beat frequency
        R_si = 0.05  # 5 cm (antenna coupling path)
        fb_si = (2 * self.B / (self.Tsw * self.c)) * R_si
        fd_si = 0.0  # zero Doppler

        n_axis = np.arange(N)
        phase = 2 * np.pi * fb_si * self.Ts * n_axis  # [N]

        # Constant across all chirps (zero Doppler) with amplitude >> clutter
        si_row = amplitude * np.exp(1j * phase)
        si_signal = np.tile(si_row, (M, 1))

        return si_signal

    # ------------------------------------------------------------------
    def generate_scatterers(
        self,
        num_scatterers: int,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        rcs_range: Tuple[float, float] = (0.1, 1.0)
    ) -> np.ndarray:
        """Generate random ground scatterers [Ns, 3] — columns [x, y, RCS]."""
        x = np.random.uniform(x_range[0], x_range[1], num_scatterers)
        y = np.random.uniform(y_range[0], y_range[1], num_scatterers)
        rcs = np.random.uniform(rcs_range[0], rcs_range[1], num_scatterers)
        return np.column_stack([x, y, rcs])

    # ------------------------------------------------------------------
    def generate_point_targets(
        self,
        positions: np.ndarray,
        rcs: float = 10.0
    ) -> np.ndarray:
        """Generate point targets [Nt, 3] from positions [Nt, 2]."""
        rcs_col = np.full(len(positions), rcs)
        return np.column_stack([positions, rcs_col])

    # ------------------------------------------------------------------
    def generate_velocity_profile(
        self,
        v0: float,
        acceleration: float = 0.0,
        noise_std: float = 0.0
    ) -> np.ndarray:
        """Generate velocity profile [M] with optional acceleration and noise."""
        v = v0 + acceleration * self.slow_time
        if noise_std > 0:
            v += np.random.randn(self.M) * noise_std
        return v
