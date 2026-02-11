"""
SAR Image Formation — Physics-Correct Phase-History Focusing

True SAR requires coherent phase accumulation:

    Platform position:  x[m] = Σ V[m] × T_PRI
    Instantaneous range: R[m] = sqrt((x[m]-x_t)² + R_c²)
    Phase history:       φ[m] = 4π R[m] / λ
    Azimuth matched filter: h[m] = exp(-j φ[m])

A single point target must produce:
    - A single sharp peak
    - Symmetric sidelobes
    - PSR > 10 dB

Azimuth resolution: ρ_az = λ / (2L)  (theoretical)
Measured resolution: 3 dB width from actual SAR image slice
"""

import numpy as np
from typing import Dict, Tuple
from scipy.interpolate import interp1d


class SARProcessor:
    """SAR Image Formation with Phase-History Matched Filtering."""

    def __init__(self, radar_config: Dict):
        self.fc = radar_config['fc']
        self.B = radar_config['B']
        self.PRI = radar_config['PRI']
        self.M = radar_config['M']
        self.N = radar_config['N']
        self.c = radar_config.get('c', 3e8)
        self.lambda_ = self.c / self.fc
        self.Tsw = radar_config.get('Tsw', 40e-6)
        self.range_resolution = self.c / (2 * self.B)

    # ------------------------------------------------------------------
    def rda_conventional(
        self,
        beat_signal: np.ndarray,
        v_platform: float,
        synthetic_aperture_length: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Phase-history based SAR focusing (constant velocity).

        1. Range compression (FFT along fast-time)
        2. For each range bin, apply azimuth matched filter
           using phase history φ[m] = 4π R[m] / λ
        """
        M, N = beat_signal.shape

        # Step 1: Range compression
        range_compressed = np.fft.fft(beat_signal, axis=1)

        # Step 2: Azimuth compression via phase-history matched filtering
        sar_complex = self._phase_history_focus(
            range_compressed, v_platform
        )

        sar_image = np.abs(sar_complex)
        range_axis = self._generate_range_axis()
        azimuth_axis = self._generate_azimuth_axis(v_platform)

        return sar_image, range_axis, azimuth_axis

    # ------------------------------------------------------------------
    def rda_interpolated(
        self,
        beat_signal: np.ndarray,
        velocity_profile: np.ndarray,
        synthetic_aperture_length: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Phase-history SAR for non-uniform velocity.
        Uses actual velocity profile for platform positions.
        """
        M, N = beat_signal.shape

        # Range compression
        range_compressed = np.fft.fft(beat_signal, axis=1)

        # Platform positions from actual velocity profile
        positions = np.concatenate(([0.0], np.cumsum(velocity_profile[:-1] * self.PRI)))

        # Azimuth compression with actual positions
        sar_complex = self._phase_history_focus_with_positions(
            range_compressed, positions
        )

        sar_image = np.abs(sar_complex)
        v_mean = float(np.mean(velocity_profile))
        range_axis = self._generate_range_axis()
        azimuth_axis = self._generate_azimuth_axis_from_positions(positions)

        return sar_image, range_axis, azimuth_axis

    # ------------------------------------------------------------------
    def _phase_history_focus(
        self,
        range_compressed: np.ndarray,
        v_platform: float
    ) -> np.ndarray:
        """
        Azimuth matched filtering using phase-history model.

        For each range bin at range R_c:
            Platform positions: x[m] = m × v × T_PRI
            Instantaneous range: R[m] = sqrt(x[m]² + R_c²)
                (target at azimuth centre, x_t = x_mid)
            Phase: φ[m] = 4π R[m] / λ
            Matched filter: h[m] = exp(-j φ[m])
        """
        M, N = range_compressed.shape

        # Platform positions relative to mid-aperture
        m_idx = np.arange(M, dtype=float)
        x_platform = (m_idx - M/2) * v_platform * self.PRI  # [M] centred

        sar_out = np.zeros_like(range_compressed)

        for n in range(N):
            R_c = max((n + 0.5) * self.range_resolution, 0.1)  # range to this bin

            # Instantaneous slant range (target at azimuth centre x_t=0)
            R_m = np.sqrt(x_platform**2 + R_c**2)  # [M]

            # Phase history
            phi = 4 * np.pi * R_m / self.lambda_  # [M]

            # Match signal phase history (exp(jφ)) for correlation
            h_m = np.exp(1j * phi)  # [M]

            # Apply matched filter in azimuth (correlation = IFFT(FFT * conj(FFT_h)))
            sig_fft = np.fft.fft(range_compressed[:, n])
            h_fft = np.fft.fft(h_m)
            sar_out[:, n] = np.fft.ifft(sig_fft * np.conj(h_fft))

        return sar_out

    # ------------------------------------------------------------------
    def _phase_history_focus_with_positions(
        self,
        range_compressed: np.ndarray,
        positions: np.ndarray
    ) -> np.ndarray:
        """
        Same as _phase_history_focus but uses actual platform positions
        instead of assuming constant velocity.
        """
        M, N = range_compressed.shape

        # Centre positions around mid-aperture
        x_mid = (positions[0] + positions[-1]) / 2
        x_centered = positions - x_mid  # [M]

        sar_out = np.zeros_like(range_compressed)

        for n in range(N):
            R_c = max((n + 0.5) * self.range_resolution, 0.1)

            R_m = np.sqrt(x_centered**2 + R_c**2)
            phi = 4 * np.pi * R_m / self.lambda_
            h_m = np.exp(1j * phi)

            sig_fft = np.fft.fft(range_compressed[:, n])
            h_fft = np.fft.fft(h_m)
            sar_out[:, n] = np.fft.ifft(sig_fft * np.conj(h_fft))

        return sar_out

    # ------------------------------------------------------------------
    def _generate_range_axis(self) -> np.ndarray:
        """R[i] = i × ΔR"""
        return np.arange(self.N) * self.range_resolution

    def _generate_azimuth_axis(self, v_platform: float) -> np.ndarray:
        """Azimuth axis: Δx = V_avg × T_PRI, centred."""
        x = np.arange(self.M) * v_platform * self.PRI
        return x - np.mean(x)  # centre around 0

    def _generate_azimuth_axis_from_positions(
        self, positions: np.ndarray
    ) -> np.ndarray:
        """Azimuth axis from integrated positions, centred."""
        return positions - np.mean(positions)

    # ------------------------------------------------------------------
    def compute_measured_resolution(
        self,
        sar_image: np.ndarray,
        azimuth_axis: np.ndarray,
        target_range_idx: int = None
    ) -> float:
        """
        MEASURED azimuth resolution: 3 dB width from SAR image.
        NOT the theoretical ρ = λ/(2L).

        1. Extract azimuth slice at target range
        2. Find peak
        3. Find width where amplitude = peak / √2
        4. Return width in metres
        """
        if target_range_idx is None:
            target_range_idx = np.argmax(np.max(sar_image, axis=0))

        profile = sar_image[:, target_range_idx]
        peak_idx = np.argmax(profile)
        peak_val = profile[peak_idx]

        if peak_val < 1e-12:
            return float('inf')

        threshold = peak_val / np.sqrt(2)  # -3 dB

        # Walk left from peak
        left = peak_idx
        while left > 0 and profile[left] > threshold:
            left -= 1

        # Walk right from peak
        right = peak_idx
        while right < len(profile) - 1 and profile[right] > threshold:
            right += 1

        # Sub-bin interpolation
        if left < peak_idx and right > peak_idx and len(azimuth_axis) > right:
            left_pos = self._interp_crossing(
                azimuth_axis[left], azimuth_axis[min(left+1, len(azimuth_axis)-1)],
                profile[left], profile[min(left+1, len(profile)-1)],
                threshold
            )
            right_pos = self._interp_crossing(
                azimuth_axis[max(right-1, 0)], azimuth_axis[right],
                profile[max(right-1, 0)], profile[right],
                threshold
            )
            return abs(right_pos - left_pos)

        return abs(azimuth_axis[right] - azimuth_axis[left])

    # ------------------------------------------------------------------
    def _interp_crossing(self, x0, x1, y0, y1, level):
        """Linear interpolation for threshold crossing."""
        if abs(y1 - y0) < 1e-15:
            return (x0 + x1) / 2
        t = (level - y0) / (y1 - y0)
        return x0 + t * (x1 - x0)
