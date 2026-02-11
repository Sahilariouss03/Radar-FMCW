"""
Range-Doppler Processing Module — Physics-Correct Implementation

Range axis:     R[i]  = i · ΔR,  ΔR = c / (2B)
Doppler axis:   f_d[k] = fftfreq(M, T_PRI)  then fftshift
Velocity axis:  v[k]  = (λ/2) · f_d[k]
Max velocity:   v_max = λ / (4·T_PRI)
"""

import numpy as np
from typing import Tuple, Dict


class RangeDopplerProcessor:
    """Range-Doppler Map Generator (Physics-Correct)"""

    def __init__(self, radar_config: Dict):
        """
        Args:
            radar_config: dict with keys fc, B, Tsw, PRI, N, M, c (optional)
        """
        self.fc = radar_config['fc']
        self.B = radar_config['B']
        self.Tsw = radar_config['Tsw']
        self.PRI = radar_config['PRI']
        self.N = radar_config['N']
        self.M = radar_config['M']
        self.c = radar_config.get('c', 3e8)

        # Derived
        self.lambda_ = self.c / self.fc
        self.Ts = self.Tsw / self.N
        self.fs = 1.0 / self.Ts
        self.range_resolution = self.c / (2 * self.B)           # ΔR
        self.max_velocity = self.lambda_ / (4 * self.PRI)       # v_max

    # ------------------------------------------------------------------
    def compute_range_doppler(
        self,
        beat_signal: np.ndarray,
        window_range: str = 'hann',
        window_doppler: str = 'hann'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Range-Doppler map via 2-D FFT.

        Returns:
            rd_map_db:      [M, N] magnitude in dB
            range_axis:     [N] in metres
            velocity_axis:  [M] in m/s (centred around 0)
        """
        M, N = beat_signal.shape

        # --- Apply range window (fast-time) ---
        if window_range:
            w_r = self._get_window(window_range, N)
            beat_signal = beat_signal * w_r[np.newaxis, :]

        # Range FFT (fast-time, axis=1)
        range_fft = np.fft.fft(beat_signal, axis=1)

        # --- Apply Doppler window (slow-time) ---
        if window_doppler:
            w_d = self._get_window(window_doppler, M)
            range_fft = range_fft * w_d[:, np.newaxis]

        # Doppler FFT (slow-time, axis=0) + fftshift to centre zero-Doppler
        rd_complex = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)

        # Magnitude in dB
        rd_mag = np.abs(rd_complex)
        rd_map_db = 20 * np.log10(rd_mag + 1e-12)

        # Axes
        range_axis = self._generate_range_axis()
        velocity_axis = self._generate_velocity_axis()

        return rd_map_db, range_axis, velocity_axis

    # ------------------------------------------------------------------
    def compute_range_doppler_complex(
        self,
        beat_signal: np.ndarray,
        window_range: str = 'hann',
        window_doppler: str = 'hann'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Same as compute_range_doppler but returns complex RD map
        (needed by interference canceller).
        """
        M, N = beat_signal.shape

        if window_range:
            w_r = self._get_window(window_range, N)
            beat_signal = beat_signal * w_r[np.newaxis, :]

        range_fft = np.fft.fft(beat_signal, axis=1)

        if window_doppler:
            w_d = self._get_window(window_doppler, M)
            range_fft = range_fft * w_d[:, np.newaxis]

        rd_complex = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)

        range_axis = self._generate_range_axis()
        velocity_axis = self._generate_velocity_axis()

        return rd_complex, range_axis, velocity_axis

    # ------------------------------------------------------------------
    def _generate_range_axis(self) -> np.ndarray:
        """
        Range axis:  R[i] = i · ΔR,  ΔR = c/(2B)

        Returns [N] in metres.
        """
        return np.arange(self.N) * self.range_resolution

    # ------------------------------------------------------------------
    def _generate_velocity_axis(self) -> np.ndarray:
        """
        Velocity axis from Doppler FFT frequencies.

        f_d[k] = fftfreq(M, T_PRI)  →  v[k] = (λ/2) · f_d[k]

        After fftshift the axis is centred around zero.
        Returns [M] in m/s.
        """
        fd = np.fft.fftshift(np.fft.fftfreq(self.M, self.PRI))
        return (self.lambda_ / 2) * fd

    # ------------------------------------------------------------------
    def _get_window(self, window_type: str, length: int) -> np.ndarray:
        if window_type == 'hann':
            return np.hanning(length)
        elif window_type == 'hamming':
            return np.hamming(length)
        elif window_type == 'blackman':
            return np.blackman(length)
        else:
            return np.ones(length)

    # ------------------------------------------------------------------
    def detect_peaks(
        self,
        rd_map: np.ndarray,
        threshold_db: float = -20,
        num_peaks: int = 10
    ) -> np.ndarray:
        """Detect peaks in RD map above threshold relative to max."""
        rd_norm = rd_map - np.max(rd_map)
        mask = rd_norm > threshold_db
        d_idx, r_idx = np.where(mask)
        values = rd_norm[d_idx, r_idx]
        order = np.argsort(values)[::-1]
        n_det = min(num_peaks, len(order))
        return np.column_stack([d_idx[order[:n_det]], r_idx[order[:n_det]]])

    # ------------------------------------------------------------------
    def extract_range_profile(self, rd_map: np.ndarray, doppler_idx: int) -> np.ndarray:
        return rd_map[doppler_idx, :]

    def extract_doppler_profile(self, rd_map: np.ndarray, range_idx: int) -> np.ndarray:
        return rd_map[:, range_idx]
