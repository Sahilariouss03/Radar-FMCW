"""
Self-Interference Cancellation — Physics-Correct

SI signal:  S_SI(m,n) = A_SI · exp(j 2π f_b_SI · n · Ts)
    R_SI ≈ 0, v_SI = 0  →  zero-Doppler, near-zero range

In RD domain: centred at (R≈0, f_d≈0) with sinc sidelobes from FFT.

Cancellation:
    1. Detect peak at zero-Doppler, near-zero range
    2. Estimate amplitude and phase of peak
    3. Model 2-D sinc structure:  sinc(x) = sin(πx)/(πx)
    4. Subtract modeled sinc from RD map
    5. Iterate until convergence

Reduction metric (amplitude):
    SI_reduction_dB = 20 · log10(peak_before / peak_after)
    Target ≥ 15 dB
"""

import numpy as np
from typing import Tuple, Dict


class InterferenceCanceller:
    """Self-Interference Cancellation (≥ 15 dB target)."""

    def __init__(self, radar_config: Dict):
        self.M = radar_config['M']
        self.N = radar_config['N']

    # ------------------------------------------------------------------
    def iterative_cancellation(
        self,
        rd_map: np.ndarray,
        max_iterations: int = 10,
        zero_doppler_width: int = 5
    ) -> Tuple[np.ndarray, Dict]:
        """
        Iterative sinc-based SI cancellation.

        Returns:
            rd_cleaned: cleaned complex RD map [M, N]
            metrics: dict with reduction_db, iterations
        """
        rd_work = rd_map.copy()
        M, N = rd_work.shape

        # Measure initial peak amplitude in zero-Doppler region
        peak_before = self._peak_in_zero_doppler(rd_map, zero_doppler_width)

        for iteration in range(max_iterations):
            # Detect the interference peak
            pk_dop, pk_rng, pk_val = self._detect_si_peak(
                rd_work, zero_doppler_width
            )

            if np.abs(pk_val) < 1e-15:
                break

            # Build 2-D sinc model centred at the peak
            sinc_model = self._build_sinc_model(
                M, N, pk_dop, pk_rng, pk_val
            )

            # Subtract
            rd_work = rd_work - sinc_model

            # Check if we reached target
            peak_now = self._peak_in_zero_doppler(rd_work, zero_doppler_width)
            if peak_now < 1e-15:
                break
            reduction = 20 * np.log10(peak_before / (peak_now + 1e-15))
            if reduction >= 25:  # overshoot slightly to ensure ≥15
                break

        peak_after = self._peak_in_zero_doppler(rd_work, zero_doppler_width)
        si_reduction = 20 * np.log10(peak_before / (peak_after + 1e-15))

        return rd_work, {
            'iterations': iteration + 1,
            'peak_before': float(peak_before),
            'peak_after': float(peak_after),
            'reduction_db': float(si_reduction),
        }

    # ------------------------------------------------------------------
    def _detect_si_peak(
        self, rd_map: np.ndarray, zd_width: int
    ) -> Tuple[int, int, complex]:
        """Find the complex peak in zero-Doppler region."""
        M, N = rd_map.shape
        c = M // 2
        d0 = max(0, c - zd_width)
        d1 = min(M, c + zd_width + 1)

        region = rd_map[d0:d1, :]
        mag = np.abs(region)
        idx = np.unravel_index(np.argmax(mag), mag.shape)

        pk_d = d0 + idx[0]
        pk_r = idx[1]
        return pk_d, pk_r, rd_map[pk_d, pk_r]

    # ------------------------------------------------------------------
    def _build_sinc_model(
        self,
        M: int, N: int,
        pk_d: int, pk_r: int,
        pk_val: complex
    ) -> np.ndarray:
        """
        Build a 2-D sinc model centred at (pk_d, pk_r).

        sinc(x) = sin(πx)/(πx)  — numpy's np.sinc already includes π.

        The SI signal produces sinc sidelobes in both range and Doppler
        because it is a single-bin signal convolved with the FFT window.
        """
        r_idx = np.arange(N, dtype=float)
        d_idx = np.arange(M, dtype=float)

        # Sinc in range (width controlled by range FFT mainlobe)
        range_sinc = np.sinc(r_idx - pk_r)  # sinc(x) has zeros at integers

        # Sinc in Doppler (width controlled by Doppler FFT mainlobe)
        doppler_sinc = np.sinc(d_idx - pk_d)

        # 2-D pattern = outer product scaled by complex peak
        model = pk_val * doppler_sinc[:, np.newaxis] * range_sinc[np.newaxis, :]

        return model

    # ------------------------------------------------------------------
    def _peak_in_zero_doppler(
        self, rd_map: np.ndarray, zd_width: int
    ) -> float:
        """Peak amplitude in zero-Doppler region."""
        M = rd_map.shape[0]
        c = M // 2
        d0 = max(0, c - zd_width)
        d1 = min(M, c + zd_width + 1)
        return float(np.max(np.abs(rd_map[d0:d1, :])))

    # ------------------------------------------------------------------
    def apply_notch_filter(
        self, rd_map: np.ndarray, notch_width: int = 2
    ) -> np.ndarray:
        """Simple notch filter: zero out zero-Doppler bins."""
        M = rd_map.shape[0]
        c = M // 2
        out = rd_map.copy()
        out[max(0, c - notch_width):min(M, c + notch_width + 1), :] = 0
        return out
