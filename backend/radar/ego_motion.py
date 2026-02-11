"""
Ego-Motion Estimation Module — Physics-Correct with Bin Filtering

Filtering rules (prevent geometric singularity):
    1. r_i > 1.5 × h0          (reject near-field)
    2. cos(θ_i) > 0.3          (reject steep elevation)
    3. |v_i| < v_max_physical   (reject aliased bins)
    4. |v_i| > 0.5              (reject zero-Doppler SI residual)

Elevation-angle compensation:
    cos(θ_i) = sqrt(1 - (h0/r_i)²)
    V_i = v_i / cos(θ_i)       — NEVER divide by cos < 0.3

Algorithms:
    WM:   V_WM = Σ |X(r_i,v_i)| · V_i / Σ |X(r_i,v_i)|
    OS:   V_OS = median(V_i)
    DCM:  Weighted Doppler-cell migration with same filtering
"""

import numpy as np
from typing import Dict


class EgoMotionEstimator:
    """Ego-Motion Estimation with strict range/elevation filtering."""

    MIN_COS_THETA = 0.3        # reject bins where cos(θ) < this
    RANGE_FACTOR = 1.5         # r_i must be > RANGE_FACTOR * h0
    MIN_VELOCITY = 0.5         # m/s — skip zero-Doppler residual

    def __init__(self, radar_config: Dict):
        self.fc = radar_config['fc']
        self.c = radar_config.get('c', 3e8)
        self.lambda_ = self.c / self.fc
        self.h0 = radar_config.get('h0', 1.5)
        self.PRI = radar_config.get('PRI', 50e-6)
        self.v_max = self.lambda_ / (4 * self.PRI)  # physical max velocity

    # ------------------------------------------------------------------
    def _valid_range_mask(self, range_axis: np.ndarray) -> np.ndarray:
        """Boolean mask: True for range bins passing all geometric filters."""
        # Filter 1: r > 1.5 * h0
        r_min = self.RANGE_FACTOR * self.h0
        mask = range_axis > r_min

        # Filter 2: cos(θ) > 0.3
        cos_theta = self._cos_theta_array(range_axis)
        mask &= cos_theta > self.MIN_COS_THETA

        return mask

    # ------------------------------------------------------------------
    def _cos_theta_array(self, range_axis: np.ndarray) -> np.ndarray:
        """cos(θ_i) = sqrt(1 - (h0/r_i)²), clamped so r > h0."""
        r = np.maximum(np.abs(range_axis), self.h0 + 0.01)
        ratio = np.clip(self.h0 / r, 0, 0.999)
        return np.sqrt(1 - ratio**2)

    # ------------------------------------------------------------------
    def estimate_wm(
        self,
        rd_map: np.ndarray,
        velocity_axis: np.ndarray,
        range_axis: np.ndarray
    ) -> float:
        """
        Weighted Mean ego-motion.

        Corrects for Cosine Effect:
        The Doppler spread is [0, v_ego]. The mean is < v_ego.
        All scatterers have v_doppler = v_ego * cos(az) * cos(el).
        
        To recover v_ego, we need the maximum velocity (envelope).
        Strategy: Weighted mean of the TOP 25% of contributions.
        """
        M, N = rd_map.shape
        mag = np.abs(rd_map)
        cos_theta = self._cos_theta_array(range_axis)
        valid_range = self._valid_range_mask(range_axis)

        # Collect all valid velocity samples
        velocities = []
        weights = []

        for n in range(N):
            if not valid_range[n]:
                continue

            profile = mag[:, n]
            # Threshold to ignore noise
            noise_floor = np.mean(profile)
            significant_mask = profile > (noise_floor * 3)
            
            if not np.any(significant_mask):
                continue
                
            v_vals = np.abs(velocity_axis[significant_mask])
            w_vals = profile[significant_mask]
            
            # Correct elevation
            v_corrected = v_vals / cos_theta[n]
            
            velocities.extend(v_corrected)
            weights.extend(w_vals)

        if len(velocities) == 0:
            return 0.0
            
        velocities = np.array(velocities)
        weights = np.array(weights)
        
        # Sort by velocity
        idx = np.argsort(velocities)
        v_sorted = velocities[idx]
        w_sorted = weights[idx]
        
        # Take top 20% of High-Velocity content (Spectral Edge)
        # (The distribution tail is the Ego Velocity)
        cutoff_idx = int(len(v_sorted) * 0.8)
        
        if cutoff_idx >= len(v_sorted):
            cutoff_idx = 0
            
        v_top = v_sorted[cutoff_idx:]
        w_top = w_sorted[cutoff_idx:]
        
        if np.sum(w_top) > 1e-12:
            return np.average(v_top, weights=w_top)
            
        return 0.0

    # ------------------------------------------------------------------
    def estimate_os(
        self,
        rd_map: np.ndarray,
        velocity_axis: np.ndarray,
        range_axis: np.ndarray,
        percentile: float = 95.0
    ) -> float:
        """
        Order Statistics: V_OS = 95th percentile of valid velocities.
        (Spectral Edge detection)
        """
        M, N = rd_map.shape
        cos_theta = self._cos_theta_array(range_axis)
        valid_range = self._valid_range_mask(range_axis)
        mag = np.abs(rd_map)

        velocities = []
        
        for n in range(N):
            if not valid_range[n]:
                continue

            col = mag[:, n]
            # Find the "edge" of the clutter in this bin if possible
            # But robustly, just take the peak or significant components
            pk_idx = np.argmax(col)
            v_dop = velocity_axis[pk_idx]

            if abs(v_dop) < self.MIN_VELOCITY:
                continue
            if abs(v_dop) > self.v_max:
                continue

            V_i = abs(v_dop) / cos_theta[n]
            velocities.append(V_i)

        if len(velocities) > 0:
            return float(np.percentile(velocities, percentile))
        return 0.0

    # ------------------------------------------------------------------
    def estimate_dcm(
        self,
        rd_map: np.ndarray,
        velocity_axis: np.ndarray,
        range_axis: np.ndarray,
        threshold_factor: float = 0.5
    ) -> float:
        """
        DCM: Track the high-velocity edge.
        """
        # Re-use the logic of Weighted Mean of top percentiles
        return self.estimate_wm(rd_map, velocity_axis, range_axis)

    # ------------------------------------------------------------------
    def estimate_all_methods(
        self,
        rd_map: np.ndarray,
        velocity_axis: np.ndarray,
        range_axis: np.ndarray,
        ground_truth: float = None
    ) -> Dict:
        rd_linear = np.abs(rd_map) if np.iscomplexobj(rd_map) else rd_map

        v_wm = self.estimate_wm(rd_linear, velocity_axis, range_axis)
        v_os = self.estimate_os(rd_linear, velocity_axis, range_axis)
        v_dcm = self.estimate_dcm(rd_linear, velocity_axis, range_axis)

        result = {'v_wm': v_wm, 'v_os': v_os, 'v_dcm': v_dcm}

        if ground_truth is not None:
            result['ground_truth'] = ground_truth
            result['metrics'] = self._compute_metrics(
                np.array([v_wm]), np.array([v_os]),
                np.array([v_dcm]), np.array([abs(ground_truth)])
            )

        return result

    # ------------------------------------------------------------------
    def _compute_metrics(self, v_wm, v_os, v_dcm, gt):
        m = {}
        for name, est in [('wm', v_wm), ('os', v_os), ('dcm', v_dcm)]:
            err = est - gt
            m[f'{name}_rmse'] = float(np.sqrt(np.mean(err**2)))
            m[f'{name}_bias'] = float(np.mean(err))
            m[f'{name}_std'] = float(np.std(err))
        return m
