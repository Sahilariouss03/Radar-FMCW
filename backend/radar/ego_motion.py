"""
Ego-Motion Estimation Module

Implements three ego-motion estimation algorithms:
1. Weighted Mean (WM): Weighted average of Doppler velocities
2. Order Statistics (OS): Median-based robust estimation
3. Doppler Cell Migration (DCM): Cell migration tracking

Includes performance metrics computation (RMSE, bias, standard deviation).
"""

import numpy as np
from typing import Dict, Tuple, List


class EgoMotionEstimator:
    """Ego-Motion Estimation Processor"""
    
    def __init__(self, radar_config: Dict):
        """
        Initialize ego-motion estimator.
        
        Args:
            radar_config: Dictionary containing:
                - fc: Carrier frequency (Hz)
                - c: Speed of light (m/s)
        """
        self.fc = radar_config['fc']
        self.c = radar_config.get('c', 3e8)
        
    def estimate_wm(
        self,
        rd_map: np.ndarray,
        velocity_axis: np.ndarray,
        range_bins: np.ndarray = None
    ) -> float:
        """
        Weighted Mean (WM) ego-motion estimation.
        
        Args:
            rd_map: Range-Doppler map (linear magnitude)
            velocity_axis: Velocity values [M]
            range_bins: Range bins to use (None = all bins)
            
        Returns:
            Estimated ego velocity (m/s)
        """
        if range_bins is None:
            range_bins = np.arange(rd_map.shape[1])
        
        # Sum power across selected range bins
        doppler_profile = np.sum(np.abs(rd_map[:, range_bins]) ** 2, axis=1)
        
        # Weighted mean
        total_power = np.sum(doppler_profile)
        if total_power > 0:
            v_ego = np.sum(velocity_axis * doppler_profile) / total_power
        else:
            v_ego = 0.0
        
        return v_ego
    
    def estimate_os(
        self,
        rd_map: np.ndarray,
        velocity_axis: np.ndarray,
        range_axis: np.ndarray,
        percentile: float = 50.0
    ) -> float:
        """
        Order Statistics (OS) ego-motion estimation using median.
        
        Args:
            rd_map: Range-Doppler map (linear magnitude)
            velocity_axis: Velocity values [M]
            range_axis: Range values [N]
            percentile: Percentile to use (50 = median)
            
        Returns:
            Estimated ego velocity (m/s)
        """
        M, N = rd_map.shape
        
        # For each range bin, find peak Doppler
        peak_velocities = []
        
        for n in range(N):
            doppler_profile = np.abs(rd_map[:, n])
            peak_idx = np.argmax(doppler_profile)
            peak_velocities.append(velocity_axis[peak_idx])
        
        # Compute percentile
        v_ego = np.percentile(peak_velocities, percentile)
        
        return v_ego
    
    def estimate_dcm(
        self,
        rd_map: np.ndarray,
        velocity_axis: np.ndarray,
        range_bins: np.ndarray = None,
        threshold_factor: float = 0.5
    ) -> float:
        """
        Doppler Cell Migration (DCM) ego-motion estimation.
        
        Tracks migration of Doppler cells across range bins.
        
        Args:
            rd_map: Range-Doppler map (linear magnitude)
            velocity_axis: Velocity values [M]
            range_bins: Range bins to analyze
            threshold_factor: Threshold relative to max for cell detection
            
        Returns:
            Estimated ego velocity (m/s)
        """
        if range_bins is None:
            range_bins = np.arange(rd_map.shape[1])
        
        # Find dominant Doppler bin for each range
        doppler_indices = []
        weights = []
        
        for n in range_bins:
            doppler_profile = np.abs(rd_map[:, n])
            max_val = np.max(doppler_profile)
            
            if max_val > threshold_factor * np.max(rd_map):
                peak_idx = np.argmax(doppler_profile)
                doppler_indices.append(peak_idx)
                weights.append(max_val)
        
        if len(doppler_indices) > 0:
            # Weighted average of peak Doppler bins
            doppler_indices = np.array(doppler_indices)
            weights = np.array(weights)
            avg_doppler_idx = int(np.average(doppler_indices, weights=weights))
            v_ego = velocity_axis[avg_doppler_idx]
        else:
            v_ego = 0.0
        
        return v_ego
    
    def estimate_all_methods(
        self,
        rd_map: np.ndarray,
        velocity_axis: np.ndarray,
        range_axis: np.ndarray,
        ground_truth: float = None
    ) -> Dict:
        """
        Estimate ego-motion using all three methods.
        
        Args:
            rd_map: Range-Doppler map (linear magnitude)
            velocity_axis: Velocity values [M]
            range_axis: Range values [N]
            ground_truth: Ground truth velocity (optional, for metrics)
            
        Returns:
            Dictionary with estimates and metrics
        """
        # Estimates
        v_wm = self.estimate_wm(rd_map, velocity_axis)
        v_os = self.estimate_os(rd_map, velocity_axis, range_axis)
        v_dcm = self.estimate_dcm(rd_map, velocity_axis)
        
        results = {
            'wm': v_wm,
            'os': v_os,
            'dcm': v_dcm,
            'ground_truth': ground_truth
        }
        
        # Compute metrics if ground truth available
        if ground_truth is not None:
            results['errors'] = {
                'wm': v_wm - ground_truth,
                'os': v_os - ground_truth,
                'dcm': v_dcm - ground_truth
            }
            
            results['metrics'] = {
                'wm_rmse': np.abs(v_wm - ground_truth),
                'os_rmse': np.abs(v_os - ground_truth),
                'dcm_rmse': np.abs(v_dcm - ground_truth)
            }
        
        return results
    
    def estimate_time_series(
        self,
        beat_signals: List[np.ndarray],
        velocity_axis: np.ndarray,
        range_axis: np.ndarray,
        ground_truth_series: np.ndarray = None
    ) -> Dict:
        """
        Estimate ego-motion over time series of Range-Doppler maps.
        
        Args:
            beat_signals: List of beat signal matrices
            velocity_axis: Velocity values
            range_axis: Range values
            ground_truth_series: Ground truth velocities over time
            
        Returns:
            Dictionary with time series estimates and metrics
        """
        num_frames = len(beat_signals)
        
        v_wm_series = np.zeros(num_frames)
        v_os_series = np.zeros(num_frames)
        v_dcm_series = np.zeros(num_frames)
        
        for i, rd_map in enumerate(beat_signals):
            estimates = self.estimate_all_methods(rd_map, velocity_axis, range_axis)
            v_wm_series[i] = estimates['wm']
            v_os_series[i] = estimates['os']
            v_dcm_series[i] = estimates['dcm']
        
        results = {
            'wm_series': v_wm_series,
            'os_series': v_os_series,
            'dcm_series': v_dcm_series,
            'ground_truth_series': ground_truth_series
        }
        
        # Compute metrics if ground truth available
        if ground_truth_series is not None:
            results['metrics'] = self._compute_metrics(
                v_wm_series, v_os_series, v_dcm_series, ground_truth_series
            )
        
        return results
    
    def _compute_metrics(
        self,
        v_wm: np.ndarray,
        v_os: np.ndarray,
        v_dcm: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict:
        """
        Compute performance metrics.
        
        Args:
            v_wm: WM estimates
            v_os: OS estimates
            v_dcm: DCM estimates
            ground_truth: Ground truth velocities
            
        Returns:
            Dictionary with RMSE, bias, and standard deviation for each method
        """
        metrics = {}
        
        for name, estimates in [('wm', v_wm), ('os', v_os), ('dcm', v_dcm)]:
            errors = estimates - ground_truth
            
            metrics[f'{name}_rmse'] = np.sqrt(np.mean(errors ** 2))
            metrics[f'{name}_bias'] = np.mean(errors)
            metrics[f'{name}_std'] = np.std(errors)
        
        return metrics
