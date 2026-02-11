"""
SAR Quality Metrics Module

Computes SAR image quality metrics:
- Azimuth resolution (3 dB width measurement)
- Peak-to-Sidelobe Ratio (PSR)
- Azimuth position error
- Resolution vs synthetic aperture size curves
"""

import numpy as np
from typing import Dict, Tuple, List
from scipy.signal import find_peaks


class SARMetrics:
    """SAR Image Quality Metrics Calculator"""
    
    def compute_azimuth_resolution(
        self,
        sar_image: np.ndarray,
        azimuth_axis: np.ndarray,
        target_range_idx: int = None
    ) -> float:
        """
        Compute azimuth resolution (3 dB width).
        
        Args:
            sar_image: SAR image [M, N]
            azimuth_axis: Azimuth positions (m)
            target_range_idx: Range index of target (None = use brightest)
            
        Returns:
            Azimuth resolution (m)
        """
        # If no range index specified, find brightest point
        if target_range_idx is None:
            target_range_idx = np.argmax(np.max(sar_image, axis=0))
        
        # Extract azimuth profile at target range
        azimuth_profile = sar_image[:, target_range_idx]
        
        # Find peak
        peak_idx = np.argmax(azimuth_profile)
        peak_value = azimuth_profile[peak_idx]
        
        # Normalize
        azimuth_profile_norm = azimuth_profile / peak_value
        
        # Find 3 dB points (-3 dB = 0.707 in linear scale)
        threshold = np.sqrt(0.5)  # -3 dB
        
        # Find left 3 dB point
        left_idx = peak_idx
        while left_idx > 0 and azimuth_profile_norm[left_idx] > threshold:
            left_idx -= 1
        
        # Find right 3 dB point
        right_idx = peak_idx
        while right_idx < len(azimuth_profile) - 1 and azimuth_profile_norm[right_idx] > threshold:
            right_idx += 1
        
        # Interpolate for more accurate measurement
        if left_idx < peak_idx and right_idx > peak_idx:
            # Linear interpolation
            left_pos = self._interpolate_3db_point(
                azimuth_axis[left_idx:left_idx+2],
                azimuth_profile_norm[left_idx:left_idx+2],
                threshold
            )
            right_pos = self._interpolate_3db_point(
                azimuth_axis[right_idx-1:right_idx+1],
                azimuth_profile_norm[right_idx-1:right_idx+1],
                threshold
            )
            
            resolution = right_pos - left_pos
        else:
            # Fallback to bin-level measurement
            resolution = azimuth_axis[right_idx] - azimuth_axis[left_idx]
        
        return resolution
    
    def compute_psr(
        self,
        sar_image: np.ndarray,
        target_range_idx: int = None
    ) -> float:
        """
        Compute Peak-to-Sidelobe Ratio (PSR).
        
        Args:
            sar_image: SAR image [M, N]
            target_range_idx: Range index of target (None = use brightest)
            
        Returns:
            PSR in dB
        """
        # If no range index specified, find brightest point
        if target_range_idx is None:
            target_range_idx = np.argmax(np.max(sar_image, axis=0))
        
        # Extract azimuth profile at target range
        azimuth_profile = sar_image[:, target_range_idx]
        
        # Find main peak
        peak_idx = np.argmax(azimuth_profile)
        peak_value = azimuth_profile[peak_idx]
        
        # Find sidelobes (peaks excluding main lobe region)
        # Exclude region around main peak (±10% of array length)
        exclusion_width = int(0.1 * len(azimuth_profile))
        
        # Create mask excluding main lobe
        mask = np.ones(len(azimuth_profile), dtype=bool)
        mask[max(0, peak_idx - exclusion_width):min(len(azimuth_profile), peak_idx + exclusion_width + 1)] = False
        
        # Find peaks in sidelobe region
        sidelobe_region = azimuth_profile[mask]
        
        if len(sidelobe_region) > 0:
            max_sidelobe = np.max(sidelobe_region)
            psr_db = 20 * np.log10(peak_value / (max_sidelobe + 1e-12))
        else:
            psr_db = np.inf
        
        return psr_db
    
    def compute_position_error(
        self,
        sar_image: np.ndarray,
        azimuth_axis: np.ndarray,
        true_azimuth: float,
        target_range_idx: int = None
    ) -> float:
        """
        Compute azimuth position error.
        
        Args:
            sar_image: SAR image [M, N]
            azimuth_axis: Azimuth positions (m)
            true_azimuth: True azimuth position (m)
            target_range_idx: Range index of target
            
        Returns:
            Position error (m)
        """
        # If no range index specified, find brightest point
        if target_range_idx is None:
            target_range_idx = np.argmax(np.max(sar_image, axis=0))
        
        # Extract azimuth profile
        azimuth_profile = sar_image[:, target_range_idx]
        
        # Find peak
        peak_idx = np.argmax(azimuth_profile)
        estimated_azimuth = azimuth_axis[peak_idx]
        
        # Position error
        error = estimated_azimuth - true_azimuth
        
        return error
    
    def resolution_vs_aperture(
        self,
        radar_config: Dict,
        velocity: float,
        aperture_sizes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute theoretical resolution vs synthetic aperture size.
        
        Args:
            radar_config: Radar configuration
            velocity: Platform velocity (m/s)
            aperture_sizes: Array of aperture sizes to evaluate (m)
            
        Returns:
            aperture_sizes: Aperture sizes (m)
            resolutions: Theoretical azimuth resolutions (m)
        """
        lambda_ = radar_config.get('c', 3e8) / radar_config['fc']
        R = 100  # Reference range (m)
        
        # Theoretical azimuth resolution: ρ_az = L / 2
        # where L is synthetic aperture length
        resolutions = aperture_sizes / 2
        
        return aperture_sizes, resolutions
    
    def psr_vs_aperture(
        self,
        aperture_sizes: np.ndarray,
        window_type: str = 'hann'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute theoretical PSR vs synthetic aperture size.
        
        Args:
            aperture_sizes: Array of aperture sizes (m)
            window_type: Window function type
            
        Returns:
            aperture_sizes: Aperture sizes (m)
            psr_values: Theoretical PSR values (dB)
        """
        # PSR depends on window function
        # Typical values for common windows:
        # Rectangular: ~13 dB
        # Hann: ~32 dB
        # Hamming: ~43 dB
        # Blackman: ~58 dB
        
        psr_base = {
            'rectangular': 13,
            'hann': 32,
            'hamming': 43,
            'blackman': 58
        }.get(window_type, 32)
        
        # PSR generally improves slightly with aperture size
        # due to better Doppler resolution
        psr_values = psr_base + 10 * np.log10(aperture_sizes / aperture_sizes[0])
        
        return aperture_sizes, psr_values
    
    def compute_all_metrics(
        self,
        sar_image: np.ndarray,
        azimuth_axis: np.ndarray,
        true_azimuth: float = None,
        target_range_idx: int = None
    ) -> Dict:
        """
        Compute all SAR quality metrics.
        
        Args:
            sar_image: SAR image
            azimuth_axis: Azimuth positions (m)
            true_azimuth: True azimuth position (optional)
            target_range_idx: Range index of target
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Azimuth resolution
        metrics['azimuth_resolution'] = self.compute_azimuth_resolution(
            sar_image, azimuth_axis, target_range_idx
        )
        
        # PSR
        metrics['psr'] = self.compute_psr(sar_image, target_range_idx)
        
        # Position error (if ground truth available)
        if true_azimuth is not None:
            metrics['position_error'] = self.compute_position_error(
                sar_image, azimuth_axis, true_azimuth, target_range_idx
            )
        else:
            metrics['position_error'] = None
        
        return metrics
    
    def _interpolate_3db_point(
        self,
        x: np.ndarray,
        y: np.ndarray,
        threshold: float
    ) -> float:
        """
        Interpolate 3 dB point position.
        
        Args:
            x: X values (2 points)
            y: Y values (2 points)
            threshold: Threshold value
            
        Returns:
            Interpolated x position
        """
        if len(x) < 2 or len(y) < 2:
            return x[0]
        
        # Linear interpolation
        x_interp = x[0] + (threshold - y[0]) * (x[1] - x[0]) / (y[1] - y[0] + 1e-12)
        
        return x_interp
