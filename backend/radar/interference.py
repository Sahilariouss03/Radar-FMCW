"""
Self-Interference Cancellation Module

Implements iterative interference cancellation:
- Zero-Doppler peak detection
- Sinc pattern modeling
- Iterative subtraction algorithm
- Performance metrics (dB reduction, iteration count)
"""

import numpy as np
from typing import Tuple, Dict


class InterferenceCanceller:
    """Self-Interference Cancellation Processor"""
    
    def __init__(self, radar_config: Dict):
        """
        Initialize interference canceller.
        
        Args:
            radar_config: Dictionary containing radar parameters
        """
        self.M = radar_config['M']
        self.N = radar_config['N']
        
    def iterative_cancellation(
        self,
        rd_map: np.ndarray,
        max_iterations: int = 5,
        threshold_db: float = -30,
        zero_doppler_width: int = 3
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform iterative self-interference cancellation.
        
        Args:
            rd_map: Range-Doppler map (linear scale, complex)
            max_iterations: Maximum number of iterations
            threshold_db: Convergence threshold (dB)
            zero_doppler_width: Width of zero-Doppler region (bins)
            
        Returns:
            rd_cleaned: Cleaned Range-Doppler map
            metrics: Dictionary with cancellation metrics
        """
        rd_working = rd_map.copy()
        
        # Track interference power
        interference_power = []
        
        for iteration in range(max_iterations):
            # Detect interference peak in zero-Doppler region
            peak_doppler, peak_range, peak_value = self._detect_zero_doppler_peak(
                rd_working, zero_doppler_width
            )
            
            # Model sinc pattern
            interference_model = self._model_sinc_pattern(
                rd_working, peak_doppler, peak_range, peak_value
            )
            
            # Subtract interference
            rd_working = rd_working - interference_model
            
            # Measure residual interference power
            residual_power = self._measure_interference_power(
                rd_working, zero_doppler_width
            )
            interference_power.append(residual_power)
            
            # Check convergence
            if iteration > 0:
                reduction_db = 10 * np.log10(
                    interference_power[iteration - 1] / (residual_power + 1e-12)
                )
                if reduction_db < threshold_db:
                    break
        
        # Compute metrics
        initial_power = self._measure_interference_power(rd_map, zero_doppler_width)
        final_power = interference_power[-1]
        total_reduction_db = 10 * np.log10(initial_power / (final_power + 1e-12))
        
        metrics = {
            'iterations': iteration + 1,
            'initial_power': initial_power,
            'final_power': final_power,
            'reduction_db': total_reduction_db,
            'power_history': interference_power
        }
        
        return rd_working, metrics
    
    def _detect_zero_doppler_peak(
        self,
        rd_map: np.ndarray,
        zero_doppler_width: int
    ) -> Tuple[int, int, complex]:
        """
        Detect peak in zero-Doppler region.
        
        Args:
            rd_map: Range-Doppler map (complex)
            zero_doppler_width: Width of zero-Doppler region
            
        Returns:
            peak_doppler: Doppler index of peak
            peak_range: Range index of peak
            peak_value: Complex value at peak
        """
        M, N = rd_map.shape
        center_doppler = M // 2
        
        # Extract zero-Doppler region
        doppler_start = max(0, center_doppler - zero_doppler_width)
        doppler_end = min(M, center_doppler + zero_doppler_width + 1)
        
        zero_doppler_region = rd_map[doppler_start:doppler_end, :]
        
        # Find peak
        magnitude = np.abs(zero_doppler_region)
        peak_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        
        peak_doppler = doppler_start + peak_idx[0]
        peak_range = peak_idx[1]
        peak_value = rd_map[peak_doppler, peak_range]
        
        return peak_doppler, peak_range, peak_value
    
    def _model_sinc_pattern(
        self,
        rd_map: np.ndarray,
        peak_doppler: int,
        peak_range: int,
        peak_value: complex
    ) -> np.ndarray:
        """
        Model sinc-like interference pattern.
        
        Args:
            rd_map: Range-Doppler map
            peak_doppler: Doppler index of interference
            peak_range: Range index of interference
            peak_value: Peak value
            
        Returns:
            Interference model [M, N]
        """
        M, N = rd_map.shape
        
        # Create sinc pattern in range dimension
        range_indices = np.arange(N)
        range_sinc = np.sinc((range_indices - peak_range) / 5.0)
        
        # Create narrow pattern in Doppler dimension
        doppler_indices = np.arange(M)
        doppler_pattern = np.exp(-((doppler_indices - peak_doppler) ** 2) / (2 * 2**2))
        
        # 2D pattern
        pattern_2d = doppler_pattern[:, np.newaxis] * range_sinc[np.newaxis, :]
        
        # Scale by peak value
        interference_model = peak_value * pattern_2d
        
        return interference_model
    
    def _measure_interference_power(
        self,
        rd_map: np.ndarray,
        zero_doppler_width: int
    ) -> float:
        """
        Measure interference power in zero-Doppler region.
        
        Args:
            rd_map: Range-Doppler map
            zero_doppler_width: Width of zero-Doppler region
            
        Returns:
            Total power in zero-Doppler region
        """
        M = rd_map.shape[0]
        center_doppler = M // 2
        
        doppler_start = max(0, center_doppler - zero_doppler_width)
        doppler_end = min(M, center_doppler + zero_doppler_width + 1)
        
        zero_doppler_region = rd_map[doppler_start:doppler_end, :]
        power = np.sum(np.abs(zero_doppler_region) ** 2)
        
        return power
    
    def apply_notch_filter(
        self,
        rd_map: np.ndarray,
        notch_width: int = 2
    ) -> np.ndarray:
        """
        Apply simple notch filter at zero Doppler (alternative method).
        
        Args:
            rd_map: Range-Doppler map
            notch_width: Width of notch (bins on each side of zero)
            
        Returns:
            Filtered Range-Doppler map
        """
        M = rd_map.shape[0]
        center_doppler = M // 2
        
        rd_filtered = rd_map.copy()
        
        # Zero out zero-Doppler bins
        doppler_start = max(0, center_doppler - notch_width)
        doppler_end = min(M, center_doppler + notch_width + 1)
        rd_filtered[doppler_start:doppler_end, :] = 0
        
        return rd_filtered
