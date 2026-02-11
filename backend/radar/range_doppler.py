"""
Range-Doppler Processing Module

Implements 2D FFT-based Range-Doppler map generation:
- Fast-time FFT for range compression
- Slow-time FFT for Doppler processing
- Axis generation (range in meters, velocity in m/s)
- Magnitude map computation
"""

import numpy as np
from typing import Tuple, Dict


class RangeDopplerProcessor:
    """Range-Doppler Map Generator"""
    
    def __init__(self, radar_config: Dict):
        """
        Initialize Range-Doppler processor.
        
        Args:
            radar_config: Dictionary containing:
                - fc: Carrier frequency (Hz)
                - B: Bandwidth (Hz)
                - Tsw: Sweep time (s)
                - PRI: Pulse Repetition Interval (s)
                - N: ADC samples per chirp
                - M: Number of chirps
                - c: Speed of light (m/s)
        """
        self.fc = radar_config['fc']
        self.B = radar_config['B']
        self.Tsw = radar_config['Tsw']
        self.PRI = radar_config['PRI']
        self.N = radar_config['N']
        self.M = radar_config['M']
        self.c = radar_config.get('c', 3e8)
        
        # Derived parameters
        self.fs = self.N / self.Tsw
        self.chirp_rate = self.B / self.Tsw
        
    def compute_range_doppler(
        self,
        beat_signal: np.ndarray,
        window_range: str = 'hann',
        window_doppler: str = 'hann'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Range-Doppler map using 2D FFT.
        
        Args:
            beat_signal: Complex beat signal [M, N]
            window_range: Window function for range FFT ('hann', 'hamming', 'blackman', None)
            window_doppler: Window function for Doppler FFT
            
        Returns:
            rd_map: Range-Doppler magnitude map [M, N] (dB scale)
            range_axis: Range values (m)
            velocity_axis: Velocity values (m/s)
        """
        M, N = beat_signal.shape
        
        # Apply range window
        if window_range:
            range_window = self._get_window(window_range, N)
            beat_signal = beat_signal * range_window[np.newaxis, :]
        
        # Range FFT (fast-time)
        range_fft = np.fft.fft(beat_signal, axis=1)
        
        # Apply Doppler window
        if window_doppler:
            doppler_window = self._get_window(window_doppler, M)
            range_fft = range_fft * doppler_window[:, np.newaxis]
        
        # Doppler FFT (slow-time)
        rd_complex = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
        
        # Magnitude in dB
        rd_magnitude = np.abs(rd_complex)
        rd_map = 20 * np.log10(rd_magnitude + 1e-12)  # Add small value to avoid log(0)
        
        # Generate axes
        range_axis = self._generate_range_axis()
        velocity_axis = self._generate_velocity_axis()
        
        return rd_map, range_axis, velocity_axis
    
    def _generate_range_axis(self) -> np.ndarray:
        """
        Generate range axis in meters.
        
        Returns:
            Range values [N]
        """
        freq_axis = np.fft.fftfreq(self.N, 1 / self.fs)
        range_axis = (freq_axis * self.c) / (2 * self.chirp_rate)
        
        # Keep only positive ranges
        range_axis = range_axis[:self.N]
        
        return range_axis
    
    def _generate_velocity_axis(self) -> np.ndarray:
        """
        Generate velocity axis in m/s.
        
        Returns:
            Velocity values [M] (centered around zero)
        """
        doppler_freq = np.fft.fftshift(np.fft.fftfreq(self.M, self.PRI))
        velocity_axis = (doppler_freq * self.c) / (2 * self.fc)
        
        return velocity_axis
    
    def _get_window(self, window_type: str, length: int) -> np.ndarray:
        """
        Get window function.
        
        Args:
            window_type: Window type ('hann', 'hamming', 'blackman')
            length: Window length
            
        Returns:
            Window array
        """
        if window_type == 'hann':
            return np.hanning(length)
        elif window_type == 'hamming':
            return np.hamming(length)
        elif window_type == 'blackman':
            return np.blackman(length)
        else:
            return np.ones(length)
    
    def detect_peaks(
        self,
        rd_map: np.ndarray,
        threshold_db: float = -20,
        num_peaks: int = 10
    ) -> np.ndarray:
        """
        Detect peaks in Range-Doppler map.
        
        Args:
            rd_map: Range-Doppler map in dB
            threshold_db: Detection threshold relative to max
            num_peaks: Maximum number of peaks to return
            
        Returns:
            Peak positions [num_detected, 2] with columns [doppler_idx, range_idx]
        """
        # Normalize to max
        rd_normalized = rd_map - np.max(rd_map)
        
        # Find peaks above threshold
        mask = rd_normalized > threshold_db
        
        # Get indices
        doppler_idx, range_idx = np.where(mask)
        values = rd_normalized[doppler_idx, range_idx]
        
        # Sort by magnitude
        sorted_indices = np.argsort(values)[::-1]
        
        # Return top peaks
        num_detected = min(num_peaks, len(sorted_indices))
        peaks = np.column_stack([
            doppler_idx[sorted_indices[:num_detected]],
            range_idx[sorted_indices[:num_detected]]
        ])
        
        return peaks
    
    def extract_range_profile(self, rd_map: np.ndarray, doppler_idx: int) -> np.ndarray:
        """
        Extract range profile at specific Doppler bin.
        
        Args:
            rd_map: Range-Doppler map
            doppler_idx: Doppler bin index
            
        Returns:
            Range profile [N]
        """
        return rd_map[doppler_idx, :]
    
    def extract_doppler_profile(self, rd_map: np.ndarray, range_idx: int) -> np.ndarray:
        """
        Extract Doppler profile at specific range bin.
        
        Args:
            rd_map: Range-Doppler map
            range_idx: Range bin index
            
        Returns:
            Doppler profile [M]
        """
        return rd_map[:, range_idx]
