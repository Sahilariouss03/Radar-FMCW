"""
SAR Image Formation Module

Implements Range-Doppler Algorithm (RDA) for SAR imaging:
- Range compression
- Range Cell Migration Correction (RCMC)
- Azimuth matched filtering
- Conventional RDA (constant velocity assumption)
- Interpolation-based RDA (nonuniform velocity)
"""

import numpy as np
from typing import Dict, Tuple
from scipy.interpolate import interp1d


class SARProcessor:
    """SAR Image Formation using Range-Doppler Algorithm"""
    
    def __init__(self, radar_config: Dict):
        """
        Initialize SAR processor.
        
        Args:
            radar_config: Dictionary containing:
                - fc: Carrier frequency (Hz)
                - B: Bandwidth (Hz)
                - lambda_: Wavelength (m)
                - PRI: Pulse Repetition Interval (s)
                - M: Number of chirps (azimuth samples)
                - N: ADC samples per chirp
                - c: Speed of light (m/s)
        """
        self.fc = radar_config['fc']
        self.B = radar_config['B']
        self.PRI = radar_config['PRI']
        self.M = radar_config['M']
        self.N = radar_config['N']
        self.c = radar_config.get('c', 3e8)
        self.lambda_ = self.c / self.fc
        
    def rda_conventional(
        self,
        beat_signal: np.ndarray,
        v_platform: float,
        synthetic_aperture_length: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Conventional Range-Doppler Algorithm (constant velocity).
        
        Args:
            beat_signal: Complex beat signal [M, N]
            v_platform: Platform velocity (m/s)
            synthetic_aperture_length: Synthetic aperture size (m)
            
        Returns:
            sar_image: SAR image magnitude [M, N]
            range_axis: Range axis (m)
            azimuth_axis: Azimuth axis (m)
        """
        M, N = beat_signal.shape
        
        # Step 1: Range compression (FFT along fast-time)
        range_compressed = np.fft.fft(beat_signal, axis=1)
        
        # Step 2: Range Cell Migration Correction (RCMC)
        rcmc_corrected = self._apply_rcmc(range_compressed, v_platform)
        
        # Step 3: Azimuth compression (matched filter)
        sar_complex = self._azimuth_compression(rcmc_corrected, v_platform)
        
        # Magnitude image
        sar_image = np.abs(sar_complex)
        
        # Generate axes
        range_axis = self._generate_range_axis()
        azimuth_axis = self._generate_azimuth_axis(v_platform)
        
        return sar_image, range_axis, azimuth_axis
    
    def rda_interpolated(
        self,
        beat_signal: np.ndarray,
        velocity_profile: np.ndarray,
        synthetic_aperture_length: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolation-based RDA for nonuniform velocity.
        
        Args:
            beat_signal: Complex beat signal [M, N]
            velocity_profile: Velocity at each chirp [M]
            synthetic_aperture_length: Synthetic aperture size (m)
            
        Returns:
            sar_image: SAR image magnitude [M, N]
            range_axis: Range axis (m)
            azimuth_axis: Azimuth axis (m)
        """
        M, N = beat_signal.shape
        
        # Compute actual azimuth positions
        azimuth_positions = np.cumsum(velocity_profile * self.PRI)
        azimuth_positions = np.insert(azimuth_positions[:-1], 0, 0)
        
        # Create uniform azimuth grid
        uniform_azimuth = np.linspace(
            azimuth_positions[0],
            azimuth_positions[-1],
            M
        )
        
        # Interpolate beat signal to uniform grid
        beat_uniform = np.zeros((M, N), dtype=complex)
        
        for n in range(N):
            # Interpolate real and imaginary parts separately
            interp_real = interp1d(
                azimuth_positions,
                beat_signal[:, n].real,
                kind='cubic',
                fill_value='extrapolate'
            )
            interp_imag = interp1d(
                azimuth_positions,
                beat_signal[:, n].imag,
                kind='cubic',
                fill_value='extrapolate'
            )
            
            beat_uniform[:, n] = interp_real(uniform_azimuth) + 1j * interp_imag(uniform_azimuth)
        
        # Apply conventional RDA on uniform grid
        v_avg = np.mean(velocity_profile)
        sar_image, range_axis, _ = self.rda_conventional(
            beat_uniform, v_avg, synthetic_aperture_length
        )
        
        azimuth_axis = uniform_azimuth
        
        return sar_image, range_axis, azimuth_axis
    
    def _apply_rcmc(
        self,
        range_compressed: np.ndarray,
        v_platform: float
    ) -> np.ndarray:
        """
        Apply Range Cell Migration Correction.
        
        Args:
            range_compressed: Range-compressed signal [M, N]
            v_platform: Platform velocity (m/s)
            
        Returns:
            RCMC-corrected signal [M, N]
        """
        M, N = range_compressed.shape
        
        # Azimuth frequency axis
        f_azimuth = np.fft.fftfreq(M, self.PRI)
        
        # Range frequency axis
        f_range = np.fft.fftfreq(N, 1 / (N / self.B))
        
        # Create 2D grids
        F_azimuth, F_range = np.meshgrid(f_azimuth, f_range, indexing='ij')
        
        # Range cell migration (simplified model)
        # Migration = lambda^2 * R * f_azimuth^2 / (8 * v^2)
        # For simplicity, use approximate correction
        
        R_ref = self.c / (2 * self.B)  # Reference range
        migration = (self.lambda_**2 * R_ref * F_azimuth**2) / (8 * v_platform**2)
        
        # Phase correction
        phase_correction = np.exp(-1j * 2 * np.pi * F_range * migration)
        
        # Apply correction in 2D frequency domain
        spectrum_2d = np.fft.fft2(range_compressed)
        corrected_spectrum = spectrum_2d * np.fft.fftshift(phase_correction)
        rcmc_corrected = np.fft.ifft2(corrected_spectrum)
        
        return rcmc_corrected
    
    def _azimuth_compression(
        self,
        signal: np.ndarray,
        v_platform: float
    ) -> np.ndarray:
        """
        Apply azimuth matched filtering.
        
        Args:
            signal: Input signal [M, N]
            v_platform: Platform velocity (m/s)
            
        Returns:
            Azimuth-compressed signal [M, N]
        """
        M, N = signal.shape
        
        # Azimuth matched filter (FFT along slow-time)
        azimuth_spectrum = np.fft.fft(signal, axis=0)
        
        # Apply Doppler-dependent phase correction
        f_azimuth = np.fft.fftfreq(M, self.PRI)
        
        # Matched filter (simplified)
        # In practice, this would be range-dependent
        R_ref = self.c / (2 * self.B)
        
        for n in range(N):
            phase_correction = np.exp(
                1j * np.pi * self.lambda_ * R_ref * f_azimuth**2 / v_platform**2
            )
            azimuth_spectrum[:, n] *= phase_correction
        
        # Inverse FFT
        compressed = np.fft.ifft(azimuth_spectrum, axis=0)
        
        return compressed
    
    def _generate_range_axis(self) -> np.ndarray:
        """Generate range axis in meters."""
        freq_axis = np.fft.fftfreq(self.N, 1 / (self.N / self.B))
        range_axis = (freq_axis * self.c) / (2 * self.B)
        return range_axis[:self.N]
    
    def _generate_azimuth_axis(self, v_platform: float) -> np.ndarray:
        """Generate azimuth axis in meters."""
        azimuth_axis = np.arange(self.M) * v_platform * self.PRI
        return azimuth_axis
    
    def compute_resolution(
        self,
        sar_image: np.ndarray,
        azimuth_axis: np.ndarray,
        target_azimuth: float
    ) -> float:
        """
        Compute azimuth resolution (3 dB width).
        
        Args:
            sar_image: SAR image
            azimuth_axis: Azimuth positions (m)
            target_azimuth: Azimuth position of target (m)
            
        Returns:
            Azimuth resolution (m)
        """
        # Find target position
        target_idx = np.argmin(np.abs(azimuth_axis - target_azimuth))
        
        # Extract azimuth profile (sum over range)
        azimuth_profile = np.sum(sar_image, axis=1)
        
        # Normalize
        azimuth_profile = azimuth_profile / np.max(azimuth_profile)
        
        # Find 3 dB points (-3 dB = 0.707 in linear scale)
        threshold = 0.707
        
        # Find left and right 3 dB points
        left_idx = target_idx
        while left_idx > 0 and azimuth_profile[left_idx] > threshold:
            left_idx -= 1
        
        right_idx = target_idx
        while right_idx < len(azimuth_profile) - 1 and azimuth_profile[right_idx] > threshold:
            right_idx += 1
        
        # Resolution is width between 3 dB points
        resolution = azimuth_axis[right_idx] - azimuth_axis[left_idx]
        
        return resolution
