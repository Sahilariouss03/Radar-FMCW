"""
FMCW Radar Signal Simulation Module

Implements fast-chirp FMCW radar beat signal generation with:
- Configurable radar parameters (carrier frequency, bandwidth, sweep time, PRI)
- Vehicle motion modeling (constant velocity, acceleration, velocity noise)
- Ground scatterer and point target simulation
- Self-interference injection
- Noise modeling
"""

import numpy as np
from typing import Dict, Tuple, Optional


class FMCWRadar:
    """FMCW Radar Signal Simulator"""
    
    def __init__(self, config: Dict):
        """
        Initialize FMCW radar with configuration parameters.
        
        Args:
            config: Dictionary containing:
                - fc: Carrier frequency (Hz)
                - B: Bandwidth (Hz)
                - Tsw: Sweep time (s)
                - PRI: Pulse Repetition Interval (s)
                - M: Number of chirps
                - N: ADC samples per chirp
                - h0: Radar height (m)
                - c: Speed of light (m/s, default 3e8)
        """
        self.fc = config['fc']
        self.B = config['B']
        self.Tsw = config['Tsw']
        self.PRI = config['PRI']
        self.M = config['M']
        self.N = config['N']
        self.h0 = config['h0']
        self.c = config.get('c', 3e8)
        
        # Derived parameters
        self.chirp_rate = self.B / self.Tsw  # Hz/s
        self.fs = self.N / self.Tsw  # Sampling frequency
        self.dt = 1 / self.fs  # Fast-time sampling interval
        self.range_resolution = self.c / (2 * self.B)
        self.max_range = self.c * self.fs / (4 * self.chirp_rate)
        
        # Time axes
        self.fast_time = np.arange(self.N) * self.dt
        self.slow_time = np.arange(self.M) * self.PRI
        
    def generate_beat_signal(
        self,
        velocity_profile: np.ndarray,
        scatterers: np.ndarray,
        point_targets: Optional[np.ndarray] = None,
        noise_power: float = 1e-6,
        si_amplitude: float = 0.1
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate FMCW beat signal with vehicle motion, scatterers, and interference.
        
        Args:
            velocity_profile: Vehicle velocity at each chirp [M] (m/s)
            scatterers: Ground scatterers [Ns, 3] with columns [x, y, RCS]
            point_targets: Optional point targets [Nt, 3] with columns [x, y, RCS]
            noise_power: Noise power level
            si_amplitude: Self-interference amplitude
            
        Returns:
            beat_signal: Complex beat signal matrix [M, N]
            metadata: Dictionary with simulation info
        """
        M, N = self.M, self.N
        beat_signal = np.zeros((M, N), dtype=complex)
        
        # Vehicle position integration
        vehicle_x = np.cumsum(velocity_profile * self.PRI)
        vehicle_x = np.insert(vehicle_x[:-1], 0, 0)  # Start at x=0
        
        # Process ground scatterers
        for m in range(M):
            vx = vehicle_x[m]
            
            for scatterer in scatterers:
                sx, sy, rcs = scatterer
                
                # Range from radar to scatterer
                R = np.sqrt((sx - vx)**2 + sy**2 + self.h0**2)
                
                # Time delay
                tau = 2 * R / self.c
                
                # Beat frequency
                fb = self.chirp_rate * tau
                
                # Doppler shift (radial velocity component)
                v_radial = -velocity_profile[m] * (sx - vx) / np.sqrt((sx - vx)**2 + sy**2)
                fd = 2 * v_radial * self.fc / self.c
                
                # Signal contribution
                phase = 2 * np.pi * (fb * self.fast_time + fd * self.slow_time[m])
                amplitude = np.sqrt(rcs) / (R**2)
                beat_signal[m, :] += amplitude * np.exp(1j * phase)
        
        # Add point targets if provided
        if point_targets is not None:
            for target in point_targets:
                tx, ty, rcs = target
                
                for m in range(M):
                    vx = vehicle_x[m]
                    R = np.sqrt((tx - vx)**2 + ty**2 + self.h0**2)
                    tau = 2 * R / self.c
                    fb = self.chirp_rate * tau
                    v_radial = -velocity_profile[m] * (tx - vx) / np.sqrt((tx - vx)**2 + ty**2)
                    fd = 2 * v_radial * self.fc / self.c
                    
                    phase = 2 * np.pi * (fb * self.fast_time + fd * self.slow_time[m])
                    amplitude = np.sqrt(rcs) / (R**2)
                    beat_signal[m, :] += amplitude * np.exp(1j * phase)
        
        # Add self-interference (zero-Doppler, near-zero range)
        si_signal = self._generate_self_interference(si_amplitude)
        beat_signal += si_signal
        
        # Add noise
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(M, N) + 1j * np.random.randn(M, N)
        )
        beat_signal += noise
        
        metadata = {
            'vehicle_x': vehicle_x,
            'velocity_profile': velocity_profile,
            'range_resolution': self.range_resolution,
            'max_range': self.max_range,
            'num_scatterers': len(scatterers),
            'num_targets': len(point_targets) if point_targets is not None else 0
        }
        
        return beat_signal, metadata
    
    def _generate_self_interference(self, amplitude: float) -> np.ndarray:
        """
        Generate self-interference signal (zero-Doppler, near-zero range).
        
        Args:
            amplitude: Interference amplitude
            
        Returns:
            Interference signal [M, N]
        """
        # Zero-Doppler interference with sinc-like sidelobes
        M, N = self.M, self.N
        si = np.zeros((M, N), dtype=complex)
        
        # Main lobe at zero range
        range_idx = int(0.05 * N)  # Near-zero range
        window = np.hamming(int(0.1 * N))
        
        for m in range(M):
            # Zero Doppler, concentrated in range
            si[m, range_idx:range_idx + len(window)] = amplitude * window * np.exp(
                1j * 2 * np.pi * np.random.rand()
            )
        
        return si
    
    def generate_scatterers(
        self,
        num_scatterers: int,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        rcs_range: Tuple[float, float] = (0.1, 1.0)
    ) -> np.ndarray:
        """
        Generate random ground scatterers.
        
        Args:
            num_scatterers: Number of scatterers
            x_range: (min, max) x position (m)
            y_range: (min, max) y position (m)
            rcs_range: (min, max) radar cross section
            
        Returns:
            Scatterers array [num_scatterers, 3] with columns [x, y, RCS]
        """
        x = np.random.uniform(x_range[0], x_range[1], num_scatterers)
        y = np.random.uniform(y_range[0], y_range[1], num_scatterers)
        rcs = np.random.uniform(rcs_range[0], rcs_range[1], num_scatterers)
        
        return np.column_stack([x, y, rcs])
    
    def generate_point_targets(
        self,
        positions: np.ndarray,
        rcs: float = 10.0
    ) -> np.ndarray:
        """
        Generate point targets at specific positions.
        
        Args:
            positions: Target positions [num_targets, 2] with columns [x, y]
            rcs: Radar cross section for all targets
            
        Returns:
            Point targets array [num_targets, 3] with columns [x, y, RCS]
        """
        num_targets = len(positions)
        rcs_array = np.full(num_targets, rcs)
        
        return np.column_stack([positions, rcs_array])
    
    def generate_velocity_profile(
        self,
        v0: float,
        acceleration: float = 0.0,
        noise_std: float = 0.0
    ) -> np.ndarray:
        """
        Generate vehicle velocity profile.
        
        Args:
            v0: Initial velocity (m/s)
            acceleration: Constant acceleration (m/s²)
            noise_std: Velocity noise standard deviation (m/s)
            
        Returns:
            Velocity at each chirp [M]
        """
        t = self.slow_time
        velocity = v0 + acceleration * t
        
        if noise_std > 0:
            velocity += np.random.randn(self.M) * noise_std
        
        return velocity
