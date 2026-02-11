"""
FastAPI Application for Automotive FMCW SAR Simulation

RESTful API with endpoints:
- POST /simulate: Run complete simulation pipeline
- GET /default-parameters: Get default radar configuration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np

from radar.fmcw import FMCWRadar
from radar.range_doppler import RangeDopplerProcessor
from radar.interference import InterferenceCanceller
from radar.ego_motion import EgoMotionEstimator
from radar.sar import SARProcessor
from radar.metrics import SARMetrics


app = FastAPI(
    title="Automotive FMCW SAR Simulation API",
    description="Backend API for radar signal processing and SAR imaging",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class SimulationParameters(BaseModel):
    """Simulation configuration parameters"""
    # Radar parameters
    fc: float = Field(77e9, description="Carrier frequency (Hz)")
    B: float = Field(500e6, description="Bandwidth (Hz)")
    Tsw: float = Field(40e-6, description="Sweep time (s)")
    PRI: float = Field(50e-6, description="Pulse Repetition Interval (s)")
    M: int = Field(256, description="Number of chirps")
    N: int = Field(512, description="ADC samples per chirp")
    
    # Vehicle parameters
    v0: float = Field(20.0, description="Initial velocity (m/s)")
    acceleration: float = Field(0.0, description="Acceleration (m/s²)")
    velocity_noise_std: float = Field(0.0, description="Velocity noise std dev (m/s)")
    h0: float = Field(1.5, description="Radar height (m)")
    
    # Scene parameters
    synthetic_aperture_length: float = Field(10.0, description="Synthetic aperture size (m)")
    num_scatterers: int = Field(100, description="Number of ground scatterers")
    num_point_targets: int = Field(3, description="Number of point targets")
    
    # Noise and interference
    noise_power: float = Field(1e-6, description="Noise power level")
    si_amplitude: float = Field(0.1, description="Self-interference amplitude")
    
    # Processing parameters
    max_si_iterations: int = Field(5, description="Max interference cancellation iterations")


class SimulationResult(BaseModel):
    """Simulation output data"""
    # Range-Doppler maps
    rd_map_before_si: List[List[float]]
    rd_map_after_si: List[List[float]]
    range_axis: List[float]
    velocity_axis: List[float]
    
    # Interference cancellation metrics
    si_reduction_db: float
    si_iterations: int
    
    # Ego-motion estimates
    velocity_wm: float
    velocity_os: float
    velocity_dcm: float
    velocity_ground_truth: float
    ego_motion_metrics: dict
    
    # SAR images
    sar_image_conventional: List[List[float]]
    sar_image_interpolated: List[List[float]]
    sar_range_axis: List[float]
    sar_azimuth_axis: List[float]
    
    # SAR quality metrics
    azimuth_resolution: float
    psr: float
    position_error: Optional[float]
    
    # Performance curves
    resolution_vs_aperture: dict
    psr_vs_aperture: dict


@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "message": "Automotive FMCW SAR Simulation API",
        "version": "1.0.0"
    }


@app.get("/default-parameters")
async def get_default_parameters():
    """Get default automotive radar configuration"""
    return {
        "fc": 77e9,  # 77 GHz automotive radar
        "B": 500e6,  # 500 MHz bandwidth
        "Tsw": 40e-6,  # 40 μs sweep time
        "PRI": 50e-6,  # 50 μs PRI
        "M": 256,  # 256 chirps
        "N": 512,  # 512 samples per chirp
        "v0": 20.0,  # 20 m/s (72 km/h)
        "acceleration": 0.0,
        "velocity_noise_std": 0.0,
        "h0": 1.5,  # 1.5 m radar height
        "synthetic_aperture_length": 10.0,  # 10 m aperture
        "num_scatterers": 100,
        "num_point_targets": 3,
        "noise_power": 1e-6,
        "si_amplitude": 0.1,
        "max_si_iterations": 5
    }


@app.post("/simulate", response_model=SimulationResult)
async def run_simulation(params: SimulationParameters):
    """
    Run complete FMCW SAR simulation pipeline.
    
    Returns all processing results including Range-Doppler maps,
    ego-motion estimates, SAR images, and quality metrics.
    """
    try:
        # Initialize radar configuration
        radar_config = {
            'fc': params.fc,
            'B': params.B,
            'Tsw': params.Tsw,
            'PRI': params.PRI,
            'M': params.M,
            'N': params.N,
            'h0': params.h0,
            'c': 3e8
        }
        
        # Step 1: Generate FMCW beat signal
        radar = FMCWRadar(radar_config)
        
        velocity_profile = radar.generate_velocity_profile(
            params.v0,
            params.acceleration,
            params.velocity_noise_std
        )
        
        scatterers = radar.generate_scatterers(
            params.num_scatterers,
            x_range=(0, params.synthetic_aperture_length * 2),
            y_range=(10, 100)
        )
        
        # Generate point targets at specific positions
        target_positions = np.array([
            [params.synthetic_aperture_length * 0.3, 30],
            [params.synthetic_aperture_length * 0.5, 50],
            [params.synthetic_aperture_length * 0.7, 70]
        ])[:params.num_point_targets]
        
        point_targets = radar.generate_point_targets(target_positions, rcs=10.0)
        
        beat_signal, metadata = radar.generate_beat_signal(
            velocity_profile,
            scatterers,
            point_targets,
            params.noise_power,
            params.si_amplitude
        )
        
        # Step 2: Range-Doppler processing
        rd_processor = RangeDopplerProcessor(radar_config)
        rd_map_db, range_axis, velocity_axis = rd_processor.compute_range_doppler(beat_signal)
        
        # Convert to linear scale for interference cancellation
        rd_map_linear = 10 ** (rd_map_db / 20)
        rd_map_complex = rd_map_linear * np.exp(1j * np.angle(
            np.fft.fftshift(np.fft.fft2(beat_signal), axes=0)
        ))
        
        # Step 3: Self-interference cancellation
        si_canceller = InterferenceCanceller(radar_config)
        rd_cleaned, si_metrics = si_canceller.iterative_cancellation(
            rd_map_complex,
            max_iterations=params.max_si_iterations
        )
        
        # Convert cleaned map to dB
        rd_cleaned_db = 20 * np.log10(np.abs(rd_cleaned) + 1e-12)
        
        # Step 4: Ego-motion estimation
        ego_estimator = EgoMotionEstimator(radar_config)
        ego_results = ego_estimator.estimate_all_methods(
            np.abs(rd_cleaned),
            velocity_axis,
            range_axis,
            ground_truth=params.v0
        )
        
        # Step 5: SAR image formation
        sar_processor = SARProcessor(radar_config)
        
        # Conventional RDA
        sar_conventional, sar_range_axis, sar_azimuth_axis = sar_processor.rda_conventional(
            beat_signal,
            params.v0,
            params.synthetic_aperture_length
        )
        
        # Interpolated RDA
        sar_interpolated, _, _ = sar_processor.rda_interpolated(
            beat_signal,
            velocity_profile,
            params.synthetic_aperture_length
        )
        
        # Step 6: SAR quality metrics
        sar_metrics_calc = SARMetrics()
        sar_quality = sar_metrics_calc.compute_all_metrics(
            sar_conventional,
            sar_azimuth_axis,
            true_azimuth=target_positions[0, 0] if len(target_positions) > 0 else None
        )
        
        # Resolution vs aperture curve
        aperture_sizes = np.linspace(2, 20, 10)
        apertures, resolutions = sar_metrics_calc.resolution_vs_aperture(
            radar_config,
            params.v0,
            aperture_sizes
        )
        
        # PSR vs aperture curve
        _, psr_values = sar_metrics_calc.psr_vs_aperture(aperture_sizes)
        
        # Prepare response
        result = SimulationResult(
            # Range-Doppler maps
            rd_map_before_si=rd_map_db.tolist(),
            rd_map_after_si=rd_cleaned_db.tolist(),
            range_axis=range_axis.tolist(),
            velocity_axis=velocity_axis.tolist(),
            
            # Interference cancellation
            si_reduction_db=float(si_metrics['reduction_db']),
            si_iterations=int(si_metrics['iterations']),
            
            # Ego-motion
            velocity_wm=float(ego_results['wm']),
            velocity_os=float(ego_results['os']),
            velocity_dcm=float(ego_results['dcm']),
            velocity_ground_truth=float(params.v0),
            ego_motion_metrics=ego_results.get('metrics', {}),
            
            # SAR images
            sar_image_conventional=sar_conventional.tolist(),
            sar_image_interpolated=sar_interpolated.tolist(),
            sar_range_axis=sar_range_axis.tolist(),
            sar_azimuth_axis=sar_azimuth_axis.tolist(),
            
            # SAR quality
            azimuth_resolution=float(sar_quality['azimuth_resolution']),
            psr=float(sar_quality['psr']),
            position_error=float(sar_quality['position_error']) if sar_quality['position_error'] is not None else None,
            
            # Performance curves
            resolution_vs_aperture={
                'aperture_sizes': apertures.tolist(),
                'resolutions': resolutions.tolist()
            },
            psr_vs_aperture={
                'aperture_sizes': aperture_sizes.tolist(),
                'psr_values': psr_values.tolist()
            }
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
