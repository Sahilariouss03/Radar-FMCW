"""
FastAPI — Physics-Correct Simulation Pipeline

Validation (Prompts 6-7):
    1. max(|v_bins|) == λ/(4·T_PRI)       Doppler axis check
    2. |V_est − V_true| > 5 m/s           ego-motion flag
    3. PSR < 5 dB                          focusing failure flag
    4. Resolution increases with aperture   inversion bug flag

Resolution (Prompt 8): return MEASURED 3dB width, not theoretical.
Azimuth axis (Prompt 5):  Δx = V_avg × T_PRI, from integrated motion.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import time

from .fmcw import FMCWRadar
from .range_doppler import RangeDopplerProcessor
from .interference import InterferenceCanceller
from .ego_motion import EgoMotionEstimator
from .sar import SARProcessor
from .metrics import SARMetrics

# ── App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Automotive FMCW SAR Simulator",
    description="Physics-correct radar signal processing v2.1",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────────────────
class SimulationParameters(BaseModel):
    fc: float = Field(77e9, description="Carrier frequency (Hz)")
    B: float = Field(500e6, description="Bandwidth (Hz)")
    Tsw: float = Field(40e-6, description="Sweep time (s)")
    PRI: float = Field(50e-6, description="Pulse repetition interval (s)")
    M: int = Field(256, description="Number of chirps")
    N: int = Field(512, description="ADC samples per chirp")
    v0: float = Field(20.0, description="Vehicle velocity (m/s)")
    acceleration: float = Field(0.0, description="Acceleration (m/s²)")
    velocity_noise_std: float = Field(0.0, description="Velocity noise std (m/s)")
    h0: float = Field(1.5, description="Radar height (m)")
    synthetic_aperture_length: float = Field(10.0, description="Aperture length (m)")
    num_scatterers: int = Field(100, description="Number of ground scatterers")
    num_point_targets: int = Field(3, description="Number of point targets")
    noise_power: float = Field(1e-6, description="Noise power")
    si_amplitude: float = Field(0.1, description="Self-interference amplitude")
    max_si_iterations: int = Field(10, description="Max SI cancellation iterations")
    debug: bool = Field(False, description="Return debug arrays")


class SimulationResult(BaseModel):
    # Range-Doppler
    rd_map_before_si: List[List[float]]
    rd_map_after_si: List[List[float]]
    range_axis: List[float]
    velocity_axis: List[float]
    si_reduction_db: float
    si_iterations: int
    # Ego-motion
    velocity_ground_truth: float
    velocity_wm: float
    velocity_os: float
    velocity_dcm: float
    ego_motion_metrics: dict
    # SAR
    sar_image_conventional: List[List[float]]
    sar_image_interpolated: List[List[float]]
    sar_range_axis: List[float]
    sar_azimuth_axis: List[float]
    # Metrics — MEASURED not theoretical
    azimuth_resolution_measured: float
    psr: float
    position_error: Optional[float]
    resolution_vs_aperture: dict
    psr_vs_aperture: dict
    # Validation
    validation_warnings: List[str]
    computation_time: float
    debug_info: Optional[dict] = None


# ── Endpoints ──────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "Automotive FMCW SAR Simulator",
        "version": "2.1.0 — physics-correct",
        "endpoints": ["/simulate", "/default-parameters", "/docs"],
    }


@app.get("/default-parameters")
async def get_default_parameters():
    return SimulationParameters().model_dump()


@app.post("/simulate", response_model=SimulationResult)
async def run_simulation(params: SimulationParameters):
    t0 = time.time()
    warnings: List[str] = []

    try:
        config = {
            'fc': params.fc, 'B': params.B, 'Tsw': params.Tsw,
            'PRI': params.PRI, 'M': params.M, 'N': params.N,
            'h0': params.h0, 'c': 3e8,
        }
        lambda_ = 3e8 / params.fc

        # ── 1. FMCW signal ─────────────────────────────────────
        radar = FMCWRadar(config)
        vel_prof = radar.generate_velocity_profile(
            params.v0, params.acceleration, params.velocity_noise_std
        )
        gt_v = float(np.mean(vel_prof))

        # Calculate ACTUAL physical aperture traversed
        # L_actual = duration * speed
        actual_aperture_length = params.M * params.PRI * gt_v

        # Scatterers: Full road scene to ensure:
        # 1. Forward scatterers (x >> 0) -> High Doppler for Ego-Motion
        # 2. Side scatterers (x ~ 0) -> SAR imaging background
        scatterers = radar.generate_scatterers(
            params.num_scatterers,
            x_range=(-50, 50),
            y_range=(5, 50),
        )

        # Point targets: Place at x=0 (Broadside)
        # This is where SAR resolution is best (sin(theta)=1)
        # and they are inside the physical aperture x in [-L/2, L/2]
        target_true_azimuth = 0.0
        point_targets = None
        if params.num_point_targets > 0:
            tx = np.zeros(params.num_point_targets)  # All at x=0 azimuth
            # Spread in range (y) instead of azimuth
            ty = np.linspace(15, 25, params.num_point_targets)
            positions = np.column_stack([tx, ty])
            point_targets = radar.generate_point_targets(positions, rcs=1000.0)
            target_true_azimuth = 0.0

        beat, meta = radar.generate_beat_signal(
            vel_prof, scatterers, point_targets,
            noise_power=params.noise_power,
            si_amplitude=params.si_amplitude,
        )

        # ── 2. Range-Doppler ───────────────────────────────────
        rd_proc = RangeDopplerProcessor(config)
        rd_complex, range_axis, vel_axis = rd_proc.compute_range_doppler_complex(beat)
        rd_before_db = 20 * np.log10(np.abs(rd_complex) + 1e-12)

        # ── 3. Self-interference cancellation ──────────────────
        canceller = InterferenceCanceller(config)
        rd_clean, si_met = canceller.iterative_cancellation(
            rd_complex, max_iterations=params.max_si_iterations
        )
        rd_after_db = 20 * np.log10(np.abs(rd_clean) + 1e-12)

        # ── 4. Ego-motion ──────────────────────────────────────
        ego = EgoMotionEstimator(config)
        ego_res = ego.estimate_all_methods(
            rd_clean, vel_axis, range_axis, ground_truth=gt_v
        )

        # ── 5. SAR image ──────────────────────────────────────
        sar_proc = SARProcessor(config)

        v_platform = ego_res['v_wm']
        if abs(v_platform) < 0.5:
            v_platform = gt_v

        sar_conv, sar_r, sar_az = sar_proc.rda_conventional(
            beat, v_platform, params.synthetic_aperture_length
        )
        sar_interp, _, _ = sar_proc.rda_interpolated(
            beat, vel_prof, params.synthetic_aperture_length
        )

        # ── 6. MEASURED metrics (Prompt 8) ─────────────────────
        metrics = SARMetrics()

        # Measured resolution from 3dB width (NOT theoretical)
        az_res_measured = metrics.compute_azimuth_resolution(sar_conv, sar_az)
        psr = metrics.compute_psr(sar_conv)

        pos_err = None
        if point_targets is not None:
            pos_err = metrics.compute_position_error(
                sar_conv, sar_az, true_azimuth=target_true_azimuth
            )

        # Resolution & PSR vs aperture (theoretical curves)
        ap_sizes = np.linspace(actual_aperture_length * 0.5, actual_aperture_length * 5.0, 10)
        _, resolutions = metrics.resolution_vs_aperture(config, gt_v, ap_sizes)
        _, psr_vals = metrics.psr_vs_aperture(ap_sizes)

        # ── 7. VALIDATION CHECKS (Prompts 6-7) ────────────────
        # Check 1: Doppler axis max == λ/(4·T_PRI)
        expected_vmax = lambda_ / (4 * params.PRI)
        actual_vmax = float(np.max(np.abs(vel_axis)))
        if abs(actual_vmax - expected_vmax) / expected_vmax > 0.02:
            warnings.append(
                f"DOPPLER: max(|v|)={actual_vmax:.2f} ≠ λ/(4·PRI)={expected_vmax:.2f}"
            )

        # Check 2: Ego-motion error
        for name, v_est in [('WM', ego_res['v_wm']),
                            ('OS', ego_res['v_os']),
                            ('DCM', ego_res['v_dcm'])]:
            err = abs(v_est - gt_v)
            if err > 5.0:
                warnings.append(f"{name} ego error = {err:.2f} m/s (> 5 threshold)")

        # Check 3: PSR < 5 dB → focusing failure
        if psr < 5.0:
            warnings.append(f"PSR = {psr:.1f} dB (< 5 dB threshold — focus failure)")

        # Check 4: Resolution must decrease with aperture
        if len(resolutions) >= 2 and resolutions[-1] > resolutions[0]:
            warnings.append("Resolution INCREASES with aperture — inversion bug")

        # ── 8. Build response ──────────────────────────────────
        rd_b = _sub2d(rd_before_db, 128, 128)
        rd_a = _sub2d(rd_after_db, 128, 128)
        sc = _sub2d(sar_conv, 128, 128)
        si = _sub2d(sar_interp, 128, 128)
        r_ax = _sub1d(range_axis, 128)
        v_ax = _sub1d(vel_axis, 128)
        sr = _sub1d(sar_r, 128)
        sa = _sub1d(sar_az, 128)

        dt = time.time() - t0

        debug = None
        if params.debug:
            debug = {
                'doppler_axis_hz': _sub1d(
                    np.fft.fftshift(np.fft.fftfreq(params.M, params.PRI)), 128
                ).tolist(),
                'velocity_bins': v_ax.tolist(),
                'theoretical_resolution': float(lambda_ / (2 * actual_aperture_length)),
                'measured_resolution': float(az_res_measured),
                'expected_vmax': float(expected_vmax),
                'actual_vmax': float(actual_vmax),
                'lambda': float(lambda_),
                'range_resolution': float(3e8 / (2 * params.B)),
            }

        return SimulationResult(
            rd_map_before_si=rd_b.tolist(),
            rd_map_after_si=rd_a.tolist(),
            range_axis=r_ax.tolist(),
            velocity_axis=v_ax.tolist(),
            si_reduction_db=float(si_met['reduction_db']),
            si_iterations=int(si_met['iterations']),
            velocity_ground_truth=gt_v,
            velocity_wm=float(ego_res['v_wm']),
            velocity_os=float(ego_res['v_os']),
            velocity_dcm=float(ego_res['v_dcm']),
            ego_motion_metrics=ego_res.get('metrics', {}),
            sar_image_conventional=sc.tolist(),
            sar_image_interpolated=si.tolist(),
            sar_range_axis=sr.tolist(),
            sar_azimuth_axis=sa.tolist(),
            azimuth_resolution_measured=float(az_res_measured),
            psr=float(psr),
            position_error=float(pos_err) if pos_err is not None else None,
            resolution_vs_aperture={
                'aperture_sizes': ap_sizes.tolist(),
                'resolutions': resolutions.tolist(),
            },
            psr_vs_aperture={
                'aperture_sizes': ap_sizes.tolist(),
                'psr_values': psr_vals.tolist(),
            },
            validation_warnings=warnings,
            computation_time=dt,
            debug_info=debug,
        )

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# ── Helpers ────────────────────────────────────────────────────────
def _sub2d(a, mr, mc):
    r, c = a.shape
    return a[::max(1, r//mr), ::max(1, c//mc)]

def _sub1d(a, ms):
    return a[::max(1, len(a)//ms)]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
