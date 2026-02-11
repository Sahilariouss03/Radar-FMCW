"""Debug: inspect per-range-bin Doppler peaks."""
import numpy as np
from radar.fmcw import FMCWRadar
from radar.range_doppler import RangeDopplerProcessor
from radar.interference import InterferenceCanceller
from radar.ego_motion import EgoMotionEstimator

config = {"fc": 77e9, "B": 500e6, "Tsw": 40e-6, "PRI": 50e-6,
          "M": 128, "N": 256, "h0": 1.5, "c": 3e8}

radar = FMCWRadar(config)
v_prof = radar.generate_velocity_profile(20.0)

# Use scatterers directly ahead (large x, small y) so radial velocity = ego velocity
scat = radar.generate_scatterers(100, x_range=(5, 100), y_range=(-2, 2))
print(f"Scatterer x range: {scat[:,0].min():.1f} to {scat[:,0].max():.1f}")
print(f"Scatterer y range: {scat[:,1].min():.1f} to {scat[:,1].max():.1f}")

beat, meta = radar.generate_beat_signal(v_prof, scat, noise_power=1e-8, si_amplitude=0.01)

proc = RangeDopplerProcessor(config)
rd_c, r_ax, v_ax = proc.compute_range_doppler_complex(beat)
rd_mag = np.abs(rd_c)

print(f"\nVelocity axis: {v_ax[0]:.2f} to {v_ax[-1]:.2f}, {len(v_ax)} bins")
print(f"Range axis: {r_ax[0]:.2f} to {r_ax[-1]:.2f}, {len(r_ax)} bins")

# Top 10 peaks
print("\nTop 10 peaks in RD map:")
rd_temp = rd_mag.copy()
for i in range(10):
    pk = np.unravel_index(np.argmax(rd_temp), rd_temp.shape)
    print(f"  #{i}: v={v_ax[pk[0]]:.2f} m/s, R={r_ax[pk[1]]:.1f} m, mag={rd_temp[pk]:.4f}")
    rd_temp[max(0,pk[0]-2):pk[0]+3, max(0,pk[1]-2):pk[1]+3] = 0

# Check if ego-motion finds the right velocities
ego = EgoMotionEstimator(config)
r = ego.estimate_all_methods(rd_c, v_ax, r_ax, ground_truth=20.0)
print(f"\nWM={r['v_wm']:.2f}  OS={r['v_os']:.2f}  DCM={r['v_dcm']:.2f}")
