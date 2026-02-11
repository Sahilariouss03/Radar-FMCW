"""Comprehensive physics validation — writes to file for full output."""
import requests, json

r = requests.post("http://localhost:8000/simulate", json={
    "fc": 77e9, "B": 500e6, "Tsw": 40e-6, "PRI": 50e-6,
    "M": 128, "N": 256, "v0": 20.0, "h0": 1.5,
    "synthetic_aperture_length": 10.0, "num_scatterers": 50,
    "num_point_targets": 3, "noise_power": 1e-6,
    "si_amplitude": 0.1, "max_si_iterations": 10, "debug": True,
}, timeout=300)

lines = []
if r.status_code != 200:
    lines.append(f"ERR {r.status_code}")
    lines.append(r.text[:3000])
else:
    d = r.json()
    gt = d["velocity_ground_truth"]
    lines.append("=== ROUND 2 PHYSICS VALIDATION ===")
    lines.append(f"GT={gt:.1f}  WM={d['velocity_wm']:.2f}  OS={d['velocity_os']:.2f}  DCM={d['velocity_dcm']:.2f}")
    lines.append(f"WM_err={abs(d['velocity_wm']-gt):.2f}  OS_err={abs(d['velocity_os']-gt):.2f}  DCM_err={abs(d['velocity_dcm']-gt):.2f}")
    lines.append(f"SI reduction: {d['si_reduction_db']:.1f} dB (target >=15)")
    lines.append(f"SI iterations: {d['si_iterations']}")
    lines.append(f"PSR: {d['psr']:.1f} dB (target >10)")
    lines.append(f"Measured resolution: {d['azimuth_resolution_measured']:.6f} m")
    lines.append(f"Position error: {d.get('position_error')}")
    lines.append(f"Computation time: {d['computation_time']:.2f} s")
    rv = d["resolution_vs_aperture"]
    lines.append(f"Res@2m={rv['resolutions'][0]:.6f}  Res@20m={rv['resolutions'][-1]:.6f}")
    lines.append(f"Res decreases: {rv['resolutions'][-1] < rv['resolutions'][0]}")
    lines.append(f"Warnings: {json.dumps(d['validation_warnings'], indent=2)}")
    db = d.get("debug_info", {})
    if db:
        lines.append(f"TheoRes={db['theoretical_resolution']:.6f}  MeasRes={db['measured_resolution']:.6f}")
        lines.append(f"vmax: actual={db['actual_vmax']:.2f}  expected={db['expected_vmax']:.2f}")

with open("test_results.txt", "w") as f:
    f.write("\n".join(lines))
print("Results written to test_results.txt")
