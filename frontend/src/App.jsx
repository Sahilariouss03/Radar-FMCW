import { useState } from 'react';
import './App.css';

function App() {
  const [simulationData, setSimulationData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('rd');

  const handleRunSimulation = async (params) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Simulation failed (${response.status}): ${text}`);
      }

      const data = await response.json();
      setSimulationData(data);
    } catch (err) {
      setError(err.message);
      console.error('Simulation error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Automotive FMCW SAR Simulation</h1>
        <p>Production-Grade Radar Signal Processing &amp; Synthetic Aperture Imaging</p>
      </header>

      <ParameterPanel onRunSimulation={handleRunSimulation} isLoading={isLoading} />

      {error && (
        <div className="glass-card" style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)' }}>
          <p style={{ color: '#ef4444' }}><strong>Error:</strong> {error}</p>
          <p style={{ color: '#9ca3af', marginTop: '0.5rem', fontSize: '0.9rem' }}>
            Make sure the backend server is running on http://localhost:8000
          </p>
        </div>
      )}

      {simulationData && (
        <div className="fade-in">
          <div className="glass-card" style={{ marginBottom: '2rem', padding: '1rem' }}>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              {[
                { key: 'rd', label: 'Range-Doppler Maps' },
                { key: 'ego', label: 'Ego-Motion' },
                { key: 'sar', label: 'SAR Images' },
                { key: 'metrics', label: 'Quality Metrics' }
              ].map(tab => (
                <button
                  key={tab.key}
                  className={`btn ${activeTab === tab.key ? 'btn-primary' : ''}`}
                  style={activeTab !== tab.key ? {
                    background: 'rgba(30,30,50,0.6)', color: '#9ca3af',
                    border: '1px solid rgba(75,85,99,0.3)', cursor: 'pointer'
                  } : {}}
                  onClick={() => setActiveTab(tab.key)}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>

          {activeTab === 'rd' && <RDSection data={simulationData} />}
          {activeTab === 'ego' && <EgoMotionSection data={simulationData} />}
          {activeTab === 'sar' && <SARSection data={simulationData} />}
          {activeTab === 'metrics' && <MetricsSection data={simulationData} />}
        </div>
      )}

      {!simulationData && !error && !isLoading && (
        <div className="glass-card" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
          <h2 style={{ color: '#d1d5db', marginBottom: '1rem' }}>Ready to Simulate</h2>
          <p style={{ color: '#9ca3af' }}>Configure parameters above and click &quot;Run Simulation&quot; to begin</p>
        </div>
      )}
    </div>
  );
}

/* ===== PARAMETER PANEL ===== */
function ParameterPanel({ onRunSimulation, isLoading }) {
  const [params, setParams] = useState({
    fc: 77e9, B: 500e6, Tsw: 40e-6, PRI: 50e-6,
    M: 256, N: 512, v0: 20.0, acceleration: 0.0,
    velocity_noise_std: 0.0, h0: 1.5,
    synthetic_aperture_length: 10.0, num_scatterers: 100,
    num_point_targets: 3, noise_power: 1e-6,
    si_amplitude: 0.1, max_si_iterations: 5
  });

  const handleChange = (key, value) => {
    setParams(prev => ({ ...prev, [key]: parseFloat(value) }));
  };

  const sliders = [
    { key: 'fc', label: 'Carrier Frequency (GHz)', min: 76, max: 81, step: 0.5, scale: 1e9, display: v => (v / 1e9).toFixed(1) },
    { key: 'B', label: 'Bandwidth (MHz)', min: 100, max: 1000, step: 50, scale: 1e6, display: v => (v / 1e6).toFixed(0) },
    { key: 'v0', label: 'Velocity (m/s)', min: 5, max: 40, step: 1, scale: 1, display: v => v.toFixed(1) },
    { key: 'acceleration', label: 'Acceleration (m/s\u00B2)', min: -5, max: 5, step: 0.5, scale: 1, display: v => v.toFixed(1) },
    { key: 'h0', label: 'Radar Height (m)', min: 0.5, max: 3, step: 0.1, scale: 1, display: v => v.toFixed(1) },
    { key: 'synthetic_aperture_length', label: 'Aperture Length (m)', min: 5, max: 20, step: 1, scale: 1, display: v => v.toFixed(1) },
    { key: 'num_scatterers', label: 'Scatterers', min: 50, max: 200, step: 10, scale: 1, display: v => v.toFixed(0) },
    { key: 'noise_power', label: 'Noise (dB)', min: -80, max: -40, step: 5, scale: 'log', display: v => (10 * Math.log10(v)).toFixed(1) }
  ];

  return (
    <div className="glass-card parameter-panel">
      <h2 className="card-title">Simulation Parameters</h2>
      <div className="parameter-grid">
        {sliders.map(s => (
          <div key={s.key} className="parameter-group">
            <label className="parameter-label">
              {s.label}
              <span className="parameter-value">{s.display(params[s.key])}</span>
            </label>
            <input
              type="range"
              min={s.min} max={s.max} step={s.step}
              value={s.scale === 'log' ? 10 * Math.log10(params[s.key]) : params[s.key] / s.scale}
              onChange={(e) => {
                const raw = parseFloat(e.target.value);
                const val = s.scale === 'log' ? Math.pow(10, raw / 10) : raw * s.scale;
                handleChange(s.key, val);
              }}
            />
          </div>
        ))}
      </div>
      <button className="btn btn-primary btn-full" onClick={() => onRunSimulation(params)} disabled={isLoading}>
        {isLoading ? (<><span className="spinner"></span> Running Simulation...</>) : 'Run Simulation'}
      </button>
      <div className="info-panel">
        <p><strong>Tip:</strong> Higher bandwidth improves range resolution. Longer synthetic aperture improves azimuth resolution.</p>
      </div>
    </div>
  );
}

/* ===== RANGE-DOPPLER SECTION ===== */
function RDSection({ data }) {
  const rd = data;
  return (
    <div className="glass-card">
      <h2 className="card-title">Range-Doppler Maps</h2>
      <div className="metrics-grid" style={{ marginBottom: '1.5rem' }}>
        <div className="metric-card">
          <div className="metric-label">SI Reduction</div>
          <div className="metric-value">{rd.si_reduction_db?.toFixed(1) ?? 'N/A'}<span className="metric-unit"> dB</span></div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Iterations</div>
          <div className="metric-value">{rd.si_iterations ?? 'N/A'}</div>
        </div>
      </div>
      <div className="comparison-grid">
        <div className="comparison-item">
          <h3>Before Interference Cancellation</h3>
          <HeatmapCanvas mapData={rd.rd_map_before_si} xAxis={rd.range_axis} yAxis={rd.velocity_axis} xLabel="Range (m)" yLabel="Velocity (m/s)" cmap="jet" />
        </div>
        <div className="comparison-item">
          <h3>After Interference Cancellation</h3>
          <HeatmapCanvas mapData={rd.rd_map_after_si} xAxis={rd.range_axis} yAxis={rd.velocity_axis} xLabel="Range (m)" yLabel="Velocity (m/s)" cmap="jet" />
        </div>
      </div>
      <div className="info-panel">
        <p><strong>Range-Doppler Processing:</strong> 2D FFT converts beat signal to range and velocity domains. Self-interference appears as a bright zero-Doppler line and is removed iteratively.</p>
      </div>
    </div>
  );
}

/* ===== EGO-MOTION SECTION ===== */
function EgoMotionSection({ data }) {
  const methods = [
    { label: 'Ground Truth', value: data.velocity_ground_truth, color: '#10b981', rmse: null },
    { label: 'Weighted Mean', value: data.velocity_wm, color: '#3b82f6', rmse: data.ego_motion_metrics?.wm_rmse },
    { label: 'Order Statistics', value: data.velocity_os, color: '#8b5cf6', rmse: data.ego_motion_metrics?.os_rmse },
    { label: 'DCM', value: data.velocity_dcm, color: '#f59e0b', rmse: data.ego_motion_metrics?.dcm_rmse }
  ];
  const maxV = Math.max(...methods.map(m => Math.abs(m.value || 0)), 1);

  return (
    <div className="glass-card">
      <h2 className="card-title">Ego-Motion Estimation</h2>
      <div className="metrics-grid">
        {methods.map(m => (
          <div key={m.label} className="metric-card">
            <div className="metric-label">{m.label}</div>
            <div className="metric-value" style={{ color: m.color }}>
              {(m.value ?? 0).toFixed(2)}<span className="metric-unit"> m/s</span>
            </div>
            {m.rmse != null && (
              <div style={{ fontSize: '0.8rem', color: '#9ca3af', marginTop: '0.5rem' }}>RMSE: {m.rmse.toFixed(3)} m/s</div>
            )}
          </div>
        ))}
      </div>
      <div className="plot-container" style={{ marginTop: '2rem', padding: '1.5rem' }}>
        <h3 style={{ color: '#d1d5db', marginBottom: '1rem', textAlign: 'center' }}>Velocity Comparison</h3>
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: '1rem', justifyContent: 'center', height: '200px', padding: '0 2rem' }}>
          {methods.map(m => {
            const barH = Math.max((Math.abs(m.value || 0) / maxV) * 180, 10);
            return (
              <div key={m.label} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ color: '#d1d5db', fontSize: '0.8rem' }}>{(m.value ?? 0).toFixed(1)}</span>
                <div style={{ width: '50px', height: barH + 'px', background: `linear-gradient(180deg, ${m.color}, ${m.color}88)`, borderRadius: '6px 6px 0 0' }} />
                <span style={{ color: '#9ca3af', fontSize: '0.7rem', textAlign: 'center', maxWidth: '70px' }}>{m.label}</span>
              </div>
            );
          })}
        </div>
      </div>
      <div className="info-panel">
        <p><strong>Ego-Motion:</strong> WM uses power-weighted average, OS uses median (robust to outliers), DCM tracks Doppler cell migration across range bins.</p>
      </div>
    </div>
  );
}

/* ===== SAR SECTION ===== */
function SARSection({ data }) {
  return (
    <div className="glass-card">
      <h2 className="card-title">SAR Image Formation</h2>
      <div className="comparison-grid">
        <div className="comparison-item">
          <h3>Conventional RDA</h3>
          <HeatmapCanvas mapData={data.sar_image_conventional} xAxis={data.sar_azimuth_axis} yAxis={data.sar_range_axis} xLabel="Azimuth (m)" yLabel="Range (m)" cmap="hot" />
        </div>
        <div className="comparison-item">
          <h3>Interpolated RDA</h3>
          <HeatmapCanvas mapData={data.sar_image_interpolated} xAxis={data.sar_azimuth_axis} yAxis={data.sar_range_axis} xLabel="Azimuth (m)" yLabel="Range (m)" cmap="hot" />
        </div>
      </div>
      <div className="info-panel">
        <p><strong>SAR Imaging:</strong> RDA performs range compression, RCMC, and azimuth matched filtering. Interpolated version compensates for non-uniform velocity.</p>
      </div>
    </div>
  );
}

/* ===== METRICS SECTION ===== */
function MetricsSection({ data }) {
  return (
    <div className="glass-card">
      <h2 className="card-title">SAR Performance Metrics</h2>
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-label">Azimuth Resolution</div>
          <div className="metric-value">{data.azimuth_resolution?.toFixed(3) ?? 'N/A'}<span className="metric-unit"> m</span></div>
        </div>
        <div className="metric-card">
          <div className="metric-label">PSR</div>
          <div className="metric-value">{data.psr?.toFixed(1) ?? 'N/A'}<span className="metric-unit"> dB</span></div>
        </div>
        {data.position_error != null && (
          <div className="metric-card">
            <div className="metric-label">Position Error</div>
            <div className="metric-value">{Math.abs(data.position_error).toFixed(3)}<span className="metric-unit"> m</span></div>
          </div>
        )}
        <div className="metric-card">
          <div className="metric-label">SI Reduction</div>
          <div className="metric-value">{data.si_reduction_db?.toFixed(1) ?? 'N/A'}<span className="metric-unit"> dB</span></div>
        </div>
      </div>

      {data.resolution_vs_aperture && data.psr_vs_aperture && (
        <div className="grid grid-2" style={{ marginTop: '2rem' }}>
          <div>
            <h3 style={{ color: '#d1d5db', marginBottom: '1rem', textAlign: 'center' }}>Resolution vs Aperture</h3>
            <LineChartCanvas xData={data.resolution_vs_aperture.aperture_sizes} yData={data.resolution_vs_aperture.resolutions} xLabel="Aperture (m)" yLabel="Resolution (m)" color="#3b82f6" />
          </div>
          <div>
            <h3 style={{ color: '#d1d5db', marginBottom: '1rem', textAlign: 'center' }}>PSR vs Aperture</h3>
            <LineChartCanvas xData={data.psr_vs_aperture.aperture_sizes} yData={data.psr_vs_aperture.psr_values} xLabel="Aperture (m)" yLabel="PSR (dB)" color="#8b5cf6" />
          </div>
        </div>
      )}

      <div className="info-panel">
        <p><strong>Quality:</strong> Lower resolution value = better separation. Higher PSR = better sidelobe suppression. Larger aperture improves both.</p>
      </div>
    </div>
  );
}

/* ===== CANVAS HEATMAP COMPONENT ===== */
function HeatmapCanvas({ mapData, xAxis, yAxis, xLabel, yLabel, cmap }) {
  const [tooltip, setTooltip] = useState(null);
  const [el, setEl] = useState(null);

  const W = 500, H = 350;
  const mg = { t: 10, r: 20, b: 40, l: 55 };
  const pW = W - mg.l - mg.r, pH = H - mg.t - mg.b;

  const refCb = (canvas) => {
    if (!canvas || !mapData || mapData.length === 0) return;
    setEl(canvas);
    const ctx = canvas.getContext('2d');
    const rows = mapData.length, cols = mapData[0].length;

    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++) {
        const v = mapData[i][j];
        if (isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; }
      }
    const rng = mx - mn || 1;

    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, W, H);

    const cW = pW / cols, cH = pH / rows;
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++) {
        const norm = isFinite(mapData[i][j]) ? (mapData[i][j] - mn) / rng : 0;
        ctx.fillStyle = cmap === 'hot' ? hotColor(norm) : jetColor(norm);
        ctx.fillRect(mg.l + j * cW, mg.t + i * cH, Math.ceil(cW), Math.ceil(cH));
      }

    ctx.strokeStyle = '#4b5563'; ctx.lineWidth = 1;
    ctx.strokeRect(mg.l, mg.t, pW, pH);

    ctx.fillStyle = '#d1d5db'; ctx.font = '12px Inter, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(xLabel, mg.l + pW / 2, H - 5);
    ctx.save(); ctx.translate(12, mg.t + pH / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0); ctx.restore();

    ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
    for (let i = 0; i <= 5; i++) {
      const idx = Math.min(Math.floor(i * (xAxis.length - 1) / 5), xAxis.length - 1);
      ctx.fillText(xAxis[idx].toFixed(1), mg.l + (idx / (xAxis.length - 1)) * pW, H - mg.b + 15);
    }
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const idx = Math.min(Math.floor(i * (yAxis.length - 1) / 5), yAxis.length - 1);
      ctx.fillText(yAxis[idx].toFixed(1), mg.l - 5, mg.t + (idx / (yAxis.length - 1)) * pH + 4);
    }

    const bx = W - 15, bw = 10;
    for (let i = 0; i < pH; i++) {
      ctx.fillStyle = cmap === 'hot' ? hotColor(1 - i / pH) : jetColor(1 - i / pH);
      ctx.fillRect(bx, mg.t + i, bw, 1);
    }
    ctx.font = '9px Inter, sans-serif'; ctx.fillStyle = '#9ca3af'; ctx.textAlign = 'left';
    ctx.fillText(mx.toFixed(0), bx + bw + 2, mg.t + 8);
    ctx.fillText(mn.toFixed(0), bx + bw + 2, mg.t + pH);
  };

  const onMove = (e) => {
    if (!el || !mapData) return;
    const rect = el.getBoundingClientRect();
    const sx = (e.clientX - rect.left) * (W / rect.width);
    const sy = (e.clientY - rect.top) * (H / rect.height);
    const rows = mapData.length, cols = mapData[0].length;
    const col = Math.floor((sx - mg.l) / (pW / cols));
    const row = Math.floor((sy - mg.t) / (pH / rows));
    if (col >= 0 && col < cols && row >= 0 && row < rows) {
      setTooltip({
        x: e.clientX + 10, y: e.clientY - 40,
        text: `${xLabel.split('(')[0].trim()}: ${xAxis[col]?.toFixed(2)}, ${yLabel.split('(')[0].trim()}: ${yAxis[row]?.toFixed(2)}, Value: ${mapData[row][col]?.toFixed(1)}`
      });
    } else setTooltip(null);
  };

  return (
    <div className="plot-container" style={{ position: 'relative' }}>
      <canvas ref={refCb} width={W} height={H} style={{ width: '100%', height: 'auto', cursor: 'crosshair' }} onMouseMove={onMove} onMouseLeave={() => setTooltip(null)} />
      {tooltip && (
        <div style={{ position: 'fixed', left: tooltip.x, top: tooltip.y, background: 'rgba(31,41,55,0.95)', color: '#f9fafb', padding: '6px 10px', borderRadius: '6px', fontSize: '12px', pointerEvents: 'none', zIndex: 9999, border: '1px solid rgba(75,85,99,0.5)' }}>
          {tooltip.text}
        </div>
      )}
    </div>
  );
}

/* ===== CANVAS LINE CHART COMPONENT ===== */
function LineChartCanvas({ xData, yData, xLabel, yLabel, color }) {
  const W = 400, H = 250;
  const mg = { t: 15, r: 15, b: 40, l: 55 };
  const pW = W - mg.l - mg.r, pH = H - mg.t - mg.b;

  const refCb = (canvas) => {
    if (!canvas || !xData || !yData || xData.length === 0) return;
    const ctx = canvas.getContext('2d');
    const xMin = Math.min(...xData), xMax = Math.max(...xData);
    const yMin = Math.min(...yData), yMax = Math.max(...yData);
    const xR = xMax - xMin || 1, yR = yMax - yMin || 1;

    ctx.fillStyle = '#111827'; ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = 'rgba(75,85,99,0.3)'; ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = mg.t + (i / 4) * pH;
      ctx.beginPath(); ctx.moveTo(mg.l, y); ctx.lineTo(mg.l + pW, y); ctx.stroke();
    }

    ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i < xData.length; i++) {
      const x = mg.l + ((xData[i] - xMin) / xR) * pW;
      const y = mg.t + pH - ((yData[i] - yMin) / yR) * pH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = color;
    for (let i = 0; i < xData.length; i++) {
      const x = mg.l + ((xData[i] - xMin) / xR) * pW;
      const y = mg.t + pH - ((yData[i] - yMin) / yR) * pH;
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
    }

    ctx.strokeStyle = '#4b5563'; ctx.lineWidth = 1;
    ctx.strokeRect(mg.l, mg.t, pW, pH);

    ctx.fillStyle = '#d1d5db'; ctx.font = '12px Inter, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(xLabel, mg.l + pW / 2, H - 5);
    ctx.save(); ctx.translate(12, mg.t + pH / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0); ctx.restore();

    ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
    for (let i = 0; i <= 4; i++) ctx.fillText((xMin + (i / 4) * xR).toFixed(1), mg.l + (i / 4) * pW, H - mg.b + 15);
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) ctx.fillText((yMax - (i / 4) * yR).toFixed(1), mg.l - 5, mg.t + (i / 4) * pH + 4);
  };

  return (
    <div className="plot-container">
      <canvas ref={refCb} width={W} height={H} style={{ width: '100%', height: 'auto' }} />
    </div>
  );
}

/* ===== COLORMAP UTILITIES ===== */
function jetColor(t) {
  t = Math.max(0, Math.min(1, t));
  let r, g, b;
  if (t < 0.125) { r = 0; g = 0; b = 0.5 + t * 4; }
  else if (t < 0.375) { r = 0; g = (t - 0.125) * 4; b = 1; }
  else if (t < 0.625) { r = (t - 0.375) * 4; g = 1; b = 1 - (t - 0.375) * 4; }
  else if (t < 0.875) { r = 1; g = 1 - (t - 0.625) * 4; b = 0; }
  else { r = 1 - (t - 0.875) * 4; g = 0; b = 0; }
  return `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
}

function hotColor(t) {
  t = Math.max(0, Math.min(1, t));
  let r, g, b;
  if (t < 0.33) { r = t / 0.33; g = 0; b = 0; }
  else if (t < 0.67) { r = 1; g = (t - 0.33) / 0.34; b = 0; }
  else { r = 1; g = 1; b = (t - 0.67) / 0.33; }
  return `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
}

export default App;
