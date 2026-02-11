import { useState } from 'react';

const ParameterPanel = ({ onRunSimulation, isLoading }) => {
  const [params, setParams] = useState({
    fc: 77e9,
    B: 500e6,
    Tsw: 40e-6,
    PRI: 50e-6,
    M: 256,
    N: 512,
    v0: 20.0,
    acceleration: 0.0,
    velocity_noise_std: 0.0,
    h0: 1.5,
    synthetic_aperture_length: 10.0,
    num_scatterers: 100,
    num_point_targets: 3,
    noise_power: 1e-6,
    si_amplitude: 0.1,
    max_si_iterations: 5
  });

  const handleChange = (key, value) => {
    setParams(prev => ({ ...prev, [key]: parseFloat(value) }));
  };

  const handleSubmit = () => {
    onRunSimulation(params);
  };

  return (
    <div className="glass-card parameter-panel">
      <h2 className="card-title">Simulation Parameters</h2>

      <div className="parameter-grid">
        {/* Radar Parameters */}
        <div className="parameter-group">
          <label className="parameter-label">
            Carrier Frequency (GHz)
            <span className="parameter-value">{(params.fc / 1e9).toFixed(1)}</span>
          </label>
          <input
            type="range"
            min="76"
            max="81"
            step="0.5"
            value={params.fc / 1e9}
            onChange={(e) => handleChange('fc', e.target.value * 1e9)}
          />
        </div>

        <div className="parameter-group">
          <label className="parameter-label">
            Bandwidth (MHz)
            <span className="parameter-value">{(params.B / 1e6).toFixed(0)}</span>
          </label>
          <input
            type="range"
            min="100"
            max="1000"
            step="50"
            value={params.B / 1e6}
            onChange={(e) => handleChange('B', e.target.value * 1e6)}
          />
        </div>

        <div className="parameter-group">
          <label className="parameter-label">
            Velocity (m/s)
            <span className="parameter-value">{params.v0.toFixed(1)}</span>
          </label>
          <input
            type="range"
            min="5"
            max="40"
            step="1"
            value={params.v0}
            onChange={(e) => handleChange('v0', e.target.value)}
          />
        </div>

        <div className="parameter-group">
          <label className="parameter-label">
            Acceleration (m/s²)
            <span className="parameter-value">{params.acceleration.toFixed(1)}</span>
          </label>
          <input
            type="range"
            min="-5"
            max="5"
            step="0.5"
            value={params.acceleration}
            onChange={(e) => handleChange('acceleration', e.target.value)}
          />
        </div>

        <div className="parameter-group">
          <label className="parameter-label">
            Radar Height (m)
            <span className="parameter-value">{params.h0.toFixed(1)}</span>
          </label>
          <input
            type="range"
            min="0.5"
            max="3"
            step="0.1"
            value={params.h0}
            onChange={(e) => handleChange('h0', e.target.value)}
          />
        </div>

        <div className="parameter-group">
          <label className="parameter-label">
            Aperture Length (m)
            <span className="parameter-value">{params.synthetic_aperture_length.toFixed(1)}</span>
          </label>
          <input
            type="range"
            min="5"
            max="20"
            step="1"
            value={params.synthetic_aperture_length}
            onChange={(e) => handleChange('synthetic_aperture_length', e.target.value)}
          />
        </div>

        <div className="parameter-group">
          <label className="parameter-label">
            Number of Scatterers
            <span className="parameter-value">{params.num_scatterers}</span>
          </label>
          <input
            type="range"
            min="50"
            max="200"
            step="10"
            value={params.num_scatterers}
            onChange={(e) => handleChange('num_scatterers', e.target.value)}
          />
        </div>

        <div className="parameter-group">
          <label className="parameter-label">
            Noise Level (dB)
            <span className="parameter-value">{(10 * Math.log10(params.noise_power)).toFixed(1)}</span>
          </label>
          <input
            type="range"
            min="-80"
            max="-40"
            step="5"
            value={10 * Math.log10(params.noise_power)}
            onChange={(e) => handleChange('noise_power', Math.pow(10, e.target.value / 10))}
          />
        </div>
      </div>

      <button
        className="btn btn-primary btn-full"
        onClick={handleSubmit}
        disabled={isLoading}
      >
        {isLoading ? (
          <>
            <span className="spinner"></span>
            Running Simulation...
          </>
        ) : (
          'Run Simulation'
        )}
      </button>

      <div className="info-panel">
        <p>
          <strong>Tip:</strong> Adjust parameters to explore different radar configurations.
          Higher bandwidth improves range resolution, while longer synthetic aperture improves azimuth resolution.
        </p>
      </div>
    </div>
  );
};

export default ParameterPanel;
