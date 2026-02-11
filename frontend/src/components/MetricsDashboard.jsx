
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const MetricsDashboard = ({ data }) => {
    if (!data) {
        return (
            <div className="glass-card">
                <h2 className="card-title">SAR Performance Metrics</h2>
                <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '3rem' }}>
                    Run simulation to view SAR quality metrics
                </p>
            </div>
        );
    }

    const { azimuth_resolution, psr, position_error, resolution_vs_aperture, psr_vs_aperture } = data;

    // Prepare data for resolution curve
    const resolutionData = resolution_vs_aperture.aperture_sizes.map((aperture, idx) => ({
        aperture: aperture,
        resolution: resolution_vs_aperture.resolutions[idx]
    }));

    // Prepare data for PSR curve
    const psrData = psr_vs_aperture.aperture_sizes.map((aperture, idx) => ({
        aperture: aperture,
        psr: psr_vs_aperture.psr_values[idx]
    }));

    return (
        <div className="glass-card">
            <h2 className="card-title">SAR Performance Metrics</h2>

            <div className="metrics-grid">
                <div className="metric-card">
                    <div className="metric-label">Azimuth Resolution</div>
                    <div className="metric-value">
                        {azimuth_resolution.toFixed(3)}
                        <span className="metric-unit">m</span>
                    </div>
                </div>

                <div className="metric-card">
                    <div className="metric-label">Peak-to-Sidelobe Ratio</div>
                    <div className="metric-value">
                        {psr.toFixed(1)}
                        <span className="metric-unit">dB</span>
                    </div>
                </div>

                {position_error !== null && (
                    <div className="metric-card">
                        <div className="metric-label">Position Error</div>
                        <div className="metric-value">
                            {Math.abs(position_error).toFixed(3)}
                            <span className="metric-unit">m</span>
                        </div>
                    </div>
                )}

                <div className="metric-card">
                    <div className="metric-label">Interference Reduction</div>
                    <div className="metric-value">
                        {data.si_reduction_db?.toFixed(1) || 'N/A'}
                        <span className="metric-unit">dB</span>
                    </div>
                </div>
            </div>

            <div className="grid grid-2" style={{ marginTop: '2rem' }}>
                <div>
                    <h3 style={{ color: 'var(--text-secondary)', marginBottom: '1rem', textAlign: 'center' }}>
                        Resolution vs Aperture Size
                    </h3>
                    <div className="plot-container" style={{ height: '300px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={resolutionData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(75, 85, 99, 0.3)" />
                                <XAxis
                                    dataKey="aperture"
                                    stroke="#d1d5db"
                                    label={{ value: 'Aperture Size (m)', position: 'insideBottom', offset: -5, fill: '#d1d5db' }}
                                />
                                <YAxis
                                    stroke="#d1d5db"
                                    label={{ value: 'Resolution (m)', angle: -90, position: 'insideLeft', fill: '#d1d5db' }}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: 'rgba(31, 41, 55, 0.95)',
                                        border: '1px solid rgba(75, 85, 99, 0.3)',
                                        borderRadius: '8px',
                                        color: '#f9fafb'
                                    }}
                                />
                                <Line type="monotone" dataKey="resolution" stroke="#3b82f6" strokeWidth={3} dot={{ fill: '#3b82f6', r: 5 }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div>
                    <h3 style={{ color: 'var(--text-secondary)', marginBottom: '1rem', textAlign: 'center' }}>
                        PSR vs Aperture Size
                    </h3>
                    <div className="plot-container" style={{ height: '300px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={psrData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(75, 85, 99, 0.3)" />
                                <XAxis
                                    dataKey="aperture"
                                    stroke="#d1d5db"
                                    label={{ value: 'Aperture Size (m)', position: 'insideBottom', offset: -5, fill: '#d1d5db' }}
                                />
                                <YAxis
                                    stroke="#d1d5db"
                                    label={{ value: 'PSR (dB)', angle: -90, position: 'insideLeft', fill: '#d1d5db' }}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: 'rgba(31, 41, 55, 0.95)',
                                        border: '1px solid rgba(75, 85, 99, 0.3)',
                                        borderRadius: '8px',
                                        color: '#f9fafb'
                                    }}
                                />
                                <Line type="monotone" dataKey="psr" stroke="#8b5cf6" strokeWidth={3} dot={{ fill: '#8b5cf6', r: 5 }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            <div className="info-panel">
                <p>
                    <strong>Quality Metrics:</strong> Azimuth resolution measures target separation capability (lower is better).
                    PSR indicates sidelobe suppression (higher is better). Larger synthetic aperture improves resolution but may increase processing complexity.
                </p>
            </div>
        </div>
    );
};

export default MetricsDashboard;
