
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

const VelocityGraph = ({ data }) => {
    if (!data) {
        return (
            <div className="glass-card">
                <h2 className="card-title">Ego-Motion Estimation</h2>
                <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '3rem' }}>
                    Run simulation to view velocity estimates
                </p>
            </div>
        );
    }

    const { velocity_wm, velocity_os, velocity_dcm, velocity_ground_truth, ego_motion_metrics } = data;

    // Prepare data for comparison chart
    const comparisonData = [
        {
            method: 'Ground Truth',
            velocity: velocity_ground_truth,
            fill: '#10b981'
        },
        {
            method: 'Weighted Mean',
            velocity: velocity_wm,
            fill: '#3b82f6'
        },
        {
            method: 'Order Statistics',
            velocity: velocity_os,
            fill: '#8b5cf6'
        },
        {
            method: 'DCM',
            velocity: velocity_dcm,
            fill: '#f59e0b'
        }
    ];

    return (
        <div className="glass-card">
            <h2 className="card-title">Ego-Motion Estimation</h2>

            <div className="metrics-grid">
                <div className="metric-card">
                    <div className="metric-label">Ground Truth</div>
                    <div className="metric-value">
                        {velocity_ground_truth.toFixed(2)}
                        <span className="metric-unit">m/s</span>
                    </div>
                </div>

                <div className="metric-card">
                    <div className="metric-label">Weighted Mean (WM)</div>
                    <div className="metric-value">
                        {velocity_wm.toFixed(2)}
                        <span className="metric-unit">m/s</span>
                    </div>
                    {ego_motion_metrics?.wm_rmse && (
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                            RMSE: {ego_motion_metrics.wm_rmse.toFixed(3)} m/s
                        </div>
                    )}
                </div>

                <div className="metric-card">
                    <div className="metric-label">Order Statistics (OS)</div>
                    <div className="metric-value">
                        {velocity_os.toFixed(2)}
                        <span className="metric-unit">m/s</span>
                    </div>
                    {ego_motion_metrics?.os_rmse && (
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                            RMSE: {ego_motion_metrics.os_rmse.toFixed(3)} m/s
                        </div>
                    )}
                </div>

                <div className="metric-card">
                    <div className="metric-label">Doppler Cell Migration (DCM)</div>
                    <div className="metric-value">
                        {velocity_dcm.toFixed(2)}
                        <span className="metric-unit">m/s</span>
                    </div>
                    {ego_motion_metrics?.dcm_rmse && (
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                            RMSE: {ego_motion_metrics.dcm_rmse.toFixed(3)} m/s
                        </div>
                    )}
                </div>
            </div>

            <div className="plot-container" style={{ marginTop: '2rem', height: '300px' }}>
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={comparisonData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(75, 85, 99, 0.3)" />
                        <XAxis dataKey="method" stroke="#d1d5db" />
                        <YAxis stroke="#d1d5db" label={{ value: 'Velocity (m/s)', angle: -90, position: 'insideLeft', fill: '#d1d5db' }} />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: 'rgba(31, 41, 55, 0.95)',
                                border: '1px solid rgba(75, 85, 99, 0.3)',
                                borderRadius: '8px',
                                color: '#f9fafb'
                            }}
                        />
                        <Bar dataKey="velocity" fill="#3b82f6" />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            <div className="info-panel">
                <p>
                    <strong>Ego-Motion Estimation:</strong> Three algorithms estimate vehicle velocity from Range-Doppler map.
                    WM uses weighted average, OS uses median (robust to outliers), and DCM tracks Doppler cell migration across range bins.
                </p>
            </div>
        </div>
    );
};

export default VelocityGraph;
