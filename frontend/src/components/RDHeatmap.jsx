import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

const RDHeatmap = ({ data }) => {
    const beforeRef = useRef(null);
    const afterRef = useRef(null);

    useEffect(() => {
        if (!data) return;

        const { rd_map_before_si, rd_map_after_si, range_axis, velocity_axis, si_reduction_db } = data;

        // Plot configuration
        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(17, 24, 39, 0.8)',
            font: { color: '#f9fafb', family: 'Inter, sans-serif' },
            xaxis: {
                title: 'Range (m)',
                gridcolor: 'rgba(75, 85, 99, 0.3)',
                color: '#d1d5db'
            },
            yaxis: {
                title: 'Velocity (m/s)',
                gridcolor: 'rgba(75, 85, 99, 0.3)',
                color: '#d1d5db'
            },
            margin: { l: 60, r: 60, t: 40, b: 60 },
            hovermode: 'closest'
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        };

        // Before interference cancellation
        const traceBefore = {
            z: rd_map_before_si,
            x: range_axis,
            y: velocity_axis,
            type: 'heatmap',
            colorscale: 'Jet',
            colorbar: {
                title: 'Magnitude (dB)',
                titleside: 'right',
                tickfont: { color: '#d1d5db' },
                titlefont: { color: '#f9fafb' }
            },
            hovertemplate: 'Range: %{x:.2f} m<br>Velocity: %{y:.2f} m/s<br>Magnitude: %{z:.1f} dB<extra></extra>'
        };

        Plotly.newPlot(
            beforeRef.current,
            [traceBefore],
            { ...layout, title: 'Before Interference Cancellation' },
            config
        );

        // After interference cancellation
        const traceAfter = {
            z: rd_map_after_si,
            x: range_axis,
            y: velocity_axis,
            type: 'heatmap',
            colorscale: 'Jet',
            colorbar: {
                title: 'Magnitude (dB)',
                titleside: 'right',
                tickfont: { color: '#d1d5db' },
                titlefont: { color: '#f9fafb' }
            },
            hovertemplate: 'Range: %{x:.2f} m<br>Velocity: %{y:.2f} m/s<br>Magnitude: %{z:.1f} dB<extra></extra>'
        };

        Plotly.newPlot(
            afterRef.current,
            [traceAfter],
            { ...layout, title: `After Cancellation (${si_reduction_db.toFixed(1)} dB reduction)` },
            config
        );

    }, [data]);

    if (!data) {
        return (
            <div className="glass-card">
                <h2 className="card-title">Range-Doppler Maps</h2>
                <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '3rem' }}>
                    Run simulation to view Range-Doppler maps
                </p>
            </div>
        );
    }

    return (
        <div className="glass-card">
            <h2 className="card-title">Range-Doppler Maps</h2>

            <div className="comparison-grid">
                <div className="comparison-item">
                    <div ref={beforeRef} className="plot-container"></div>
                </div>
                <div className="comparison-item">
                    <div ref={afterRef} className="plot-container"></div>
                </div>
            </div>

            <div className="info-panel">
                <p>
                    <strong>Range-Doppler Processing:</strong> 2D FFT converts beat signal to range (horizontal) and velocity (vertical) domains.
                    Zero-Doppler interference appears as bright horizontal line at zero velocity and is removed through iterative cancellation.
                </p>
            </div>
        </div>
    );
};

export default RDHeatmap;
