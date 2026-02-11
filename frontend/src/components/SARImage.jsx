import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

const SARImage = ({ data }) => {
    const conventionalRef = useRef(null);
    const interpolatedRef = useRef(null);

    useEffect(() => {
        if (!data) return;

        const { sar_image_conventional, sar_image_interpolated, sar_range_axis, sar_azimuth_axis } = data;

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(17, 24, 39, 0.8)',
            font: { color: '#f9fafb', family: 'Inter, sans-serif' },
            xaxis: {
                title: 'Azimuth (m)',
                gridcolor: 'rgba(75, 85, 99, 0.3)',
                color: '#d1d5db'
            },
            yaxis: {
                title: 'Range (m)',
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

        // Conventional RDA
        const traceConventional = {
            z: sar_image_conventional,
            x: sar_azimuth_axis,
            y: sar_range_axis,
            type: 'heatmap',
            colorscale: 'Hot',
            colorbar: {
                title: 'Magnitude',
                titleside: 'right',
                tickfont: { color: '#d1d5db' },
                titlefont: { color: '#f9fafb' }
            },
            hovertemplate: 'Azimuth: %{x:.2f} m<br>Range: %{y:.2f} m<br>Magnitude: %{z:.2f}<extra></extra>'
        };

        Plotly.newPlot(
            conventionalRef.current,
            [traceConventional],
            { ...layout, title: 'Conventional RDA' },
            config
        );

        // Interpolated RDA
        const traceInterpolated = {
            z: sar_image_interpolated,
            x: sar_azimuth_axis,
            y: sar_range_axis,
            type: 'heatmap',
            colorscale: 'Hot',
            colorbar: {
                title: 'Magnitude',
                titleside: 'right',
                tickfont: { color: '#d1d5db' },
                titlefont: { color: '#f9fafb' }
            },
            hovertemplate: 'Azimuth: %{x:.2f} m<br>Range: %{y:.2f} m<br>Magnitude: %{z:.2f}<extra></extra>'
        };

        Plotly.newPlot(
            interpolatedRef.current,
            [traceInterpolated],
            { ...layout, title: 'Interpolated RDA (Nonuniform Velocity)' },
            config
        );

    }, [data]);

    if (!data) {
        return (
            <div className="glass-card">
                <h2 className="card-title">SAR Images</h2>
                <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '3rem' }}>
                    Run simulation to view SAR images
                </p>
            </div>
        );
    }

    return (
        <div className="glass-card">
            <h2 className="card-title">SAR Image Formation</h2>

            <div className="comparison-grid">
                <div className="comparison-item">
                    <div ref={conventionalRef} className="plot-container"></div>
                </div>
                <div className="comparison-item">
                    <div ref={interpolatedRef} className="plot-container"></div>
                </div>
            </div>

            <div className="info-panel">
                <p>
                    <strong>SAR Imaging:</strong> Range-Doppler Algorithm (RDA) forms high-resolution 2D image.
                    Conventional RDA assumes constant velocity. Interpolated RDA compensates for nonuniform velocity through resampling.
                </p>
            </div>
        </div>
    );
};

export default SARImage;
