function initializeVisualizations() {
    const visualType = document.getElementById('visualizationType');
    const visualColumn = document.getElementById('visualizationColumn');

    if (visualType && visualColumn) {
        visualType.addEventListener('change', updateVisualization);
        visualColumn.addEventListener('change', updateVisualization);
    }
}

function updateVisualization() {
    const type = document.getElementById('visualizationType').value;
    const column = document.getElementById('visualizationColumn').value;

    if (!currentData?.preview) return;

    switch(type) {
        case 'missing':
            createMissingValuesPlot();
            createDataTypesPlot();
            break;
        case 'distribution':
            if (column) createDistributionPlot(column);
            break;
        case 'correlation':
            createCorrelationMatrix();
            break;
        case 'boxplot':
            if (column) createBoxPlot(column);
            break;
    }
}

function createMissingValuesPlot() {
    if (!currentData?.analysis?.missing_values) return;

    const data = [{
        y: Object.keys(currentData.analysis.missing_values),
        x: Object.values(currentData.analysis.missing_values),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: '#8b5cf6'
        }
    }];

    const layout = {
        title: 'Missing Values by Column',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#fff' },
        xaxis: { title: 'Count' },
        yaxis: { title: 'Column' }
    };

    Plotly.newPlot('primaryPlot', data, layout);
}

function createDistributionPlot(column) {
    if (!currentData?.preview) return;

    const values = currentData.preview
        .map(row => parseFloat(row[column]))
        .filter(val => !isNaN(val));

    const trace = {
        x: values,
        type: 'histogram',
        marker: {
            color: '#8b5cf6'
        }
    };

    const layout = {
        title: `Distribution of ${column}`,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#fff' },
        xaxis: { title: column },
        yaxis: { title: 'Count' }
    };

    Plotly.newPlot('primaryPlot', [trace], layout);
}

function createBoxPlot(column) {
    if (!currentData?.preview) return;

    const values = currentData.preview
        .map(row => parseFloat(row[column]))
        .filter(val => !isNaN(val));

    const trace = {
        y: values,
        type: 'box',
        name: column,
        marker: {
            color: '#8b5cf6'
        }
    };

    const layout = {
        title: `Box Plot of ${column}`,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#fff' }
    };

    Plotly.newPlot('primaryPlot', [trace], layout);
}

function createCorrelationMatrix() {
    if (!currentData?.preview || !currentData?.analysis?.numeric_columns) return;

    const numericColumns = currentData.analysis.numeric_columns;
    const data = currentData.preview;

    // Calculate correlation matrix
    const matrix = [];
    const values = numericColumns.map(col => 
        data.map(row => parseFloat(row[col])).filter(val => !isNaN(val))
    );

    for (let i = 0; i < numericColumns.length; i++) {
        matrix[i] = [];
        for (let j = 0; j < numericColumns.length; j++) {
            matrix[i][j] = calculateCorrelation(values[i], values[j]);
        }
    }

    const trace = {
        z: matrix,
        x: numericColumns,
        y: numericColumns,
        type: 'heatmap',
        colorscale: 'Viridis'
    };

    const layout = {
        title: 'Correlation Matrix',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#fff' }
    };

    Plotly.newPlot('primaryPlot', [trace], layout);
}

function calculateCorrelation(x, y) {
    const n = x.length;
    const sum_x = x.reduce((a, b) => a + b, 0);
    const sum_y = y.reduce((a, b) => a + b, 0);
    const sum_xy = x.reduce((a, b, i) => a + b * y[i], 0);
    const sum_xx = x.reduce((a, b) => a + b * b, 0);
    const sum_yy = y.reduce((a, b) => a + b * b, 0);

    const correlation = (n * sum_xy - sum_x * sum_y) / 
        Math.sqrt((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y));

    return isNaN(correlation) ? 0 : correlation;
}
