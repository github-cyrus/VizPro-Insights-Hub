let currentVisualization = null;

function initializeVisualizations() {
    const visType = document.getElementById('visualizationType');
    const visColumn = document.getElementById('visualizationColumn');

    visType.addEventListener('change', () => {
        updateColumnVisibility();
    });

    updateColumnVisibility();
}

function updateColumnVisibility() {
    const visType = document.getElementById('visualizationType').value;
    const columnSelect = document.getElementById('visualizationColumn');
    const columnContainer = columnSelect.parentElement;

    // Hide column selector for visualizations that don't need it
    if (['summary', 'correlation', 'missing'].includes(visType)) {
        columnContainer.style.display = 'none';
    } else {
        columnContainer.style.display = 'block';
    }
}

function generateVisualization() {
    const visType = document.getElementById('visualizationType').value;
    const selectedColumn = document.getElementById('visualizationColumn').value;

    // Validate inputs
    if (!currentData) {
        showError('No data available for visualization');
        return;
    }

    if (!['summary', 'correlation', 'missing'].includes(visType) && !selectedColumn) {
        showError('Please select a column for visualization');
        return;
    }

    // Show loading state
    const container = document.getElementById('visualization-container');
    container.innerHTML = '<div class="flex justify-center items-center h-[400px]"><div class="animate-spin rounded-full h-32 w-32 border-b-2 border-purple-500"></div></div>';

    // Prepare visualization request
    const request = {
        type: visType,
        data: currentData.preview,
        column: selectedColumn
    };

    // Send request to server
    fetch('/visualize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(request)
    })
    .then(response => response.json())
    .then(result => {
        if (result.success && result.plot) {
            // Clear previous visualization
            container.innerHTML = '';
            
            // Create new visualization
            createVisualization(result.plot, 'visualization-container');
            currentVisualization = {data: result.plot.data, layout: result.plot.layout};
        } else {
            throw new Error(result.error || 'Failed to create visualization');
        }
    })
    .catch(error => {
        console.error('Visualization error:', error);
        showError(`Failed to create visualization: ${error.message}`);
    });
}

function updateVisualizationColumns(columns) {
    const columnSelect = document.getElementById('visualizationColumn');
    columnSelect.innerHTML = '<option value="">Select Column</option>';
    
    if (columns && columns.length) {
        columns.forEach(column => {
            const option = document.createElement('option');
            option.value = column;
            option.textContent = column;
            columnSelect.appendChild(option);
        });
    }
}

function resizeVisualization() {
    if (currentVisualization) {
        Plotly.relayout('visualization-container', {
            width: document.getElementById('visualization-container').offsetWidth,
            height: 500
        });
    }
}

// Add window resize handler
window.addEventListener('resize', debounce(resizeVisualization, 250));

// Utility function for debouncing
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function createVisualization(plotData, containerId) {
    const container = document.getElementById(containerId);
    
    // Set responsive behavior
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };
    
    // Create the plot with the data and layout from the server
    Plotly.newPlot(containerId, plotData.data, plotData.layout, config);
    
    // Handle window resize
    window.addEventListener('resize', function() {
        Plotly.Plots.resize(containerId);
    });
}

function updatePlotSize(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        Plotly.Plots.resize(containerId);
    }
}
