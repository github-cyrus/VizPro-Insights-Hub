// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeFileUpload();
    initializeChartTypeFilter();
});

function initializeFileUpload() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('dataFile');

    if (!dropZone || !fileInput) return;

    // Handle click to upload
    dropZone.addEventListener('click', () => fileInput.click());

    // Handle drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-primary');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('border-primary');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-primary');
        
        const files = e.dataTransfer.files;
        if (files.length) {
            fileInput.files = files;
            handleFileUpload();
        }
    });

    // Handle file input change
    fileInput.addEventListener('change', handleFileUpload);
}

function initializeChartTypeFilter() {
    const chartTypeSelect = document.getElementById('chartType');
    if (!chartTypeSelect) return;

    chartTypeSelect.addEventListener('change', function() {
        const selectedType = this.value;
        const cards = document.querySelectorAll('.visualization-card');
        
        cards.forEach(card => {
            if (selectedType === 'all' || card.dataset.type === selectedType) {
                card.style.display = 'block';
            } else {
                card.style.display = 'none';
            }
        });
    });
}

// Add debug function to log information
function debug(message, data) {
    console.log(message, data);
    
    // If debug mode is enabled, add to debug panel
    const debugElement = document.getElementById('debugData');
    if (debugElement && window.location.search.includes('debug=true')) {
        const timestamp = new Date().toLocaleTimeString();
        let content = debugElement.innerHTML;
        content += `[${timestamp}] ${message}\n`;
        
        if (data) {
            try {
                if (typeof data === 'object') {
                    content += JSON.stringify(data, null, 2) + '\n\n';
                } else {
                    content += data + '\n\n';
                }
            } catch (e) {
                content += '[Unable to stringify data]\n\n';
            }
        }
        
        debugElement.innerHTML = content;
        
        // Auto-scroll to bottom
        debugElement.scrollTop = debugElement.scrollHeight;
    }
}

// Update handleFileUpload to include debug information
function handleFileUpload() {
    const fileInput = document.getElementById('dataFile');
    if (!fileInput || !fileInput.files.length) {
        showToast('Please select a file', 'error');
        return;
    }

    const file = fileInput.files[0];
    debug('Processing file', file.name);
    
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showToast('Please upload a CSV file', 'error');
        return;
    }

    showLoading();

    const formData = new FormData();
    formData.append('file', file);

    // First, handle the file upload
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Upload failed with status: ' + response.status);
        }
        return response.json();
    })
    .then(data => {
        debug('Upload response', data);
        if (data.error) throw new Error(data.error);
        showToast('File uploaded successfully', 'success');
        
        // Show preview section
        const previewSection = document.getElementById('previewSection');
        if (previewSection) previewSection.style.display = 'block';
        
        // Get data preview
        return fetch('/preview');
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Preview failed with status: ' + response.status);
        }
        return response.json();
    })
    .then(data => {
        debug('Preview data', data);
        if (data.error) throw new Error(data.error);
        displayDataPreview(data);
        
        // Now get visualizations
        return fetch('/visualize');
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Visualization failed with status: ' + response.status);
        }
        return response.json();
    })
    .then(data => {
        debug('Visualization data', data);
        if (data.error) throw new Error(data.error);
        
        const visualizationSection = document.getElementById('visualizationSection');
        if (visualizationSection) {
            visualizationSection.style.display = 'block';
        }
        
        renderVisualizations(data);
        hideLoading();
    })
    .catch(error => {
        debug('Error in processing', error.message);
        hideLoading();
        showToast(error.message || 'An error occurred', 'error');
        console.error('Error:', error);
    });
}

function renderVisualizations(data) {
    debug('Starting visualization rendering', null);
    
    if (!data) {
        showToast('No visualization data received', 'error');
        return;
    }
    
    if (data.error) {
        showToast(data.error, 'error');
        console.error('Error from server:', data.error);
        return;
    }
    
    if (!data.visualizations) {
        showToast('No visualization data available', 'error');
        return;
    }

    try {
        // Clear existing content
        const plotContainers = document.querySelectorAll('.plotly-graph');
        plotContainers.forEach(container => {
            container.innerHTML = '';
        });

        // Distribution plot
        renderDistributionPlot(data.visualizations.distribution);
        
        // Correlation matrix
        renderCorrelationMatrix(data.visualizations.correlation);
        
        // Feature relationships plot
        renderFeatureRelationships(data.visualizations.feature_relationships);
        
        // Render insights
        renderInsights(data.insights);
        
        debug('Visualization rendering completed', null);
    } catch (error) {
        debug('Error in visualization rendering', error.message);
        console.error('Error rendering visualizations:', error);
        showToast('Error rendering visualizations: ' + error.message, 'error');
    }
}

function renderDistributionPlot(distribution) {
    const container = document.getElementById('distributionPlot');
    if (!container) {
        console.error('Distribution plot container not found');
        return;
    }
    
    if (!distribution || !distribution.length) {
        container.innerHTML = '<div class="alert alert-warning">No distribution data available</div>';
        return;
    }
    
    try {
        console.log('Rendering distribution plot with data:', distribution[0]);
        const plotData = distribution[0].plot.data;
        const plotLayout = distribution[0].plot.layout;
        
        Plotly.newPlot(container, plotData, plotLayout, {responsive: true});
    } catch (err) {
        console.error('Error rendering distribution plot:', err);
        container.innerHTML = '<div class="alert alert-danger">Failed to render distribution plot: ' + err.message + '</div>';
    }
}

function renderCorrelationMatrix(correlation) {
    const container = document.getElementById('correlationPlot');
    if (!container) {
        console.error('Correlation plot container not found');
        return;
    }
    
    if (!correlation) {
        container.innerHTML = '<div class="alert alert-warning">No correlation data available</div>';
        return;
    }
    
    try {
        console.log('Rendering correlation matrix with data:', correlation);
        const plotData = correlation.data;
        const plotLayout = correlation.layout;
        
        Plotly.newPlot(container, plotData, plotLayout, {responsive: true});
    } catch (err) {
        console.error('Error rendering correlation matrix:', err);
        container.innerHTML = '<div class="alert alert-danger">Failed to render correlation matrix: ' + err.message + '</div>';
    }
}

function renderFeatureRelationships(relationships) {
    const container = document.getElementById('relationshipsPlot');
    if (!container) {
        console.error('Relationships plot container not found');
        return;
    }
    
    if (!relationships || !relationships.length) {
        container.innerHTML = '<div class="alert alert-warning">No feature relationship data available</div>';
        return;
    }
    
    try {
        console.log('Rendering feature relationships with data:', relationships[0]);
        const plotData = relationships[0].plot.data;
        const plotLayout = relationships[0].plot.layout;
        
        Plotly.newPlot(container, plotData, plotLayout, {responsive: true});
    } catch (err) {
        console.error('Error rendering feature relationships:', err);
        container.innerHTML = '<div class="alert alert-danger">Failed to render feature relationships: ' + err.message + '</div>';
    }
}

function renderInsights(insights) {
    const insightsList = document.getElementById('insightsList');
    if (!insightsList) return;

    if (!insights || !insights.length) {
        insightsList.innerHTML = `
            <div class="list-group-item">
                <i class="fas fa-info-circle text-info me-2"></i>
                No significant insights found in the data.
            </div>`;
        return;
    }

    insightsList.innerHTML = insights.map(insight => `
        <div class="list-group-item">
            <i class="fas fa-lightbulb text-warning me-2"></i>
            ${insight.text}
        </div>
    `).join('');
}

function displayDataPreview(data) {
    const previewDiv = document.getElementById('dataPreview');
    if (!previewDiv || !data || !data.length) {
        if (previewDiv) {
            previewDiv.innerHTML = '<p class="text-white">No data to display</p>';
        }
        return;
    }

    const headers = Object.keys(data[0]);
    const table = `
        <div class="table-responsive">
            <table class="table table-dark table-striped table-hover">
                <thead>
                    <tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>
                </thead>
                <tbody>
                    ${data.map(row => `
                        <tr>${headers.map(h => `<td>${row[h]}</td>`).join('')}</tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
    previewDiv.innerHTML = table;
}

function showLoading() {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) {
        overlay.style.display = 'flex';
    }
}

function hideLoading() {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type === 'error' ? 'bg-danger' : 'bg-success'}`;
    toast.innerHTML = `
        <div class="toast-content">
            <i class="fas ${type === 'error' ? 'fa-exclamation-circle' : 'fa-check-circle'} me-2"></i>
            <span>${message}</span>
        </div>
    `;

    container.appendChild(toast);
    setTimeout(() => {
        toast.remove();
    }, 3000);
} 