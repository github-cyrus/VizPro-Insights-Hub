let currentData = null;
let originalData = null;

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    initializeVisualizations();
});

function initializeApp() {
    initializeFileUpload();
    initializeEventListeners();
    initializeVisualizations(); // Add this line
}

function initializeEventListeners() {
    // Initialize missing data operation handler
    const missingDataOp = document.getElementById('missingDataOperation');
    if (missingDataOp) {
        missingDataOp.addEventListener('change', (e) => {
            const customValueInput = document.getElementById('customValue');
            customValueInput.style.display = e.target.value === 'fill_value' ? 'block' : 'none';
        });
    }
}

async function handleFileUpload(file) {
    if (!file) return;

    showLoadingState('Uploading and analyzing data...');

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/upload', { method: 'POST', body: formData });
        const data = await response.json();

        if (!data.success) throw new Error(data.error);

        currentData = data;
        originalData = {...data};
        
        updateUI(data);
        showSuccess('File uploaded and analyzed successfully');
    } catch (error) {
        showError(error.message);
        console.error('Upload error:', error);
    }
}

// UI Update Functions
function updateUI(data) {
    if (!data?.preview) {
        console.error('Invalid data structure received');
        return;
    }

    try {
        updateDataPreview(data.preview);
        updateColumnSelectors(data.columns || Object.keys(data.preview[0]));
        if (data.analysis) {
            updateColumnAnalysis(data.analysis);
            updateVisualizationColumns(data.columns); // Add this line
        }
        updateVisualization(); // Add this line
        updateVisualizationOptions(data); // Add this line
    } catch (error) {
        console.error('Error updating UI:', error);
    }
}

function updateColumnAnalysis(analysis) {
    const analysisDiv = document.getElementById('columnAnalysis');
    if (!analysisDiv) return;

    let html = '';
    
    // For each column
    for (const column of [...analysis.numeric_columns, ...analysis.categorical_columns]) {
        const dataType = analysis.data_types[column];
        const missingCount = analysis.missing_values[column];
        const uniqueCount = analysis.unique_counts[column];
        const stats = analysis.column_stats[column];
        
        html += `
            <div class="bg-gray-700 p-4 rounded-lg">
                <h3 class="text-lg font-semibold text-purple-300 mb-2">${column}</h3>
                <div class="text-sm text-gray-300">
                    <p><span class="font-medium">Type:</span> ${dataType}</p>
                    <p><span class="font-medium">Missing Values:</span> ${missingCount}</p>
                    <p><span class="font-medium">Unique Values:</span> ${uniqueCount}</p>
                    ${renderColumnStats(stats, analysis.numeric_columns.includes(column))}
                </div>
            </div>
        `;
    }
    
    analysisDiv.innerHTML = html;
}

function renderColumnStats(stats, isNumeric) {
    if (!stats) return '';
    
    if (isNumeric) {
        return `
            <div class="mt-2">
                <p><span class="font-medium">Mean:</span> ${stats.mean?.toFixed(2) ?? 'N/A'}</p>
                <p><span class="font-medium">Median:</span> ${stats.median?.toFixed(2) ?? 'N/A'}</p>
                <p><span class="font-medium">Std:</span> ${stats.std?.toFixed(2) ?? 'N/A'}</p>
                <p><span class="font-medium">Min:</span> ${stats.min?.toFixed(2) ?? 'N/A'}</p>
                <p><span class="font-medium">Max:</span> ${stats.max?.toFixed(2) ?? 'N/A'}</p>
            </div>
        `;
    } else {
        return `
            <div class="mt-2">
                <p class="font-medium">Top Values:</p>
                ${Object.entries(stats.top_values)
                    .map(([value, count]) => `<p>${value}: ${count}</p>`)
                    .join('')}
            </div>
        `;
    }
}

// Update data preview table
function updateDataPreview(preview) {
    const previewDiv = document.getElementById('dataPreview');
    
    // Handle empty or invalid data
    if (!preview) {
        previewDiv.innerHTML = '<p class="text-gray-400 text-center">No data available</p>';
        return;
    }

    try {
        // Convert to array if it's an object
        const previewData = Array.isArray(preview) ? preview : [preview];
        
        if (previewData.length === 0) {
            previewDiv.innerHTML = '<p class="text-gray-400 text-center">No data available</p>';
            return;
        }

        // Get columns from the first row
        const columns = Object.keys(previewData[0]);
        
        // Create table HTML
        let tableHTML = `
            <table class="min-w-full bg-gray-700 rounded-lg overflow-hidden">
                <thead class="bg-gray-600">
                    <tr>
                        ${columns.map(col => `<th class="px-4 py-2 text-left text-sm font-semibold text-gray-200">${col}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
        `;

        // Add data rows
        previewData.forEach((row, index) => {
            tableHTML += `
                <tr class="${index % 2 === 0 ? 'bg-gray-800' : 'bg-gray-750'}">
                    ${columns.map(col => `<td class="px-4 py-2 text-sm text-gray-300">${row[col] ?? ''}</td>`).join('')}
                </tr>
            `;
        });

        tableHTML += '</tbody></table>';
        previewDiv.innerHTML = tableHTML;

    } catch (error) {
        console.error('Error updating preview:', error);
        previewDiv.innerHTML = '<p class="text-gray-400 text-center">Error displaying data preview</p>';
    }
}

// Update column selectors
function updateColumnSelectors(columns) {
    // Guard clause for undefined columns
    if (!columns || !Array.isArray(columns)) {
        console.error('Invalid columns data');
        return;
    }

    const selectors = ['missingDataColumn', 'dataTypeColumn'];
    selectors.forEach(selectorId => {
        const select = document.getElementById(selectorId);
        if (!select) return; // Skip if element doesn't exist

        select.innerHTML = '<option value="">Select Column</option>';
        columns.forEach(column => {
            const option = document.createElement('option');
            option.value = column;
            option.textContent = column;
            select.appendChild(option);
        });
    });

    // Update duplicate columns checkboxes
    const duplicateColumns = document.getElementById('duplicateColumns');
    if (duplicateColumns) {
        duplicateColumns.innerHTML = columns.map(column => `
            <label class="flex items-center mb-2">
                <input type="checkbox" value="${column}" class="form-checkbox text-purple-600 mr-2">
                <span class="text-sm">${column}</span>
            </label>
        `).join('');
    }
}

function updateVisualizationColumns(columns) {
    const visualColumn = document.getElementById('visualizationColumn');
    if (!visualColumn) return;

    visualColumn.innerHTML = '<option value="">Select Column</option>';
    columns.forEach(column => {
        const option = document.createElement('option');
        option.value = column;
        option.textContent = column;
        visualColumn.appendChild(option);
    });
}

// Update visualizations with null checks
function updateVisualizations(data) {
    try {
        // Missing data plot
        if (data.missing_plot) {
            Plotly.newPlot('missingDataPlot', JSON.parse(data.missing_plot).data, JSON.parse(data.missing_plot).layout);
        }

        // Data types plot - with null checks
        const dataTypes = data.analysis?.data_types ? 
            Object.entries(data.analysis.data_types).map(([column, type]) => ({
                column,
                type: String(type)
            })) : [];

        if (dataTypes.length > 0) {
            const dataTypesPlot = {
                data: [{
                    type: 'bar',
                    x: dataTypes.map(d => d.column),
                    y: dataTypes.map(d => 1),
                    text: dataTypes.map(d => d.type),
                    textposition: 'auto',
                    marker: {
                        color: '#8b5cf6'
                    }
                }],
                layout: {
                    title: 'Data Types by Column',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {
                        color: '#fff'
                    },
                    showlegend: false
                }
            };
            
            Plotly.newPlot('dataTypesPlot', dataTypesPlot.data, dataTypesPlot.layout);
        }
    } catch (error) {
        console.error('Error updating visualizations:', error);
    }
}

// Apply missing data operation
async function applyMissingDataOperation() {
    const column = document.getElementById('missingDataColumn').value;
    const operation = document.getElementById('missingDataOperation').value;
    const customValue = document.getElementById('customValue').value;

    if (!column) {
        showError('Please select a column');
        return;
    }

    try {
        // Use the full dataset if available, otherwise use preview
        const dataToClean = currentData.full_data || currentData.preview;
        
        // Sanitize data before sending
        const cleanData = dataToClean.map(row => {
            const cleanRow = {};
            for (const [key, value] of Object.entries(row)) {
                cleanRow[key] = value === "NaN" || value === null ? null : value;
            }
            return cleanRow;
        });

        const response = await fetch('/clean', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                data: cleanData,
                operations: [{
                    type: operation,
                    column: column,
                    value: customValue || null
                }]
            })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Operation failed');

        // Update current data with sanitized response
        currentData = {
            ...currentData,
            preview: sanitizeResponseData(data.preview),
            full_data: data.full_data ? sanitizeResponseData(data.full_data) : null,
            analysis: data.analysis
        };
        
        updateUI(currentData);
        showSuccess('Operation applied successfully');
    } catch (error) {
        showError(error.message);
        console.error('Cleaning error:', error);
    }
}

// Add this helper function
function sanitizeResponseData(data) {
    if (!Array.isArray(data)) return [];
    return data.map(row => {
        const cleanRow = {};
        Object.entries(row).forEach(([key, value]) => {
            cleanRow[key] = value === "NaN" || value === null ? '' : value;
        });
        return cleanRow;
    });
}

// Remove duplicates
async function removeDuplicates() {
    const selectedColumns = Array.from(document.querySelectorAll('#duplicateColumns input:checked')).map(cb => cb.value);
    
    if (selectedColumns.length === 0) {
        Swal.fire({
            title: 'Error!',
            text: 'Please select at least one column',
            icon: 'error',
            background: '#1f2937',
            color: '#fff'
        });
        return;
    }

    const operations = [{
        type: 'remove_duplicates',
        columns: selectedColumns
    }];

    try {
        // Use the full dataset if available, otherwise use preview
        const dataToClean = currentData.full_data || currentData.preview;

        const response = await fetch('/clean', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                data: dataToClean,
                operations: operations
            })
        });

        const data = await response.json();
        if (response.ok) {
            currentData = {
                ...currentData,
                preview: sanitizeResponseData(data.preview),
                full_data: data.full_data ? sanitizeResponseData(data.full_data) : null,
                analysis: data.analysis
            };
            updateUI(currentData);
            Swal.fire({
                title: 'Success!',
                text: 'Duplicates removed successfully',
                icon: 'success',
                background: '#1f2937',
                color: '#fff'
            });
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        Swal.fire({
            title: 'Error!',
            text: error.message,
            icon: 'error',
            background: '#1f2937',
            color: '#fff'
        });
    }
}

// Change data type
async function changeDataType() {
    const column = document.getElementById('dataTypeColumn').value;
    const newType = document.getElementById('newDataType').value;

    if (!column) {
        Swal.fire({
            title: 'Error!',
            text: 'Please select a column',
            icon: 'error',
            background: '#1f2937',
            color: '#fff'
        });
        return;
    }

    const operations = [{
        type: 'change_type',
        column: column,
        new_type: newType
    }];

    try {
        // Use the full dataset if available, otherwise use preview
        const dataToClean = currentData.full_data || currentData.preview;

        const response = await fetch('/clean', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                data: dataToClean,
                operations: operations
            })
        });

        const data = await response.json();
        if (response.ok) {
            currentData = {
                ...currentData,
                preview: sanitizeResponseData(data.preview),
                full_data: data.full_data ? sanitizeResponseData(data.full_data) : null,
                analysis: data.analysis
            };
            updateUI(currentData);
            Swal.fire({
                title: 'Success!',
                text: 'Data type changed successfully',
                icon: 'success',
                background: '#1f2937',
                color: '#fff'
            });
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        Swal.fire({
            title: 'Error!',
            text: error.message,
            icon: 'error',
            background: '#1f2937',
            color: '#fff'
        });
    }
}

// Download cleaned data
async function downloadData() {
    if (!currentData) {
        Swal.fire({
            title: 'Error!',
            text: 'No data available to download',
            icon: 'error',
            background: '#1f2937',
            color: '#fff'
        });
        return;
    }

    try {
        const response = await fetch('/download', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                data: currentData.full_data || currentData.preview  // Use full_data if available, fallback to preview
            })
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'cleaned_data.csv';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } else {
            const data = await response.json();
            throw new Error(data.error);
        }
    } catch (error) {
        Swal.fire({
            title: 'Error!',
            text: error.message,
            icon: 'error',
            background: '#1f2937',
            color: '#fff'
        });
    }
}

// Show/hide custom value input based on operation selection
document.getElementById('missingDataOperation').addEventListener('change', (e) => {
    const customValueInput = document.getElementById('customValue');
    customValueInput.style.display = e.target.value === 'fill_value' ? 'block' : 'none';
});

function initializeVisualizations() {
    const visTypeSelect = document.getElementById('visualizationType');
    const visColumnSelect = document.getElementById('visualizationColumn');

    // Update column options when visualization type changes
    visTypeSelect.addEventListener('change', () => {
        updateVisualizationOptions(currentData);
    });

    // Create visualization when column is selected
    visColumnSelect.addEventListener('change', () => {
        updateVisualization();
    });
}

function updateVisualizationOptions(data) {
    const visType = document.getElementById('visualizationType').value;
    const columnSelect = document.getElementById('visualizationColumn');
    columnSelect.innerHTML = '<option value="">Select Column</option>';

    if (!data || !data.length) return;

    const columns = Object.keys(data[0]);
    columns.forEach(column => {
        const option = document.createElement('option');
        option.value = column;
        option.textContent = column;
        columnSelect.appendChild(option);
    });
}

function updateVisualization() {
    const visType = document.getElementById('visualizationType').value;
    const selectedColumn = document.getElementById('visualizationColumn').value;

    if (!currentData || !selectedColumn) return;

    try {
        switch (visType) {
            case 'missing':
                visualizeData('missing_matrix');
                break;
            case 'distribution':
                visualizeData('scatter', { x_column: selectedColumn, y_column: selectedColumn });
                break;
            case 'correlation':
                visualizeData('correlation');
                break;
            case 'boxplot':
                visualizeData('anomalies', { column: selectedColumn });
                break;
            default:
                console.warn('Unsupported visualization type:', visType);
        }
    } catch (error) {
        console.error('Error creating visualization:', error);
        alert('Failed to create visualization');
    }
}
