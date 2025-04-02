async function handleFileUpload(file) {
    if (!file) return;

    showLoadingState('Uploading and analyzing data...');

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/upload', { 
            method: 'POST', 
            body: formData 
        });
    
        const data = await response.json();
        
        if (!data.success) throw new Error(data.error || 'Upload failed');
        
        // Ensure data structure
        const processedData = {
            preview: sanitizeData(data.preview || []),
            columns: data.columns || [],
            analysis: data.analysis || {},
            total_rows: data.total_rows || 0,
            total_columns: data.total_columns || 0,
            success: true
        };

        currentData = processedData;
        originalData = {...processedData};
        
        updateUI(processedData);
        showSuccess('File uploaded and analyzed successfully');
    } catch (error) {
        showError(error.message);
        console.error('Upload error:', error);
    }
}

function sanitizeData(data) {
    if (!Array.isArray(data)) return [];
    
    return data.map(row => {
        if (!row) return {};
        
        const cleanRow = {};
        Object.entries(row).forEach(([key, value]) => {
            cleanRow[key] = value === "NaN" || value === null ? '' : value;
        });
        return cleanRow;
    });
}

// Move all your data operation functions here (applyMissingDataOperation, removeDuplicates, changeDataType, etc.)
// ...existing code for data operations...
