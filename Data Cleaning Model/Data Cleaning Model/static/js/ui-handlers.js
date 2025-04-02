function showLoadingState(message) {
    Swal.fire({
        title: 'Processing...',
        text: message,
        allowOutsideClick: false,
        didOpen: () => Swal.showLoading()
    });
}

function showSuccess(message) {
    Swal.fire({
        icon: 'success',
        title: 'Success!',
        text: message
    });
}

function showError(message) {
    Swal.fire({
        icon: 'error',
        title: 'Error',
        text: message,
        background: '#1f2937',
        color: '#fff'
    });
}

function updateDataPreview(preview) {
    const previewDiv = document.getElementById('dataPreview');
    if (!preview || !Array.isArray(preview) || preview.length === 0) {
        previewDiv.innerHTML = '<p class="text-gray-400 text-center">No data available</p>';
        return;
    }

    const columns = Object.keys(preview[0]);
    const tableHTML = generateTableHTML(columns, preview);
    previewDiv.innerHTML = tableHTML;
}

function generateTableHTML(columns, data) {
    return `
        <table class="min-w-full bg-gray-700 rounded-lg overflow-hidden">
            <thead class="bg-gray-600">
                <tr>${columns.map(col => 
                    `<th class="px-4 py-2 text-left text-sm font-semibold text-gray-200">${col}</th>`
                ).join('')}</tr>
            </thead>
            <tbody>
                ${data.map((row, index) => `
                    <tr class="${index % 2 === 0 ? 'bg-gray-800' : 'bg-gray-750'}">
                        ${columns.map(col => 
                            `<td class="px-4 py-2 text-sm text-gray-300">${row[col] ?? ''}</td>`
                        ).join('')}
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

function initializeFileUpload() {
    const uploadZone = document.querySelector('.upload-zone');
    const fileInput = document.getElementById('fileInput');

    const preventDefaults = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults);
        document.body.addEventListener(eventName, preventDefaults);
    });

    uploadZone.addEventListener('dragenter', () => uploadZone.classList.add('border-purple-500'));
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('border-purple-500'));
    uploadZone.addEventListener('drop', e => handleFileUpload(e.dataTransfer.files[0]));
    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => handleFileUpload(e.target.files[0]));
}

// ... Add other UI-related functions
