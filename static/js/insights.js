$(document).ready(function() {
    $('#uploadForm').on('submit', function(e) {
        e.preventDefault();
        
        const fileInput = $('#fileInput')[0];
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        $('#loading').show();
        $('#results').hide();
        
        $.ajax({
            url: '/insights-prediction',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#loading').hide();
                $('#results').show();
                
                // Display results
                let resultsHtml = '<div class="results-content">';
                for (const [key, value] of Object.entries(response.results)) {
                    resultsHtml += `<div class="result-item">
                        <h4>${key}</h4>
                        <p>${value}</p>
                    </div>`;
                }
                resultsHtml += '</div>';
                
                $('#resultsContent').html(resultsHtml);
            },
            error: function(xhr) {
                $('#loading').hide();
                alert('Error: ' + xhr.responseJSON.error);
            }
        });
    });
}); 