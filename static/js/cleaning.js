$(document).ready(function() {
    $('#uploadForm').on('submit', function(e) {
        e.preventDefault();
        
        const fileInput = $('#fileInput')[0];
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        $('#loading').show();
        $('#results').hide();
        $('#downloadBtn').hide();
        
        $.ajax({
            url: '/data-cleaning',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#loading').hide();
                $('#results').show();
                $('#downloadBtn').show();
                
                // Display preview of cleaned data
                let previewHtml = '<div class="table-responsive"><table class="table">';
                // Add table content based on response format
                previewHtml += '</table></div>';
                
                $('#resultsContent').html(previewHtml);
            },
            error: function(xhr) {
                $('#loading').hide();
                alert('Error: ' + xhr.responseJSON.error);
            }
        });
    });
}); 