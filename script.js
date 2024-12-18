function processMedia() {
    const fileInput = document.getElementById('media-upload');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a file to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    document.getElementById('upload-section').style.display = 'none';
    document.getElementById('loading-section').style.display = 'block';

    fetch('http://localhost:5000/process', {
        method: 'POST',
        body: formData,
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('loading-section').style.display = 'none';

            if (data.message === 'Image processed successfully') {
                const outputImage = document.getElementById('processed-image');
                outputImage.src = data.image_data; // Base64 image data from backend
                outputImage.style.display = 'block';

                document.getElementById('processed-video').style.display = 'none';
                document.getElementById('output-section').style.display = 'block';
            } else {
                alert('Error processing file: ' + (data.error || 'Unknown error'));
                document.getElementById('upload-section').style.display = 'block';
            }

            console.log('Backend Response:', data); // Debugging response
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while processing the file.');
            document.getElementById('loading-section').style.display = 'none';
            document.getElementById('upload-section').style.display = 'block';
        });
}
