function processMedia() {
    const fileInput = document.getElementById('media-upload');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a file to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Show the loading section
    document.getElementById('upload-section').style.display = 'none';
    document.getElementById('loading-section').style.display = 'block';

    fetch('http://localhost:5000/process', {
        method: 'POST',
        body: formData,
    })
        .then(response => response.json())
        .then(data => {
            // Hide loading section
            document.getElementById('loading-section').style.display = 'none';

            if (data.media_type === 'image') {
                const outputImage = document.getElementById('processed-image');
                outputImage.src = data.media_data; // Base64 image data
                outputImage.style.display = 'block';
                document.getElementById('processed-video').style.display = 'none';
            } else if (data.media_type === 'video') {
                const videoSource = document.getElementById('video-source');
                videoSource.src = data.media_data; // Base64 video data or URL
                videoSource.parentElement.load(); // Reload the video element
                videoSource.parentElement.style.display = 'block';
                document.getElementById('processed-image').style.display = 'none';
            }

            document.getElementById('output-section').style.display = 'block';
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while processing the file.');
            document.getElementById('loading-section').style.display = 'none';
            document.getElementById('upload-section').style.display = 'block';
        });
}
