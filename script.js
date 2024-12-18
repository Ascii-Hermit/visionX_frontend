function processMedia() {
    const fileInput = document.getElementById('media-upload');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select an image or video');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Send the file to the backend for processing
    fetch('http://localhost:5000/process', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const outputImageContainer = document.getElementById('output-image-container');
        const outputVideoContainer = document.getElementById('output-video-container');
        
        if (data.message === 'Image processed successfully' || data.message === 'Video processed successfully') {
            if (file.type.startsWith('image')) {
                const outputImage = document.getElementById('output-image');
                outputImage.src = 'data:image/jpeg;base64,' + data.processed_image_data;
                outputImageContainer.style.display = 'block';
            } else if (file.type.startsWith('video')) {
                const outputVideo = document.getElementById('output-video');
                outputVideo.src = 'data:video/mp4;base64,' + data.processed_video_data;
                outputVideoContainer.style.display = 'block';
            }
        } else {
            alert('Error processing file: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while processing the file.');
    });
}
