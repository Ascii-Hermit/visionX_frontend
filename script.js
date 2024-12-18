function processMedia() {
    const fileInput = document.getElementById('media-upload');
    const file = fileInput.files[0]; // Get the selected file

    if (!file) {
        alert('Please select a file to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file); // Append the file to the form data

    // Show the loading section
    document.getElementById('upload-section').style.display = 'none';
    document.getElementById('loading-section').style.display = 'block';

    // Send the file to the backend for processing
    fetch('http://localhost:5000/process', {
        method: 'POST',
        body: formData,
    })
        .then(response => response.json())
        .then(data => {
            // Hide the loading section
            document.getElementById('loading-section').style.display = 'none';

            if (data.message === 'Image processed successfully' || data.message === 'Video processed successfully') {
                document.getElementById('output-section').style.display = 'block';

                if (file.type.startsWith('image')) {
                    // Display the processed image
                    const outputImage = document.getElementById('processed-image');
                    outputImage.src = data.image_data; // Base64 image data from backend
                    outputImage.style.display = 'block';
                    document.getElementById('processed-video').style.display = 'none';
                } else if (file.type.startsWith('video')) {
                    // Display the processed video
                    const outputVideo = document.getElementById('processed-video');
                    const videoSource = document.getElementById('video-source');
                    videoSource.src = data.video_url; // Video URL from backend
                    outputVideo.load(); // Reload the video
                    outputVideo.style.display = 'block';
                    document.getElementById('processed-image').style.display = 'none';
                }
            } else {
                alert('Error processing file: ' + data.error);
                document.getElementById('upload-section').style.display = 'block';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while processing the file.');
            document.getElementById('loading-section').style.display = 'none';
            document.getElementById('upload-section').style.display = 'block';
        });
}
