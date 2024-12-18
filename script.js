function processMedia() {
    const fileInput = document.getElementById('media-upload');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a file');
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    fetch('http://127.0.0.1:5000/process', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.message === 'File processed successfully') {
            if (file.type.startsWith('image')) {
                // Display the processed image
                const outputImage = document.getElementById('output-image');
                outputImage.src = 'data:image/jpeg;base64,' + data.image_data;
                outputImage.style.display = 'block';
            } else if (file.type.startsWith('video')) {
                // Display the processed video
                const outputVideo = document.getElementById('output-video');
                outputVideo.src = 'data:video/mp4;base64,' + data.video_data;
                outputVideo.style.display = 'block';
            }
        } else {
            alert('Error processing file');
        }
    })
    .catch(error => {
        console.error('Error processing file:', error);
        alert('Error processing file');
    });
}
