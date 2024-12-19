function processImage() {
    const fileInput = document.getElementById('image-upload');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select an image');
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
        if (data.message === 'Image processed successfully') {
            // Assuming the server returns base64-encoded image data
            const processedImageSrc = 'data:image/jpeg;base64,' + data.image_data; 
            document.getElementById('processed-image').src = processedImageSrc;
            document.getElementById('processed-image').style.display = 'block';
        } else {
            alert('Error processing image');
        }
    })
    .catch(error => {
        console.error('Error processing image:', error);
        alert('Error processing image');
    });
}

function runRealTimeVideo() {
    fetch('http://127.0.0.1:5000/realtime-video', {
        method: 'POST'
    })
    .then(response => {
        if (response.ok) {
            alert('Real-time video processing started successfully.');
        } else {
            alert('Error starting real-time video processing.');
        }
    })
    .catch(error => {
        console.error('Error starting real-time video processing:', error);
        alert('Error starting real-time video processing.');
    });
}
