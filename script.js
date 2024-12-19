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

function processVideo() {
    const fileInput = document.getElementById('video-upload');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a video');
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    fetch('http://127.0.0.1:5000/process-video', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        const url = URL.createObjectURL(blob);
        window.open(url, '_blank'); // Open processed video in a new tab
    })
    .catch(error => {
        console.error('Error processing video:', error);
        alert('Error processing video');
    });
}

function runRealTimeVideo() {
    fetch('http://127.0.0.1:5000/start-video', {
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

function stopProcessing() {
    fetch('http://127.0.0.1:5000/stop-video', {
        method: 'POST'
    })
    .then(response => {
        if (response.ok) {
            alert('Processing stopped successfully.');
        } else {
            alert('Error stopping the process.');
        }
    })
    .catch(error => {
        console.error('Error stopping the process:', error);
        alert('Error stopping the process.');
    });
}
