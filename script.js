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

    fetch('http://127.0.0.1:5000/process-video-matplotlib', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (response.ok) {
            alert('Matplotlib will display the video in the server environment.');
        } else {
            alert('Error processing video.');
        }
    })
    .catch(error => {
        console.error('Error processing video:', error);
        alert('Error processing video.');
    });
}
