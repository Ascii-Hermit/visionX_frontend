function uploadFile() {
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    fetch('http://127.0.0.1:5000/process', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            if (file.type.startsWith('image')) {
                const imageElement = document.getElementById('output-image');
                imageElement.src = 'data:image/jpeg;base64,' + data.image_data;
                imageElement.style.display = 'block';  // Show the image
                document.getElementById('output-video').style.display = 'none';  // Hide video
            } else if (file.type.startsWith('video')) {
                const videoElement = document.getElementById('output-video');
                videoElement.src = 'data:video/mp4;base64,' + data.video_data;
                videoElement.style.display = 'block';  // Show the video
                document.getElementById('output-image').style.display = 'none';  // Hide image
            }
        }
    })
    .catch(error => {
        alert('Error processing file: ' + error);
    });
}
