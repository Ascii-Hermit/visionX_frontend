function processImage() {
    const fileInput = document.getElementById('image-upload');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select an image');
        return;
    }

    // Hide the upload section and show loading spinner
    document.getElementById('upload-section').style.display = 'none';
    document.getElementById('loading-section').style.display = 'flex';

    const formData = new FormData();
    formData.append("file", file);

    fetch('http://localhost:5000/process', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.message === 'Image processed successfully') {
            const processedImageSrc = 'data:image/jpeg;base64,' + data.image_data;
            document.getElementById('processed-image').src = processedImageSrc;
            document.getElementById('loading-section').style.display = 'none';
            document.getElementById('output-section').style.display = 'block';
        } else {
            alert('Error processing image');
            document.getElementById('loading-section').style.display = 'none';
            document.getElementById('upload-section').style.display = 'block';
        }
    })
    .catch(error => {
        console.error('Error processing image:', error);
        alert('Error processing image');
        document.getElementById('loading-section').style.display = 'none';
        document.getElementById('upload-section').style.display = 'block';
    });
}
