function processImage() {
    const fileInput = document.getElementById('image-upload');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select an image');
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
        
        if (data.message === 'Image processed successfully') {
            const outputImage = document.getElementById('output-image');
            // Ensure that the image is properly base64 encoded
            outputImage.src = 'data:image/jpeg;base64,' + data.processed_image_data;
            outputImageContainer.style.display = 'block'; // Show image
        } else {
            alert('Error processing image: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while processing the image.');
    });
}
