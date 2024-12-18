document.getElementById('uploadForm').addEventListener('submit', async (event) => {
    event.preventDefault(); // Prevent the default form submission behavior

    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a file to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        // Send POST request to backend
        const response = await fetch('http://127.0.0.1:5000/process', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            alert(`Error: ${errorData.error}`);
            return;
        }

        const data = await response.json();

        // Check if image data is returned
        if (data.image_data) {
            // Set the processed image in the output container
            const outputImage = document.getElementById('outputImage');
            outputImage.src = `data:image/png;base64,${data.image_data}`;
            outputImage.style.display = 'block'; // Ensure the image is visible
        } else {
            alert('Processed image data is not available.');
        }
    } catch (error) {
        alert(`An error occurred: ${error.message}`);
    }
});
