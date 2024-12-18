// Function to handle file uploads
async function uploadFile() {
    const formData = new FormData();
    const fileInput = document.getElementById("file-input");
    const file = fileInput.files[0];
    formData.append("file", file);

    try {
        const response = await fetch("/process", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (data.message === "Video processed successfully") {
            // If a video is processed, handle it
            displayProcessedVideo(data.video_url);
        } else {
            // Handle image processing response (if it's an image)
            const imgData = data.image_data;
            displayProcessedImage(imgData);
        }
    } catch (error) {
        alert("Error processing the file");
    }
}

// Function to display processed image (if it's an image)
function displayProcessedImage(imgData) {
    const outputDiv = document.getElementById("output");
    outputDiv.innerHTML = `<img src="data:image/png;base64,${imgData}" alt="Processed Image" />`;
}

// Function to display processed video (if it's a video)
function displayProcessedVideo(videoUrl) {
    const outputDiv = document.getElementById("output");
    
    // Create a video element dynamically
    const videoElement = document.createElement("video");
    videoElement.setAttribute("controls", "true");
    videoElement.setAttribute("width", "640");
    videoElement.setAttribute("height", "360");

    // Set the video source to the processed video URL
    videoElement.src = videoUrl;

    // Append the video element to the output container
    outputDiv.innerHTML = '';  // Clear previous content
    outputDiv.appendChild(videoElement);
}
