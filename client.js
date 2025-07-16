const video = document.getElementById('video');  
const canvas = document.getElementById('canvas'); // Get the <canvas> element to draw frames and keypoints
const ctx = canvas.getContext('2d'); // Used to draw images, shapes, lines, or text on the canvas.
const socket = io('http://localhost:5000');  // WebSocket connection for sending-receiving messages between client and server in real time.

// Start webcam stream
navigator.mediaDevices.getUserMedia({ video: true })  // Request webcam video stream permission from user
  .then(stream => {   
    video.srcObject = stream;  // Set webcam stream as source for video element
    video.play(); // Play the video element (starts webcam preview)
    startSendingFrames();  
  })
  .catch(err => {
    console.error("Error accessing webcam: ", err);  
  });

function startSendingFrames() {
  const fps = 5;  
  const interval = 1000 / fps;  // Calculate interval in ms between frames (1000/5ms for 5fps)
  setInterval(() => {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height); // Draw current video frame onto the canvas
    canvas.toBlob(blob => { // Convert canvas image to a JPEG Blob asynchronously
      const reader = new FileReader(); // Create FileReader to read Blob as base64 data URL
      reader.onloadend = () => {
        const base64Data = reader.result.split(',')[1];  
        socket.emit('frame', base64Data); // Send base64-encoded frame data to server over Socket.IO
      };
      reader.readAsDataURL(blob);  
    }, 'image/jpeg', 0.7); // Encode canvas as JPEG at 70% quality
  }, interval);  
}