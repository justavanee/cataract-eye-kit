let photoTaken = false;

function takePhoto() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function (stream) {
                const video = document.createElement('video');
                video.srcObject = stream;
                video.play();
                
                setTimeout(() => {
                    const canvas = document.getElementById('canvas');
                    const context = canvas.getContext('2d');
                    context.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const dataUrl = canvas.toDataURL('image/png');
                    document.getElementById('photo').src = dataUrl;
                    photoTaken = true;
                    video.pause();
                    stream.getTracks().forEach(track => track.stop());
                }, 1000);
            })
            .catch(err => console.log(err));
    }
}

function sendPhoto() {
    if (photoTaken) {
        const canvas = document.getElementById('canvas');
        const dataUrl = canvas.toDataURL('image/png');

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: dataUrl.split(',')[1] }),
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('result').textContent = `Prediction: ${data.result}`;
        })
        .catch(error => console.error('Error:', error));
    } else {
        alert('Please take a photo first.');
    }
}