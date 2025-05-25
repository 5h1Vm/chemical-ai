const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const roiBox = document.getElementById("roiBox");
const resultDiv = document.getElementById("result");
let videoStream;

navigator.mediaDevices
  .getUserMedia({ video: { facingMode: "environment" } })
  .then((stream) => {
    video.srcObject = stream;
    videoStream = stream;
    video.play();
    setTimeout(positionROIBox, 500);
  });

function positionROIBox() {
  const rect = video.getBoundingClientRect();
  const size = Math.min(rect.width, rect.height) * 0.25;
  roiBox.style.width = size + "px";
  roiBox.style.height = size + "px";
  roiBox.style.left = (rect.width - size) / 2 + "px";
  roiBox.style.top = (rect.height - size) / 2 + "px";
  roiBox.style.position = "absolute";
}

function captureImage() {
  video.pause();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);

  const size = Math.min(canvas.width, canvas.height) / 2;
  const x = (canvas.width - size) / 2;
  const y = (canvas.height - size) / 2;
  const roi = ctx.getImageData(x, y, size, size);

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = size;
  tempCanvas.height = size;
  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.putImageData(roi, 0, 0);

  const dataUrl = tempCanvas.toDataURL("image/jpeg");

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl }),
  })
    .then((res) => res.json())
    .then((data) => {
      resultDiv.innerHTML = `
        <h3>Detection Result</h3>
        <p><strong>Substance:</strong> ${data.substance}</p>
        <p><strong>Color HEX:</strong> ${data.hex}</p>
        <p><strong>Confidence:</strong> ${data.confidence}%</p>
        <p><strong>Description:</strong> ${data.info}</p>
        <p><strong>Toxicity Level:</strong> ${data.toxicity}</p>
      `;
    })
    .catch((err) => {
      resultDiv.innerHTML = "<p>❌ Error in prediction.</p>";
      console.error(err);
    });
}

function resetCamera() {
  video.play();
  resultDiv.innerHTML = "";
}

window.addEventListener("resize", positionROIBox);
