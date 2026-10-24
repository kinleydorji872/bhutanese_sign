const form = document.querySelector("#recognition-form");
const input = document.querySelector("#image");
const dropzone = document.querySelector("#dropzone");
const previewWrap = document.querySelector("#preview-wrap");
const preview = document.querySelector("#preview");
const fileName = document.querySelector("#file-name");
const removeButton = document.querySelector("#remove-image");
const submitButton = document.querySelector("#submit-button");
const title = document.querySelector("#result-title");
const message = document.querySelector("#result-message");
const detail = document.querySelector("#result-detail");
let previewUrl = null;
function showMessage(heading, body) { title.textContent = heading; message.textContent = body; detail.hidden = true; }
function setFile(file) {
  if (!file) return;
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) { input.value = ""; showMessage("Unsupported file", "Please choose a JPG, PNG, or WEBP image."); return; }
  if (file.size > 4 * 1024 * 1024) { input.value = ""; showMessage("Image is too large", "Choose an image smaller than 4 MB."); return; }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file); preview.src = previewUrl; fileName.textContent = file.name;
  previewWrap.hidden = false; dropzone.hidden = true; showMessage("Image ready", "Your image is ready to be analyzed.");
}
input.addEventListener("change", () => setFile(input.files[0]));
removeButton.addEventListener("click", () => {
  input.value = ""; previewWrap.hidden = true; dropzone.hidden = false; preview.removeAttribute("src");
  if (previewUrl) URL.revokeObjectURL(previewUrl); previewUrl = null; showMessage("Ready when you are", "Choose an image to begin.");
});
["dragenter", "dragover"].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.remove("dragover"); }));
dropzone.addEventListener("drop", event => {
  const file = event.dataTransfer.files[0]; if (!file) return;
  const transfer = new DataTransfer(); transfer.items.add(file); input.files = transfer.files; setFile(file);
});
form.addEventListener("submit", async event => {
  event.preventDefault(); const file = input.files[0];
  if (!file) { showMessage("Choose an image", "Select a hand-sign image first."); return; }
  submitButton.disabled = true; submitButton.querySelector("span:first-child").textContent = "Analyzing…";
  showMessage("Analyzing image", "Checking the image with the selected model.");
  const data = new FormData(); data.append("image", file); data.append("mode", document.querySelector("#mode").value);
  try {
    const response = await fetch("/api/predict", {method: "POST", body: data});
    const result = await response.json(); if (!response.ok) throw new Error(result.error || "Recognition could not be completed.");
    showMessage("Prediction complete", result.message || "Model prediction received."); detail.hidden = false; detail.replaceChildren();
    const label = document.createElement("strong"); label.textContent = "Sign: " + result.label;
    const confidence = document.createElement("span"); confidence.textContent = "Confidence " + (result.confidence * 100).toFixed(1) + "%";
    detail.append(label, confidence);
  } catch (error) { showMessage("Recognition unavailable", error.message); }
  finally { submitButton.disabled = false; submitButton.querySelector("span:first-child").textContent = "Recognize sign"; }
});