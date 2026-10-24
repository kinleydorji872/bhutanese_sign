const form=document.querySelector("#recognition-form");
const input=document.querySelector("#image");
const dropzone=document.querySelector("#dropzone");
const previewWrap=document.querySelector("#preview-wrap");
const preview=document.querySelector("#preview");
const fileName=document.querySelector("#file-name");
const removeButton=document.querySelector("#remove-image");
const submitButton=document.querySelector("#submit-button");
const title=document.querySelector("#result-title");
const message=document.querySelector("#result-message");
const detail=document.querySelector("#result-detail");
const video=document.querySelector("#camera-video");
const cameraPlaceholder=document.querySelector("#camera-placeholder");
const startCameraButton=document.querySelector("#start-camera");
const captureButton=document.querySelector("#capture-photo");
const stopCameraButton=document.querySelector("#stop-camera");
const cameraHelp=document.querySelector("#camera-help");
const canvas=document.querySelector("#capture-canvas");
let cameraStream=null, previewUrl=null;

function showMessage(heading,body){title.textContent=heading;message.textContent=body;detail.hidden=true}
function setFile(file){
  if(!file)return;
  if(!["image/jpeg","image/png","image/webp"].includes(file.type)){input.value="";showMessage("Unsupported file","Please choose a JPG, PNG, or WEBP image.");return}
  if(file.size>4*1024*1024){input.value="";showMessage("Image is too large","Choose an image smaller than 4 MB.");return}
  if(previewUrl)URL.revokeObjectURL(previewUrl);
  previewUrl=URL.createObjectURL(file);preview.src=previewUrl;fileName.textContent=file.name;
  previewWrap.hidden=false;dropzone.hidden=true;showMessage("Image ready","Your image is ready to be analyzed.");
}
input.addEventListener("change",()=>setFile(input.files[0]));
removeButton.addEventListener("click",()=>{
  input.value="";previewWrap.hidden=true;dropzone.hidden=false;preview.removeAttribute("src");
  if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=null;
  showMessage("Ready when you are","Capture a photo or choose an image to begin.");
});
["dragenter","dragover"].forEach(name=>dropzone.addEventListener(name,event=>{event.preventDefault();dropzone.classList.add("dragover")}));
["dragleave","drop"].forEach(name=>dropzone.addEventListener(name,event=>{event.preventDefault();dropzone.classList.remove("dragover")}));
dropzone.addEventListener("drop",event=>{
  const file=event.dataTransfer.files[0];if(!file)return;
  const transfer=new DataTransfer();transfer.items.add(file);input.files=transfer.files;setFile(file);
});
async function startCamera(){
  if(!window.isSecureContext){
    cameraHelp.textContent="Camera requires a secure HTTPS page. Open the official https:// website, not an http:// link.";
    return;
  }
  if(!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia!=="function"){
    cameraHelp.textContent="This browser does not expose camera access. Try the latest Chrome or Edge, or upload an image.";
    return;
  }
  startCameraButton.disabled=true;
  cameraHelp.textContent="Requesting camera permission. If your browser asks, choose Allow.";
  try{
    let stream;
    try{
      stream=await navigator.mediaDevices.getUserMedia({
        video:{facingMode:{ideal:"environment"},width:{ideal:1280},height:{ideal:720}},
        audio:false
      });
    }catch(firstError){
      // Some laptop webcams do not support preferred camera constraints; retry with defaults.
      if(firstError.name==="NotFoundError" || firstError.name==="NotAllowedError" || firstError.name==="SecurityError") throw firstError;
      stream=await navigator.mediaDevices.getUserMedia({video:true,audio:false});
    }
    cameraStream=stream;
    video.srcObject=stream;
    video.hidden=false;
    cameraPlaceholder.hidden=true;
    try{await video.play();}catch(playError){
      // A stream can be granted but playback may need a second user gesture.
      cameraHelp.textContent="Camera permission granted. Press Start camera again if the preview does not appear.";
      throw playError;
    }
    captureButton.disabled=false;
    stopCameraButton.disabled=false;
    cameraHelp.textContent="Camera is on. Position one hand clearly in the frame, then press Capture photo.";
  }catch(error){
    if(cameraStream){cameraStream.getTracks().forEach(track=>track.stop());cameraStream=null;}
    video.srcObject=null;video.hidden=true;cameraPlaceholder.hidden=false;
    startCameraButton.disabled=false;
    const errors={
      NotAllowedError:"Camera permission is blocked. In Edge/Chrome, select the lock or site-controls icon beside the address bar, set Camera to Allow, then reload this page.",
      PermissionDeniedError:"Camera permission is blocked. Allow camera access in the browser's site settings, then reload.",
      NotFoundError:"No camera was found. Connect or enable a webcam, or use Upload an image.",
      NotReadableError:"The camera is busy or unavailable. Close Camera, Teams, Zoom, or other apps using it, then try again.",
      OverconstrainedError:"The camera does not support the requested settings. Check that the webcam is enabled and try again.",
      SecurityError:"The browser blocked camera access for security reasons. Open this site over HTTPS and check site permissions."
    };
    cameraHelp.textContent=errors[error.name]||("Could not start the camera ("+(error.name||"unknown error")+"). Check browser camera permission and whether another app is using the webcam, then try again.");
  }
}
function stopCamera(){
  if(cameraStream){cameraStream.getTracks().forEach(track=>track.stop());cameraStream=null}
  video.srcObject=null;video.hidden=true;cameraPlaceholder.hidden=false;
  startCameraButton.disabled=false;captureButton.disabled=true;stopCameraButton.disabled=true;
}
startCameraButton.addEventListener("click",startCamera);
stopCameraButton.addEventListener("click",stopCamera);
captureButton.addEventListener("click",()=>{
  if(!cameraStream||!video.videoWidth){cameraHelp.textContent="Camera is not ready yet. Wait a moment and try again.";return}
  canvas.width=video.videoWidth;canvas.height=video.videoHeight;
  const context=canvas.getContext("2d");context.drawImage(video,0,0,canvas.width,canvas.height);
  canvas.toBlob(blob=>{
    if(!blob){cameraHelp.textContent="Could not capture the frame. Please try again.";return}
    const file=new File([blob],"camera-capture.jpg",{type:"image/jpeg"});
    const transfer=new DataTransfer();transfer.items.add(file);input.files=transfer.files;setFile(file);
    cameraHelp.textContent="Photo captured. You can recognize it below or capture another frame.";
  },"image/jpeg",0.92);
});
form.addEventListener("submit",async event=>{
  event.preventDefault();const file=input.files[0];
  if(!file){showMessage("Capture or choose an image","Start the camera and capture a frame, or upload an image.");return}
  submitButton.disabled=true;submitButton.querySelector("span:first-child").textContent="Analyzing…";
  showMessage("Analyzing image","Sending the captured image to the recognition API.");
  const data=new FormData();data.append("image",file);data.append("mode",document.querySelector("#mode").value);
  try{
    const response=await fetch("/api/predict",{method:"POST",body:data});const result=await response.json();
    if(!response.ok)throw new Error(result.error||"Recognition could not be completed.");
    showMessage("Prediction complete",result.message||"Model prediction received.");detail.hidden=false;detail.replaceChildren();
    const label=document.createElement("strong");label.textContent="Sign: "+result.label;
    const confidence=document.createElement("span");confidence.textContent="Confidence "+(result.confidence*100).toFixed(1)+"%";
    detail.append(label,confidence);
  }catch(error){showMessage("Recognition unavailable",error.message)}
  finally{submitButton.disabled=false;submitButton.querySelector("span:first-child").textContent="Recognize sign"}
});
window.addEventListener("pagehide",stopCamera);