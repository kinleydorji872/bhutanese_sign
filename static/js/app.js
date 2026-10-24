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
  if(!navigator.mediaDevices?.getUserMedia){cameraHelp.textContent="Camera access is unavailable in this browser. Open the site using HTTPS or choose an image instead.";return}
  try{
    cameraHelp.textContent="Waiting for camera permission…";
    cameraStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:"environment"},width:{ideal:1280},height:{ideal:720}},audio:false});
    video.srcObject=cameraStream;video.hidden=false;cameraPlaceholder.hidden=true;
    await video.play();startCameraButton.disabled=true;captureButton.disabled=false;stopCameraButton.disabled=false;
    cameraHelp.textContent="Camera is on. Position one hand clearly in the frame, then capture.";
  }catch(error){
    const reason=error.name==="NotAllowedError"?"Camera permission was denied. Allow camera access in your browser settings and try again.":error.name==="NotFoundError"?"No camera was found. Connect a camera or upload an image.":"Could not start the camera. Check that another app is not using it, then try again.";
    cameraHelp.textContent=reason;
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