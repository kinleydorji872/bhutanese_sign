"""Web demo entrypoint for Vercel. The ML webcam app lives in scripts/webcam_app.py."""
from flask import Flask, render_template_string

app = Flask(__name__)

PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bhutanese Sign Language Recognition</title>
<style>
:root{color-scheme:dark;--bg:#0b1220;--panel:#131e31;--text:#eef4ff;--muted:#a8b7d1;--accent:#65d6c4}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(140deg,#0b1220,#142641);font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--text)}
main{max-width:900px;margin:auto;padding:32px 18px 60px}.tag{color:var(--accent);font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:.8rem}
h1{font-size:clamp(2rem,5vw,3.5rem);line-height:1.1;margin:.5rem 0 1rem}.muted{color:var(--muted)}
.panel{background:var(--panel);border:1px solid #2a3a55;border-radius:18px;padding:22px;margin:20px 0}
.actions{display:flex;flex-wrap:wrap;gap:12px}button,a.button{border:0;border-radius:10px;padding:12px 16px;font:inherit;font-weight:700;cursor:pointer;text-decoration:none}
button{background:var(--accent);color:#08201e}a.button{background:#263954;color:var(--text)}
video{width:100%;max-height:420px;background:#05080e;border-radius:12px;margin-top:14px;display:none}
.notice{border-left:4px solid #eabf66;padding:12px 14px;background:#2b281e;border-radius:8px}
input{display:block;margin:12px 0;max-width:100%}footer{margin-top:30px;color:var(--muted);font-size:.9rem}
</style>
</head>
<body><main>
<div class="tag">Prototype · Bhutan</div>
<h1>Bhutanese Sign Language Recognition</h1>
<p class="muted">A project exploring recognition of Bhutanese sign-language digits and alphabet signs.</p>
<div class="panel">
<h2>Try the camera preview</h2>
<p class="muted">This confirms browser camera access only. It does not yet recognise signs in this hosted version.</p>
<div class="actions"><button id="start">Start camera</button><button id="stop" disabled>Stop camera</button></div>
<video id="video" autoplay playsinline muted></video>
<p id="cameraStatus" class="muted" aria-live="polite">Camera is off.</p>
</div>
<div class="panel">
<h2>Recognition status</h2>
<div class="notice"><strong>Model not deployed yet.</strong><br>The repository currently does not contain trained model weights. Train and validate the model before showing predictions here.</div>
<p>For local webcam inference, follow the setup and training instructions in the GitHub README, then run <code>python scripts/webcam_app.py</code>.</p>
<div class="actions"><a class="button" href="https://github.com/kinleydorji872/bhutanese_sign" target="_blank" rel="noreferrer">View source on GitHub</a></div>
</div>
<div class="panel">
<h2>Project roadmap</h2>
<ol><li>Prepare correctly labelled Bhutanese Sign Language image data.</li><li>Train the digit model and evaluate it on held-out signers.</li><li>Collect a separately verified alphabet dataset before enabling alphabet recognition.</li><li>Integrate a trained model into a secure inference endpoint and test on phones and laptops.</li></ol>
</div>
<footer>Experimental educational prototype. Do not rely on it as an authoritative translation service.</footer>
</main>
<script>
const video=document.getElementById('video'),start=document.getElementById('start'),stop=document.getElementById('stop'),status=document.getElementById('cameraStatus');let stream=null;
start.addEventListener('click',async()=>{try{if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error('Camera access is not supported in this browser.');stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user'},audio:false});video.srcObject=stream;video.style.display='block';start.disabled=true;stop.disabled=false;status.textContent='Camera preview is running. Recognition is not enabled yet.';}catch(e){status.textContent='Could not start camera: '+e.message+' Check browser permissions and use HTTPS.';}});
stop.addEventListener('click',()=>{if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;video.style.display='none';start.disabled=false;stop.disabled=true;status.textContent='Camera is off.';});
</script></body></html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(PAGE)

@app.route("/api/health", methods=["GET"])
def health():
    return {"status": "ok", "recognition_model": "not_deployed"}, 200
