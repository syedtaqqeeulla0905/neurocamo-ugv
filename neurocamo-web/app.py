"""
NeuroCamo Web — local Flask server
==================================
Serves the single-page UI from ./static and exposes the SNN as a JSON API.

Bound to 127.0.0.1 by default, so it is reachable only from this machine. There
is no authentication on /api/detect — if you change HOST to 0.0.0.0 to reach it
from another device, anyone on that network can post images to the model.

    python app.py            ->  http://127.0.0.1:8080
"""

from __future__ import annotations

import base64
import os
import traceback

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from inference import ModelBundle, run_inference

HOST = os.environ.get("NEUROCAMO_HOST", "127.0.0.1")
PORT = int(os.environ.get("NEUROCAMO_PORT", "8080"))
MAX_UPLOAD_MB = 25

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

print("Loading NeuroCamo SNN ...")
BUNDLE = ModelBundle()
print(f"Ready on {BUNDLE.info.accelerator} ({BUNDLE.info.device})")


@app.after_request
def no_store(resp):
    # Dev server: never cache the shell or the results.
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status")
def status():
    i = BUNDLE.info
    return jsonify(
        {
            "ready": True,
            "device": i.device,
            "accelerator": i.accelerator,
            "epoch": i.epoch,
            "val_loss": round(i.val_loss, 4),
            "checkpoint_loaded": i.checkpoint_loaded,
            "checkpoint": os.path.basename(i.checkpoint_path),
            "params": i.params,
            "timesteps": i.timesteps,
            "defaulted_keys": i.defaulted_keys or [],
            "unexpected_keys": i.unexpected_keys or [],
        }
    )


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return True if obj else False
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_json(obj.tolist())
    return obj


@app.route("/api/detect", methods=["GET", "POST"])
def detect():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify({"error": "No image supplied under the 'image' field."}), 400

    ext = os.path.splitext(upload.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify(
            {"error": f"Unsupported file type '{ext}'. Use JPG, PNG, BMP or WEBP."}
        ), 415

    raw = np.frombuffer(upload.read(), dtype=np.uint8)
    if raw.size == 0:
        return jsonify({"error": "Empty upload."}), 400

    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"error": "Could not decode that file as an image."}), 400

    # Very large photos cost time in the Laplacian pass without changing the
    # verdict (the SNN sees a 128x128 crop either way), so cap the long edge.
    h, w = bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge > 1600:
        s = 1600 / long_edge
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    try:
        out = run_inference(BUNDLE, bgr)
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Inference failed. See the server log."}), 500

    annotated = out.pop("annotated_rgb")
    ok, buf = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    if not ok:
        return jsonify({"error": "Could not encode the result image."}), 500

    out["result_png"] = "data:image/png;base64," + base64.b64encode(buf).decode("ascii")
    out["filename"] = os.path.basename(upload.filename)
    return jsonify(sanitize_json(out))

import urllib.request


@app.route("/api/detect_stream", methods=["GET", "POST"])
def detect_stream():
    stream_url = ""
    if request.is_json and request.get_json(silent=True):
        stream_url = request.get_json(silent=True).get("stream_url", "")
    if not stream_url:
        stream_url = request.values.get("stream_url", "")
    stream_url = stream_url.strip()

    if not stream_url:
        return jsonify({"error": "No IP stream URL provided."}), 400

    # Auto-fix common IP Webcam app typos (if user types http://192.168.1.5:8080 without path, auto-append /shot.jpg)
    if stream_url.startswith("http") and not any(stream_url.endswith(ext) for ext in [".jpg", ".png", "/shot.jpg", "/video", "/photo.jpg", "/mempkg", "/feed"]):
        if not stream_url.endswith("/"):
            stream_url += "/"
        stream_url += "shot.jpg"

    bgr = None

    # 1. For static JPEG snapshots (/shot.jpg, /photo.jpg), use fast HTTP GET
    is_snapshot = any(stream_url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", "/shot.jpg", "/photo.jpg"])

    if is_snapshot and (stream_url.startswith("http://") or stream_url.startswith("https://")):
        try:
            req = urllib.request.Request(
                stream_url,
                headers={"User-Agent": "Mozilla/5.0 (NeuroCamo Web)"}
            )
            with urllib.request.urlopen(req, timeout=2.5) as response:
                raw = np.frombuffer(response.read(), dtype=np.uint8)
                if raw.size > 0:
                    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        except Exception:
            bgr = None

    # 2. Fall back to cv2.VideoCapture for MJPEG streams (/video), local webcam index, or snapshot fallback
    if bgr is None:
        try:
            src = int(stream_url) if stream_url.isdigit() else stream_url
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    bgr = frame
            cap.release()
        except Exception:
            bgr = None

    if bgr is None:
        return jsonify({
            "error": f"Could not fetch frame from '{stream_url}'. Make sure IP Webcam app is running on your phone, both phone & PC are on the same Wi-Fi network, and the IP is correct."
        }), 502

    # Cap long edge for fast real-time throughput
    h, w = bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge > 1200:
        s = 1200 / long_edge
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    # Encode input image as JPEG base64 for live display
    ok_in, buf_in = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    input_b64 = "data:image/jpeg;base64," + base64.b64encode(buf_in).decode("ascii") if ok_in else ""

    try:
        out = run_inference(BUNDLE, bgr)
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Inference failed. See server log."}), 500

    annotated = out.pop("annotated_rgb")
    ok_out, buf_out = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok_out:
        return jsonify({"error": "Could not encode result image."}), 500

    out["result_png"] = "data:image/jpeg;base64," + base64.b64encode(buf_out).decode("ascii")
    out["input_png"]  = input_b64
    out["filename"]   = "Phone IP Webcam"
    return jsonify(sanitize_json(out))


from flask import Response
import time

@app.route("/api/video_feed")
def video_feed():
    stream_url = request.args.get("url", "").strip()
    if not stream_url:
        return "No stream URL provided", 400

    url = stream_url
    if url.startswith("http") and not any(url.lower().endswith(ext) for ext in [".jpg", ".png", "/shot.jpg", "/video", "/photo.jpg"]):
        if not url.endswith("/"):
            url += "/"
        url += "video"

    # Fast connection check to detect refused connections instantly
    if url.startswith("http://") or url.startswith("https://"):
        conn_ok = False
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                conn_ok = True
        except Exception:
            pass

        if not conn_ok:
            cap_test = cv2.VideoCapture(url)
            conn_ok = cap_test.isOpened()
            cap_test.release()

        if not conn_ok:
            return jsonify({"error": f"Could not connect to camera at '{url}'. Make sure IP Webcam app is open on your phone, 'Start server' is tapped, and both phone & PC are on the same Wi-Fi network."}), 502

    def generate_frames():
        cap = cv2.VideoCapture(url)
        if not cap.isOpened() and url.isdigit():
            cap = cv2.VideoCapture(int(url))

        if cap.isOpened():
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        time.sleep(0.05)
                        continue
                    
                    h, w = frame.shape[:2]
                    if max(h, w) > 800:
                        s = 800 / max(h, w)
                        frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                    
                    ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if not ok:
                        continue
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    time.sleep(0.03) # ~30 FPS
            finally:
                cap.release()
        else:
            # Snapshot fallback loop for /shot.jpg endpoints
            while True:
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=1.5) as resp:
                        raw = np.frombuffer(resp.read(), dtype=np.uint8)
                        if raw.size > 0:
                            bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                            if bgr is not None:
                                ok, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                                if ok:
                                    yield (b'--frame\r\n'
                                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                except Exception:
                    pass
                time.sleep(0.1)

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File exceeds the {MAX_UPLOAD_MB} MB limit."}), 413


if __name__ == "__main__":
    print(f"\n  NeuroCamo Web  ->  http://{HOST}:{PORT}\n")
    # debug=False: the reloader would load the SNN twice.
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
