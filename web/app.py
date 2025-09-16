import base64
import os
import sys
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

# Ensure project root is importable (so we can import src.*)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vision import detect_utensil_ellipse, segment_food_in_utensil, estimate_area_and_volume


app = Flask(__name__, static_folder='static', template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')


def _decode_image_from_base64(data_url: str):
    # Accepts either data URL (data:image/jpeg;base64,...) or raw base64
    if ',' in data_url and data_url.strip().startswith('data:'):
        b64 = data_url.split(',', 1)[1]
    else:
        b64 = data_url
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return None
    nparr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def _encode_image_to_data_url(img_bgr):
    ok, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    b64 = base64.b64encode(buf).decode('ascii')
    return f"data:image/jpeg;base64,{b64}"


@app.route('/process', methods=['POST'])
def process():
    payload = request.get_json(silent=True) or {}
    img_data = payload.get('image')
    utensil = payload.get('utensil', 'auto')
    diameter_mm = payload.get('diameter_mm')
    height_mm = payload.get('assumed_height_mm', 15.0)

    if img_data is None:
        return jsonify({'error': 'missing image'}), 400

    frame = _decode_image_from_base64(img_data)
    if frame is None:
        return jsonify({'error': 'bad image'}), 400

    display = frame.copy()

    # Sensible Hough bounds based on frame size
    h, w = frame.shape[:2]
    min_radius = max(30, min(h, w) // 8)
    max_radius = max(60, min(h, w) // 2)

    ellipse = detect_utensil_ellipse(frame, utensil_hint=utensil,
                                     min_radius=min_radius, max_radius=max_radius, debug=False)

    percent_fill = None
    est_volume_ml = None

    if ellipse is not None:
        # Draw ellipse
        (cx, cy), (MA, ma), angle = ellipse
        cv2.ellipse(display, (int(cx), int(cy)), (int(MA/2), int(ma/2)), angle, 0, 360, (0,255,255), 2)

        seg_mask, _ = segment_food_in_utensil(frame, ellipse, debug=False)
        if seg_mask is not None:
            # Overlay segmentation
            overlay = display.copy()
            overlay[seg_mask > 0] = (0, 0, 255)
            display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

            # Compute metrics
            d_mm = float(diameter_mm) if diameter_mm not in (None, '') else None
            percent_fill, est_volume_ml = estimate_area_and_volume(
                ellipse, seg_mask, d_mm, utensil, float(height_mm or 0)
            )

    # HUD text
    y = 24
    txts = []
    txts.append(f"Utensil: {utensil}")
    if percent_fill is not None:
        txts.append(f"Fill: {percent_fill:.1f}%")
    if est_volume_ml is not None:
        txts.append(f"Vol: {est_volume_ml:.0f} ml")
    if ellipse is None:
        txts.append("Utensil not detected")
    for i, t in enumerate(txts):
        cv2.putText(display, t, (10, 30 + i*y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    data_url = _encode_image_to_data_url(display)
    return jsonify({
        'percent_fill': percent_fill,
        'volume_ml': est_volume_ml,
        'overlay': data_url,
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8000'))
    app.run(host='0.0.0.0', port=port, debug=True)
