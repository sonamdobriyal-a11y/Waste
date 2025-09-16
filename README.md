Waste Volume Estimator (Webcam)

Overview
- Detects a plate or bowl in a live camera feed.
- Segments visible food/waste inside the utensil.
- Reports approximate percent fill (by area) and optional volume estimate (plates only, via simple height assumption).

Important notes
- This is a practical baseline. Single-camera true volume in bowls is ambiguous without depth or an empty-bowl baseline. For bowls, we report percent area coverage; volume requires extra calibration or depth.
- For plates, you can provide the plate diameter (mm) and an assumed average food height (mm) to get a rough volume estimate.
- Best results with an overhead camera and good lighting. A high-contrast utensil interior (e.g., white plate) helps segmentation.

Quick start
1) Create a Python 3.10+ venv and install deps:
   - `python -m venv .venv`
   - `./.venv/Scripts/activate` (Windows) or `source .venv/bin/activate` (macOS/Linux)
   - `pip install -r requirements.txt`

2) Run the app (plate example):
   - `python src/main.py --cam 0 --utensil plate --diameter-mm 260 --assumed-food-height-mm 15`

3) Bowl example (percent by area only):
   - `python src/main.py --cam 0 --utensil bowl`

How it works (baseline)
- Utensil detection: Finds circular/elliptical shapes via Canny + HoughCircles with a contour-based fallback.
- Food segmentation: Samples the utensil’s inner rim to estimate the base color, converts to Lab, thresholds the delta from base, then refines with GrabCut.
- Percent fill: Ratio of segmented food pixels to utensil interior pixels.
- Volume (plates): Uses plate diameter to infer scale and multiplies food area by an assumed height. This is a crude approximation suited for flat plates.

Calibration and tips
- Overhead view: Aim camera perpendicular to plate for diameter-based scaling to be accurate.
- Diameter: Measure inner usable diameter of the plate (in mm).
- Assumed height: Start with 10–20 mm; tune per cuisine/portion style.
- Lighting: Avoid strong glare; diffuse light improves segmentation.

Controls
- Press `q` to quit.
- Press `s` to toggle segmentation visualization.

CLI options
- `--cam`: Camera index (default 0).
- `--utensil`: `plate` or `bowl` (default: auto).
- `--diameter-mm`: Inner diameter of plate/bowl opening, in mm (for scale).
- `--assumed-food-height-mm`: Average food height (plates), in mm.
- `--min-radius` / `--max-radius`: Hough circle radius bounds in pixels.
- `--debug`: Extra prints and intermediate visualizations.

Limitations and next steps
- Bowls: For real volume, add a depth model (e.g., MiDaS) or capture an empty-bowl baseline to estimate depth-to-base.
- Non-circular utensils: Extend with a utensil detector (e.g., YOLOv8-seg custom model) and fit general contours.
- Robust food segmentation: Train a food-class segmentation model on your environment for better accuracy.

Project structure
- `src/main.py` – live webcam app.
- `src/vision.py` – utensil detection, segmentation, measurements.
- `requirements.txt` – dependencies.
- `README.md` – this file.

