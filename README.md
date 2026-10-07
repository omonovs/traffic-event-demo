# NOWL — Traffic Event Detection & Accident Anticipation

A computer-vision project for detecting risky pedestrian and traffic events from video.  
The system combines **YOLO11n**, **ByteTrack**, rule-based event logic, and a **Streamlit** web demo.

**Live demo:** https://traffic-event-demo-nowl.streamlit.app/

## Overview

The pipeline detects and tracks road users, then analyzes their movement to identify safety-related events.

Supported event classes include:

- `stopped_vehicle`
- `wrong_way`
- `jaywalking`
- `congestion`
- `failure_to_yield`
- `collision_risk`

A `RiskEstimator` module uses **time-to-collision (TTC)** logic to estimate potential collision risk.

## Tech Stack

- Python
- YOLO11n / Ultralytics
- ByteTrack
- Streamlit
- OpenCV
- Rule-based event detection
- EDA and visualization

## Project Structure

```text
traffic-event-demo/
├── app.py
├── run_submission.py
├── evaluate.py
├── precompute_eda.py
├── precompute_results.py
├── solution.py
├── src/
├── config/
├── assets/
├── samples/
├── weights/
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

Run the submission pipeline:

```bash
python run_submission.py --videos /data/test --out predictions.json
```

YOLO weights are stored in:

```text
weights/yolo11n.pt
```

## Approach

The solution uses a **COCO-pretrained YOLO11n model** for object detection and **ByteTrack** for multi-object tracking.

Event detection is implemented using rule-based logic in `src/events/`. The project does **not** use fine-tuning; the event-classification layer is based on tracked motion patterns and handcrafted rules.

The public Streamlit application presents:

- project overview
- team and roles
- approach
- exploratory data analysis (EDA)
- live demo
- results
- report

## My Contribution

**Sardor Omonov — Full-Stack Web & Demo Engineer**

- Developed the Streamlit website and public demo
- Built the project pages and overall UI structure
- Integrated EDA visualizations, project results, and demo content
- Helped package the technical work into a clear, recruiter- and reviewer-friendly interface

## Team

- **Anvar Mexmonov** — AI & Pipeline Engineer  
  Model integration, event rules, RiskEstimator
- **Sirojiddin Abduraxmonov** — Data, Evaluation & Quality Engineer  
  Development set, evaluation pipeline, quality control
- **Sardor Omonov** — Full-Stack Web & Demo Engineer  
  Website, EDA presentation, Live Demo, Results

## Datasets & Licenses

- COCO — pretrained YOLO11n weights
- Ultralytics — AGPL-3.0
- ByteTrack — MIT

## Reproducibility

The inference pipeline is deterministic for the provided setup; YOLO inference uses seed `0`, with no additional randomness introduced by the project logic.

---

Built as a computer-vision hackathon project by team **NOWL**.
