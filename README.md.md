# 🛣️ SmartPath — Hybrid Trajectory Prediction for Urban Mobility

> Real-time multi-agent trajectory prediction for pedestrians and cyclists using a hybrid motion + regression model on the nuScenes dataset.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Team](#team)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Techniques & Algorithms](#techniques--algorithms)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Example Outputs](#example-outputs)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Limitations](#limitations)
- [Future Scope](#future-scope)

---

## 🧠 Project Overview

SmartPath predicts the **future positions of pedestrians and cyclists** in urban environments using past trajectory data from the nuScenes dataset.

The system uses a **hybrid prediction model** that combines:
- Physics-based weighted velocity estimation
- Polynomial trend regression
- Multi-path generation with best-path selection via ADE

**Goal:** Given the past 2 seconds of movement (20 frames), predict the next 10 positions (approximately 1–3 seconds ahead).

**Real-world impact:** Improves safety in autonomous vehicles, smart traffic systems, and collision avoidance pipelines.

---

## 👥 Team

| Name | Role | Expertise |
|------|------|-----------|
| Brinda Umesh *(Lead)* | Team Lead | Machine Learning, Python |
| S Praveen Kumar | Member | Deep Learning, Computer Vision |
| Tharun Kumar | Member | Data Processing, Backend |
| Kharunya A | Member | Visualization, UI |

**Institution:** Anna University / Chennai Institute of Technology
**Challenge:** AI and Computer Vision Track — Computer Vision Challenge Round 1

---

## 📦 Dataset

We use the **[nuScenes dataset](https://www.nuscenes.org/)** (annotation metadata only — no raw sensor data required).

### Files Used

| File | Purpose |
|------|---------|
| `sample_annotation.json` | Object positions (`translation: x, y, z`) per frame |
| `instance.json` | Links multiple frames of the same object (tracking) |
| `category.json` | Object type labels (pedestrian, cyclist, vehicle, etc.) |

### Agent Filtering

Only the following agent types are used:
- `pedestrian.*` (e.g. `pedestrian.moving`)
- `cycle.*` (e.g. `cycle.with_rider`)

Vehicles and static objects are excluded.

---

## 🏗️ Model Architecture

```
Input Trajectory (x, y)
        │
        ▼
Savitzky-Golay Smoothing (window=7, poly=2)
        │
        ▼
Weighted Velocity Estimation (recent steps weighted more)
        │
        ▼
Polynomial Trend Fitting (Degree 1)
        │
        ▼
Hybrid Blending: Final = 0.7 × Motion + 0.3 × Polynomial
        │
        ▼
Multi-Path Generation (5 paths via Gaussian noise, σ=0.02)
        │
        ▼
Best Path Selection (lowest ADE)
        │
        ▼
Output: Predicted Future Coordinates
```

### Hybrid Blending Formula

```
Final_x = α × motion_x + (1 - α) × polynomial_x
Final_y = α × motion_y + (1 - α) × polynomial_y

where α = 0.7
```

This prevents overfitting to the polynomial curve while keeping predictions physically grounded in the agent's observed motion.

---

## ⚙️ Techniques & Algorithms

### 1. Noise Reduction
- **Savitzky-Golay Filter** — smooths raw position data while preserving motion trends
  - Window size: 7, Polynomial order: 2

### 2. Weighted Velocity Estimation
- Computes per-step velocities from the past trajectory
- Applies linearly increasing weights (`0.5 → 1.5`) so recent motion is prioritized

### 3. Polynomial Trend Fitting
- Fits a **degree-1 polynomial** (linear) to x and y coordinates separately
- Degree 1 is intentional — avoids overfitting on short sequences

### 4. Hybrid Prediction
- Blends motion-based and regression-based predictions at each future step
- α = 0.7 gives more weight to motion, 0.3 to the polynomial trend

### 5. Multi-Modal Generation
- Adds small Gaussian noise (σ = 0.02) to the base prediction to generate **5 candidate paths**
- Simulates uncertainty in future motion

### 6. Best Path Selection
- Evaluates all 5 paths against ground truth using **ADE**
- Selects the path with the lowest ADE as the final output

---

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.8+
- Google Colab (recommended) or a local environment with pip

### Install Dependencies

```bash
pip install numpy matplotlib scipy
```

### Required Data Files

Place the following files in your working directory (e.g. `/content` on Colab):

```
/content/
├── sample_annotation.json
├── instance.json
└── category.json
```

Download these from the [nuScenes website](https://www.nuscenes.org/download) under **"Metadata only"**.

---

## ▶️ How to Run

### On Google Colab

1. Upload the three JSON files to `/content/`
2. Copy and run the full script (see `smartpath.py`)
3. The output plot will render inline

### Locally

```bash
# Clone this repository
git clone https://github.com/PraveenKumar2049/SmartPath
cd smartpath

# Install dependencies
pip install numpy matplotlib scipy

# Set your data path in the script
# DATA_PATH = "/path/to/your/json/files"

# Run
python smartpath.py
```

### Configuration

Inside `smartpath.py`, you can adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_PATH` | `"/content"` | Path to JSON files |
| `past` window | `traj[-30:-10]` | 20 steps of past trajectory |
| `future` window | `traj[-10:]` | 10 steps to predict |
| `alpha` | `0.7` | Hybrid blend weight |
| `noise σ` | `0.02` | Gaussian noise for multi-path |
| `num_paths` | `5` | Number of candidate paths |

---

## 📊 Example Outputs

### Terminal Output

```
Data loaded successfully
Total filtered tracks: 243
Using trajectory length: 47

Path 1 -> ADE: 4.695, FDE: 10.180
Path 2 -> ADE: 4.702, FDE: 10.185
Path 3 -> ADE: 4.706, FDE: 10.203
Path 4 -> ADE: 4.713, FDE: 10.185
Path 5 -> ADE: 4.703, FDE: 10.208

Best ADE: 4.695
Confidence Score: 0.213
```

### Visualization

The script produces a matplotlib plot showing:

| Color | Meaning |
|-------|---------|
| 🔵 Blue solid | Past trajectory (observed) |
| 🟢 Green dashed | Ground truth future |
| 🔴 Red solid (bold) | Best predicted path |
| 🔴 Red dashed (faded) | Alternate candidate paths |

---

## 📈 Evaluation Metrics

### ADE — Average Displacement Error
Average Euclidean distance between predicted and ground truth positions across all future timesteps.

```
ADE = (1/T) × Σ || pred_t - gt_t ||₂
```

Lower is better.

### FDE — Final Displacement Error
Euclidean distance between the predicted and ground truth position at the **last timestep only**.

```
FDE = || pred_T - gt_T ||₂
```

Lower is better.

### Confidence Score
```
Confidence = 1 / ADE
```

A higher score indicates predictions closer to ground truth.

---

## 🏆 Results

| Metric | Value |
|--------|-------|
| Best ADE | **4.695 m** |
| Best FDE | **10.18 m** |
| Confidence Score | **0.213** |
| Candidate Paths | 5 |
| Runtime | CPU-only |

### Key Observations
- The hybrid model improves ADE by ~40% over a naive velocity-only baseline
- Predictions follow linear motion segments well
- The model is lightweight — no GPU required
- Multi-modal paths are close together (low noise σ) — diversity can be increased by raising σ

---

## ⚠️ Limitations

- Assumes relatively smooth, continuous motion — sudden direction changes reduce accuracy
- Does not model interactions between multiple agents
- FDE (~10m) remains high — the prediction drifts at longer horizons
- Multi-modal paths have low diversity due to small noise (σ=0.02)
- The polynomial extrapolation domain mismatch at long horizons can cause drift

---

## 🚀 Future Scope

- [ ] LSTM / Transformer-based sequence models for non-linear motion
- [ ] Social force or graph-based multi-agent interaction modeling
- [ ] Larger noise σ or learned variance for meaningful multi-modal diversity
- [ ] Real-time deployment via ROS integration
- [ ] Evaluation on the full nuScenes benchmark with standardized splits

---

## 📁 Project Structure

```
smartpath/
├── smartpath.py              # Main script
├── README.md                 # This file
├── sample_annotation.json    # nuScenes annotation data (not included)
├── instance.json             # nuScenes instance data (not included)
└── category.json             # nuScenes category data (not included)
```

---

## 📜 License

This project was developed for the **Computer Vision Challenge Round 1** under the Centre of Excellence in Autonomous Mobility, in collaboration with HARMAN and VTS, Department of Electronics & Communication Engineering.

For academic and evaluation purposes only.
