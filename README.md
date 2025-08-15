
# Drone Telemetry Anomaly Detection Benchmark (Synthetic Dataset)

## Overview
This repository contains a **synthetic** benchmark dataset and example implementations of multiple anomaly detection models for drone telemetry data.  
It is designed for the **Terrier Competitions 2025** and aims to provide a reproducible testing ground for comparing different anomaly detection approaches on large-scale, time-series flight data.

### Dataset Description
The dataset simulates drone telemetry at **10 Hz** sampling rate over **8.33 hours** (300,000 rows).  
Each row contains:

| Column           | Unit     | Description                               |
|------------------|----------|-------------------------------------------|
| altitude_m       | meters   | Altitude above ground                     |
| velocity_m_s     | m/s      | Horizontal velocity                       |
| yaw_deg           | degrees | Heading/yaw orientation                   |
| pitch_deg         | degrees | Pitch angle                               |
| battery_pct       | %       | Remaining battery percentage              |
| gps_drift_m       | meters  | GPS position drift                        |
| anomaly_label     | 0/1     | 1 = anomaly present, 0 = normal           |

#### Anomaly Types Injected
Seven types of anomalies are injected (~0.8% of rows total, ≈ 2,400 anomalies), with randomized timing and duration:
1. Altitude spikes/drops
2. Velocity surges
3. Yaw angle jumps
4. Pitch instability
5. Sudden battery drops
6. GPS drift bursts
7. Combined sensor failure patterns

---

## Models Included
The repository provides ready-to-run implementations for:
- **Anomaly Transformer**  
- **MAD-GAN**  
- (Optional) LSTM-VAE & USAD — can be added for extended benchmarks.

Each model:
- Trains **only on normal sequences** (unsupervised setting).
- Uses fixed-length sliding windows (`window_size = 50`).
- Outputs anomaly scores and classification metrics.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/iiioooiso/drone-anomaly-detection.git
cd drone-anomaly-detection

# Install dependencies
pip install torch torch-geometric einops pandas numpy scikit-learn matplotlib tqdm

# Run training and evaluation
python T_25.py
