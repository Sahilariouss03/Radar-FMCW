# Automotive FMCW SAR Simulation System

A production-grade, full-stack automotive FMCW (Frequency Modulated Continuous Wave) SAR (Synthetic Aperture Radar) simulation system with complete signal processing pipeline and interactive visualization.

---

## Prerequisites

- **Python 3.9+** — [Download](https://www.python.org/downloads/)
- **Node.js 18+** — [Download](https://nodejs.org/)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Radar-Simulation.git
cd Radar-Simulation
```

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Run the Application

**Option A — One-click startup:**

| OS | Command |
|---|---|
| Windows | `start.bat` |
| Linux/Mac | `chmod +x start.sh && ./start.sh` |

**Option B — Manual (two terminals):**

Terminal 1 (Backend):
```bash
cd backend
python main.py
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

### 5. Open in Browser

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

---

## Usage

1. Adjust radar parameters using the sliders (frequency, bandwidth, velocity, etc.)
2. Click **Run Simulation**
3. Explore results across four tabs:
   - **Range-Doppler Maps** — Before/after interference cancellation
   - **Ego-Motion** — WM, OS, DCM algorithm comparison
   - **SAR Images** — Conventional vs interpolated RDA
   - **Quality Metrics** — Resolution, PSR, position error

---

## Project Structure

```
Radar-Simulation/
├── backend/
│   ├── radar/
│   │   ├── fmcw.py              # FMCW signal simulation
│   │   ├── range_doppler.py     # Range-Doppler processing
│   │   ├── interference.py      # Self-interference cancellation
│   │   ├── ego_motion.py        # Ego-motion estimation (WM, OS, DCM)
│   │   ├── sar.py               # SAR image formation (RDA)
│   │   ├── metrics.py           # SAR quality metrics
│   │   └── api.py               # FastAPI endpoints
│   ├── requirements.txt
│   └── main.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main app with all components
│   │   └── App.css              # Premium dark UI styling
│   ├── package.json
│   └── index.html
├── start.bat                    # Windows launcher
├── start.sh                     # Linux/Mac launcher
├── .gitignore
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, NumPy, SciPy |
| Frontend | React 19, Vite, HTML Canvas |
| API | REST (JSON) |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Blank page | Hard refresh (Ctrl+Shift+R) |
| Backend error on frontend | Ensure backend is running on port 8000 |
| `pip install` fails | Use `python -m pip install -r requirements.txt` |
| `npm install` fails | Delete `node_modules/` and `package-lock.json`, then retry |
