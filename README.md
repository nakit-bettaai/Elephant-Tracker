# 🐘 Elephant Detector — Forest Wildlife Alert System

AI-powered elephant detection from CCTV cameras or uploaded images, with real-time LINE Notify alerts.

## Features

- 🔍 **YOLOv8 AI detection** — detects elephants with confidence scoring
- 📤 **Image upload** — drag & drop any photo for instant analysis
- 📹 **RTSP camera monitoring** — polls live CCTV feeds at configurable intervals
- 📲 **LINE Notify alerts** — sends alert + annotated image to LINE when elephant detected
- 📋 **Alert history** — searchable log of all detection events with thumbnails
- ⚙️ **Web dashboard** — configure everything from the browser

## Setup

### 1. Install dependencies

```bash
cd elephant-detector
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your LINE_NOTIFY_TOKEN
```

**Get a LINE Notify token:**
1. Go to https://notify-bot.line.me/my/
2. Click "Generate token"
3. Select the LINE group or 1-on-1 chat to receive alerts
4. Copy the token into `.env`

### 3. Run

```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

Open http://localhost:8001

On first run, YOLOv8 (~6 MB) downloads automatically.

## Usage

### Manual detection
Go to **Detect** tab → drag & drop or upload an image → result shown instantly with annotated image and LINE alert sent if elephant found.

### Live CCTV monitoring
Go to **Cameras** tab → Add camera with RTSP URL → system polls every N seconds (configurable) and sends LINE alert on detection.

### Settings
- **Confidence threshold** — 50% default; lower = more sensitive, higher = fewer false alarms
- **Poll interval** — how often to grab a frame from each camera (seconds)
- **Alert cooldown** — minimum time between alerts for the same camera (avoids spam)

## Files

| File | Description |
|------|-------------|
| `app.py` | FastAPI web server + REST API |
| `detector.py` | YOLOv8 elephant detection logic |
| `notifier.py` | LINE Notify integration |
| `static/index.html` | Web dashboard (single-page app) |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variable template |

## Improve accuracy

The default model is `yolov8n.pt` (nano — fastest). For better accuracy:

- Change `MODEL_NAME = "yolov8s.pt"` in `detector.py` for the small model
- Use `yolov8m.pt` or `yolov8l.pt` for even better accuracy (slower)
- For production in dense forest / low light, consider fine-tuning on elephant-specific datasets

---

Built with [Claude Code](https://claude.ai/code) 🤖
