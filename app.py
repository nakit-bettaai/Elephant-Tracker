import asyncio
import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
from typing import Dict, List, Optional, Set

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from detector import capture_rtsp_frame, detect_elephants
from notifier import (build_alert_message, build_border_alert_line,
                      build_border_alert_sms, send_line_notify, send_sms)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────
LINE_TOKEN    = os.getenv("LINE_NOTIFY_TOKEN", "")
CONFIDENCE    = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
STREAM_INTERVAL = int(os.getenv("STREAM_INTERVAL", "5"))
ALERT_COOLDOWN  = int(os.getenv("ALERT_COOLDOWN", "60"))

# ─── State ───────────────────────────────────────────────────────────────────
cameras:       Dict[str, Dict] = {}
zones:         Dict[str, Dict] = {}   # border zones
alert_history: List[Dict]      = []
ws_clients:    Set[WebSocket]  = set()
settings: dict = {
    "line_token":      LINE_TOKEN,
    "confidence":      CONFIDENCE,
    "stream_interval": STREAM_INTERVAL,
    "alert_cooldown":  ALERT_COOLDOWN,
    # Twilio SMS
    "twilio_sid":      os.getenv("TWILIO_ACCOUNT_SID", ""),
    "twilio_token":    os.getenv("TWILIO_AUTH_TOKEN", ""),
    "twilio_from":     os.getenv("TWILIO_FROM_NUMBER", ""),
    # Extra LINE token specifically for border alerts (optional — falls back to main)
    "border_line_token": os.getenv("BORDER_LINE_TOKEN", ""),
}


# ─── Geo helpers ─────────────────────────────────────────────────────────────
def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Distance in metres between two GPS points."""
    R = 6_371_000
    dlat, dlng = radians(lat2 - lat1), radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def point_in_polygon(lat: float, lng: float, coords: List[List[float]]) -> bool:
    """Ray-casting polygon containment test. coords = [[lat,lng],…]"""
    inside = False
    n = len(coords)
    j = n - 1
    for i in range(n):
        yi, xi = coords[i][0], coords[i][1]
        yj, xj = coords[j][0], coords[j][1]
        if ((yi > lat) != (yj > lat)) and (lng < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def check_zones(lat: float, lng: float) -> List[Dict]:
    """Return every border zone that contains the point."""
    triggered = []
    for zone in zones.values():
        if zone["shape"] == "circle":
            dist = haversine(lat, lng, zone["lat"], zone["lng"])
            if dist <= zone["radius"]:
                triggered.append(zone)
        elif zone["shape"] == "polygon" and zone.get("coords"):
            if point_in_polygon(lat, lng, zone["coords"]):
                triggered.append(zone)
    return triggered


# ─── WebSocket broadcast ─────────────────────────────────────────────────────
async def broadcast(event: dict):
    dead = set()
    msg = json.dumps(event)
    for ws in ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)


# ─── Alert dispatch ──────────────────────────────────────────────────────────
async def dispatch_alerts(record: dict, triggered_zones: List[Dict]):
    confidences = [d["confidence"] for d in record["detections"]]
    lat, lng = record.get("lat"), record.get("lng")

    if triggered_zones:
        # ── BORDER ALERT (urgent) ──────────────────────────────────────────
        for zone in triggered_zones:
            village = zone["village"]
            logger.warning(f"BORDER ALERT: {record['elephant_count']} elephant(s) near {village}")

            # LINE — use zone-specific token or fall back to global
            line_tok = zone.get("line_token") or settings["border_line_token"] or settings["line_token"]
            if line_tok:
                msg = build_border_alert_line(
                    record["source"], village, record["elephant_count"],
                    confidences, lat, lng
                )
                asyncio.create_task(
                    send_line_notify(line_tok, msg, record.get("_raw_image"))
                )

            # SMS to all numbers in the zone
            sms_numbers = [n.strip() for n in zone.get("sms_numbers", "").split(",") if n.strip()]
            if sms_numbers and settings["twilio_sid"]:
                sms_msg = build_border_alert_sms(village, record["elephant_count"],
                                                  record["source"], lat, lng)
                asyncio.create_task(send_sms(
                    settings["twilio_sid"], settings["twilio_token"],
                    settings["twilio_from"], sms_numbers, sms_msg,
                ))
    else:
        # ── NORMAL forest alert ────────────────────────────────────────────
        if settings["line_token"]:
            msg = build_alert_message(record["source"], record["elephant_count"], confidences, lat, lng)
            asyncio.create_task(
                send_line_notify(settings["line_token"], msg, record.get("_raw_image"))
            )


# ─── Background stream monitor ───────────────────────────────────────────────
async def stream_monitor_loop():
    logger.info("Stream monitor started")
    while True:
        for cam_id, cam in list(cameras.items()):
            if not cam.get("enabled"):
                continue
            try:
                frame_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, capture_rtsp_frame, cam["url"]
                )
                if frame_bytes is None:
                    cam["status"] = "error"
                    continue
                cam["status"] = "ok"

                result = await asyncio.get_event_loop().run_in_executor(
                    None, detect_elephants, frame_bytes, settings["confidence"]
                )

                if result["detected"]:
                    now = time.time()
                    cam["last_detection"] = datetime.now().isoformat()
                    cam["last_count"] = result["elephant_count"]

                    lat, lng = cam.get("lat"), cam.get("lng")
                    triggered_zones = check_zones(lat, lng) if lat and lng else []

                    record = _make_record(cam["name"], result, lat=lat, lng=lng,
                                          cam_id=cam_id, border_zones=triggered_zones)
                    alert_history.insert(0, record)
                    if len(alert_history) > 500:
                        alert_history.pop()

                    asyncio.create_task(broadcast({"type": "detection", "record": _strip_raw(record)}))

                    if now - cam.get("last_alert", 0) > settings["alert_cooldown"]:
                        cam["last_alert"] = now
                        await dispatch_alerts(record, triggered_zones)

            except Exception as e:
                logger.error(f"[{cam['name']}] Monitor error: {e}")

        await asyncio.sleep(settings["stream_interval"])


def _make_record(source: str, result: dict, lat=None, lng=None,
                 cam_id=None, border_zones=None) -> dict:
    img_b64 = base64.b64encode(result["annotated_image"]).decode()
    return {
        "id":             f"{time.time_ns()}",
        "timestamp":      datetime.now().isoformat(),
        "source":         source,
        "cam_id":         cam_id,
        "elephant_count": result["elephant_count"],
        "detections":     result["detections"],
        "image_b64":      img_b64,
        "_raw_image":     result["annotated_image"],   # bytes — stripped before WS send
        "detected":       result["detected"],
        "lat":            lat,
        "lng":            lng,
        "border_alert":   bool(border_zones),
        "zones":          [z["name"] for z in (border_zones or [])],
        "villages":       [z["village"] for z in (border_zones or [])],
    }


def _strip_raw(record: dict) -> dict:
    """Remove binary field before JSON serialisation."""
    r = dict(record)
    r.pop("_raw_image", None)
    return r


# ─── App lifecycle ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(stream_monitor_loop())
    yield
    task.cancel()


app = FastAPI(title="Elephant Tracker", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ─── WebSocket ───────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    await ws.send_text(json.dumps({
        "type":    "init",
        "cameras": list(cameras.values()),
        "zones":   list(zones.values()),
        "history": [_strip_raw(r) for r in alert_history[:20] if r.get("lat")],
    }))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_clients.discard(ws)


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()


@app.post("/api/detect")
async def detect_upload(
    file: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lng: Optional[float] = Form(None),
    location_name: str = Form(""),
):
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 20 MB)")
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, detect_elephants, data, settings["confidence"]
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    source = location_name or file.filename or "upload"
    triggered_zones = check_zones(lat, lng) if lat and lng else []
    record = _make_record(source, result, lat=lat, lng=lng, border_zones=triggered_zones)
    alert_history.insert(0, record)
    if len(alert_history) > 500:
        alert_history.pop()

    asyncio.create_task(broadcast({"type": "detection", "record": _strip_raw(record)}))

    if result["detected"]:
        await dispatch_alerts(record, triggered_zones)

    return JSONResponse(_strip_raw(record))


@app.get("/api/history")
async def get_history(limit: int = 100, detected_only: bool = False):
    h = [_strip_raw(r) for r in alert_history]
    if detected_only:
        h = [r for r in h if r["detected"]]
    return h[:limit]


@app.get("/api/sightings")
async def get_sightings(hours: float = 24):
    cutoff = time.time() - (hours or 99999) * 3600
    return [
        _strip_raw(r) for r in alert_history
        if r["detected"] and r.get("lat") is not None
        and datetime.fromisoformat(r["timestamp"]).timestamp() > cutoff
    ]


# ─── Cameras ─────────────────────────────────────────────────────────────────
@app.get("/api/cameras")
async def get_cameras():
    return list(cameras.values())


@app.post("/api/cameras")
async def add_camera(
    name: str = Form(...), url: str = Form(...),
    lat: Optional[float] = Form(None), lng: Optional[float] = Form(None),
):
    cam_id = f"cam_{int(time.time_ns())}"
    cameras[cam_id] = {
        "id": cam_id, "name": name, "url": url, "lat": lat, "lng": lng,
        "enabled": True, "status": "connecting",
        "last_alert": 0, "last_detection": None, "last_count": 0,
    }
    asyncio.create_task(broadcast({"type": "camera_added", "camera": cameras[cam_id]}))
    return cameras[cam_id]


@app.delete("/api/cameras/{cam_id}")
async def remove_camera(cam_id: str):
    if cam_id not in cameras:
        raise HTTPException(404)
    del cameras[cam_id]
    asyncio.create_task(broadcast({"type": "camera_removed", "cam_id": cam_id}))
    return {"ok": True}


@app.patch("/api/cameras/{cam_id}/toggle")
async def toggle_camera(cam_id: str):
    if cam_id not in cameras:
        raise HTTPException(404)
    cameras[cam_id]["enabled"] = not cameras[cam_id]["enabled"]
    return cameras[cam_id]


# ─── Border Zones ─────────────────────────────────────────────────────────────
@app.get("/api/zones")
async def get_zones():
    return list(zones.values())


@app.post("/api/zones")
async def add_zone(
    name:        str   = Form(...),
    village:     str   = Form(...),
    shape:       str   = Form("circle"),      # circle | polygon
    lat:         Optional[float] = Form(None),
    lng:         Optional[float] = Form(None),
    radius:      float = Form(500),           # metres (circle only)
    coords:      str   = Form(""),            # JSON [[lat,lng],…] (polygon)
    color:       str   = Form("#ff5722"),
    sms_numbers: str   = Form(""),            # comma-separated phone numbers
    line_token:  str   = Form(""),            # zone-specific LINE token (optional)
):
    zone_id = f"zone_{int(time.time_ns())}"
    parsed_coords = json.loads(coords) if coords else []
    zones[zone_id] = {
        "id": zone_id, "name": name, "village": village,
        "shape": shape, "lat": lat, "lng": lng, "radius": radius,
        "coords": parsed_coords, "color": color,
        "sms_numbers": sms_numbers, "line_token": line_token,
    }
    asyncio.create_task(broadcast({"type": "zone_added", "zone": zones[zone_id]}))
    return zones[zone_id]


@app.delete("/api/zones/{zone_id}")
async def remove_zone(zone_id: str):
    if zone_id not in zones:
        raise HTTPException(404)
    del zones[zone_id]
    asyncio.create_task(broadcast({"type": "zone_removed", "zone_id": zone_id}))
    return {"ok": True}


# ─── Settings ─────────────────────────────────────────────────────────────────
@app.get("/api/settings")
async def get_settings():
    safe = {k: v for k, v in settings.items()
            if k not in ("line_token", "twilio_token", "border_line_token")}
    safe["line_configured"]   = bool(settings["line_token"])
    safe["twilio_configured"] = bool(settings["twilio_sid"] and settings["twilio_token"] and settings["twilio_from"])
    return safe


@app.post("/api/settings")
async def update_settings(
    line_token:        str   = Form(""),
    border_line_token: str   = Form(""),
    confidence:        float = Form(0.5),
    stream_interval:   int   = Form(5),
    alert_cooldown:    int   = Form(60),
    twilio_sid:        str   = Form(""),
    twilio_token:      str   = Form(""),
    twilio_from:       str   = Form(""),
):
    if line_token:        settings["line_token"]        = line_token
    if border_line_token: settings["border_line_token"] = border_line_token
    if twilio_sid:        settings["twilio_sid"]        = twilio_sid
    if twilio_token:      settings["twilio_token"]      = twilio_token
    if twilio_from:       settings["twilio_from"]       = twilio_from
    settings["confidence"]      = max(0.1, min(1.0, confidence))
    settings["stream_interval"] = max(1, stream_interval)
    settings["alert_cooldown"]  = max(0, alert_cooldown)
    return {"ok": True}


@app.post("/api/test-line")
async def test_line():
    if not settings["line_token"]:
        raise HTTPException(400, "LINE Notify token not configured")
    ok = await send_line_notify(
        settings["line_token"],
        "\n🔔 Elephant Tracker — ทดสอบการแจ้งเตือน / Alert test ✅",
    )
    return {"ok": ok}


@app.post("/api/test-sms")
async def test_sms(to: str = Form(...)):
    if not settings["twilio_sid"]:
        raise HTTPException(400, "Twilio credentials not configured")
    result = await send_sms(
        settings["twilio_sid"], settings["twilio_token"],
        settings["twilio_from"], [to],
        "[TEST] Elephant Tracker SMS alert working ✅",
    )
    return result


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
