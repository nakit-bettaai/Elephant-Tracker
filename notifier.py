import io
import logging
from typing import List, Optional
import httpx

logger = logging.getLogger(__name__)

LINE_NOTIFY_URL = "https://notify-api.line.me/api/notify"


# ─── LINE Notify ──────────────────────────────────────────────────────────────
async def send_line_notify(
    token: str,
    message: str,
    image_bytes: Optional[bytes] = None,
    image_filename: str = "detection.jpg",
) -> bool:
    if not token:
        logger.warning("LINE_NOTIFY_TOKEN not set — skipping")
        return False
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            if image_bytes:
                files = {"imageFile": (image_filename, io.BytesIO(image_bytes), "image/jpeg")}
                resp = await client.post(LINE_NOTIFY_URL, headers=headers,
                                         data={"message": message}, files=files)
            else:
                resp = await client.post(LINE_NOTIFY_URL, headers=headers,
                                         data={"message": message})
        if resp.status_code == 200:
            logger.info("LINE Notify sent")
            return True
        logger.error(f"LINE Notify failed: {resp.status_code} {resp.text}")
        return False
    except Exception as e:
        logger.error(f"LINE Notify error: {e}")
        return False


# ─── SMS via Twilio ───────────────────────────────────────────────────────────
async def send_sms(
    account_sid: str,
    auth_token: str,
    from_number: str,
    to_numbers: List[str],
    message: str,
) -> dict:
    """Send SMS to multiple numbers via Twilio. Returns {number: ok}."""
    if not (account_sid and auth_token and from_number and to_numbers):
        logger.warning("Twilio credentials incomplete — skipping SMS")
        return {}

    results = {}
    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"

    async with httpx.AsyncClient(timeout=15) as client:
        for to in to_numbers:
            to = to.strip()
            if not to:
                continue
            try:
                resp = await client.post(
                    url,
                    auth=(account_sid, auth_token),
                    data={"From": from_number, "To": to, "Body": message},
                )
                ok = resp.status_code in (200, 201)
                results[to] = ok
                if ok:
                    logger.info(f"SMS sent to {to}")
                else:
                    logger.error(f"SMS failed to {to}: {resp.text}")
            except Exception as e:
                logger.error(f"SMS error to {to}: {e}")
                results[to] = False
    return results


# ─── Message builders ─────────────────────────────────────────────────────────
def build_alert_message(source_name: str, elephant_count: int, confidences: List[float],
                         lat: Optional[float] = None, lng: Optional[float] = None) -> str:
    avg_conf = round(sum(confidences) / len(confidences) * 100) if confidences else 0
    gps_line = f"\n📍 GPS: {lat:.5f}, {lng:.5f}" if lat and lng else ""
    return "\n".join([
        "",
        "🐘 ELEPHANT ALERT",
        "พบช้างในพื้นที่ป่า / Elephant detected in forest",
        f"📷 Source: {source_name}",
        f"🔢 Count: {elephant_count} elephant(s)",
        f"🎯 Confidence: {avg_conf}%",
        gps_line,
        "⚠️ กรุณาดำเนินการตามขั้นตอนที่กำหนด / Follow alert protocol.",
    ])


def build_border_alert_line(source_name: str, village_name: str,
                             elephant_count: int, confidences: List[float],
                             lat: Optional[float] = None, lng: Optional[float] = None) -> str:
    avg_conf = round(sum(confidences) / len(confidences) * 100) if confidences else 0
    gps_line = f"\n📍 GPS: {lat:.5f}, {lng:.5f}" if lat and lng else ""
    maps_url = f"\n🗺️ maps.google.com/?q={lat},{lng}" if lat and lng else ""
    return "\n".join([
        "",
        "🚨🐘🚨 BORDER ALERT — URGENT 🚨🐘🚨",
        "⚠️  ช้างบุกรุกชายแดนหมู่บ้าน / Elephant near village border!",
        "",
        f"🏘️  Village at risk: {village_name}",
        f"📷  Camera: {source_name}",
        f"🔢  Count: {elephant_count} elephant(s)",
        f"🎯  Confidence: {avg_conf}%",
        gps_line,
        maps_url,
        "",
        "🔴 กรุณาแจ้งเตือนชาวบ้านทันที!",
        "   Alert villagers IMMEDIATELY.",
        "   ห้ามเข้าใกล้ช้าง / Do NOT approach.",
    ])


def build_border_alert_sms(village_name: str, elephant_count: int,
                            source_name: str,
                            lat: Optional[float] = None, lng: Optional[float] = None) -> str:
    gps = f" GPS:{lat:.4f},{lng:.4f}" if lat and lng else ""
    return (
        f"[ELEPHANT ALERT] {elephant_count} ช้าง/elephant(s) near {village_name} border! "
        f"Camera:{source_name}{gps} "
        f"ALERT VILLAGERS NOW. Do NOT approach."
    )
