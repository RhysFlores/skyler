"""Image -> boss pipeline: background removal, AI naming, save to the game folder."""
import base64
import hashlib
import io
import json
import os
import random
from pathlib import Path

import anthropic
import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from rembg import new_session, remove

register_heif_opener()  # lets Pillow open iPhone .heic photos

BOSS_DIR = Path(__file__).parent / "web" / "bosses"
MANIFEST = BOSS_DIR / "index.json"
MODEL = "claude-opus-5-5"

_rembg_session = None
_client = None


def normalize(image_bytes: bytes) -> bytes:
    """Decode any supported format (incl. HEIC), apply phone EXIF rotation, return PNG bytes."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
    img = img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") else img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def detect_faces(sprite: Image.Image) -> list:
    """Find faces in a cutout. Boxes are normalized to the sprite; eyes are normalized to their face box.
    The game crops these out as enemies and draws angry eyebrows over the eyes."""
    flat = Image.new("RGB", sprite.size, (128, 128, 128))
    flat.paste(sprite, mask=sprite.getchannel("A") if sprite.mode == "RGBA" else None)
    gray = cv2.cvtColor(np.array(flat), cv2.COLOR_RGB2GRAY)
    W, H = sprite.size
    found = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(max(24, W // 12),) * 2)
    faces = []
    for (x, y, w, h) in sorted(found, key=lambda f: -f[2] * f[3])[:3]:
        eyes = _eye_cascade.detectMultiScale(gray[y:y + int(h * 0.6), x:x + w], 1.1, 4, minSize=(w // 10, h // 10))
        eyes = sorted(sorted(eyes, key=lambda e: -e[2] * e[3])[:2], key=lambda e: e[0])
        eye_pts = [[round((ex + ew / 2) / w, 3), round((ey + eh / 2) / h, 3)] for ex, ey, ew, eh in eyes]
        if len(eye_pts) != 2 or eye_pts[1][0] - eye_pts[0][0] < 0.2:  # missed or double-detected eye
            eye_pts = [[0.32, 0.4], [0.68, 0.4]]
        faces.append({"box": [round(x / W, 4), round(y / H, 4), round(w / W, 4), round(h / H, 4)], "eyes": eye_pts})
    return faces


def backfill_faces() -> int:
    """Add face data to bosses made before face detection existed."""
    manifest = _load_manifest()
    changed = 0
    for entry in manifest:
        if "faces" not in entry and (BOSS_DIR / entry["image"]).exists():
            entry["faces"] = detect_faces(Image.open(BOSS_DIR / entry["image"]).convert("RGBA"))
            changed += 1
    if changed:
        MANIFEST.write_text(json.dumps(manifest, indent=2))
    return changed


def cut_out(image_bytes: bytes) -> Image.Image:
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = new_session("isnet-general-use")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    img.thumbnail((768, 768))
    out = remove(img, session=_rembg_session)
    bbox = out.getbbox()
    return out.crop(bbox) if bbox else out


BOSS_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Boss name, 1-3 words, e.g. 'Gregor'"},
        "title": {"type": "string", "description": "Epic epithet, e.g. 'Devourer of Pizza Rolls'"},
        "taunt": {"type": "string", "description": "One short line the boss yells when the fight starts"},
        "element": {"type": "string", "enum": ["fire", "ice", "poison", "shadow", "lightning", "chaos"]},
    },
    "required": ["name", "title", "taunt", "element"],
    "additionalProperties": False,
}

PROMPT = (
    "You're naming a boss for a goofy arcade bullet-hell game made from photos friends share. "
    "Look at this photo and invent a funny, over-the-top video game boss identity inspired by what you see "
    "(the subject, expression, pose, colors, setting, vibe). Keep it playful and good-natured, never mean about "
    "appearance, body, race, or anything sensitive."
)

_FALLBACK_NAMES = ["Grumblor", "Sir Chonk", "Vexmaw", "Lord Snoot", "Brakk", "Mildred", "The Goober"]
_FALLBACK_TITLES = ["Eater of Snacks", "the Unbothered", "Herald of Mondays", "Lord of the Group Chat"]


def name_boss(png_bytes: bytes) -> dict:
    global _client
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            if _client is None:
                _client = anthropic.Anthropic()
            resp = _client.beta.messages.create(
                model=MODEL,
                max_tokens=2000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={
                    "effort": "low",
                    "format": {"type": "json_schema", "schema": BOSS_SCHEMA},
                },
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/png",
                            "data": base64.standard_b64encode(png_bytes).decode(),
                        }},
                        {"type": "text", "text": PROMPT},
                    ],
                }],
            )
            if resp.stop_reason != "refusal":
                text = next(b.text for b in resp.content if b.type == "text")
                return json.loads(text)
        except (anthropic.APIError, StopIteration, json.JSONDecodeError) as e:
            print(f"[naming] AI naming failed, using random name: {e}")
    return {
        "name": random.choice(_FALLBACK_NAMES),
        "title": random.choice(_FALLBACK_TITLES),
        "taunt": "You dare challenge me?!",
        "element": random.choice(BOSS_SCHEMA["properties"]["element"]["enum"]),
    }


def _load_manifest() -> list:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return []


def process_image(image_bytes: bytes, source: str = "") -> dict | None:
    """Returns the new boss entry, or None if this photo is already a boss."""
    BOSS_DIR.mkdir(parents=True, exist_ok=True)
    boss_id = hashlib.sha1(image_bytes).hexdigest()[:12]
    manifest = _load_manifest()
    if any(b["id"] == boss_id for b in manifest):
        return None
    image_bytes = normalize(image_bytes)

    sprite = cut_out(image_bytes)
    buf = io.BytesIO()
    sprite.save(buf, "PNG")
    png = buf.getvalue()
    (BOSS_DIR / f"{boss_id}.png").write_bytes(png)

    # Send a small copy to the AI to keep requests cheap.
    small = sprite.copy()
    small.thumbnail((384, 384))
    sbuf = io.BytesIO()
    small.save(sbuf, "PNG")
    info = name_boss(sbuf.getvalue())

    entry = {"id": boss_id, "image": f"{boss_id}.png", "source": source, **info, "faces": detect_faces(sprite)}
    manifest.append(entry)
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    return entry

