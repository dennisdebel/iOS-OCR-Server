#
# pip3 install requests pillow opencv-python pymupdf
#
# usage python3 ocr-server-example.py

import os
import sys
import re   
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import numpy as np
import cv2

url = "http://192.168.1.73:8000/upload"  # Replace with your IP address
file_path = "input.jpg"

# ===== Select font (supports Chinese and English), font size auto-scales with box height =====
def pick_font_path():
    font_candidates = [
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        # Windows
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msjh.ttc",
        r"C:\Windows\Fonts\arialuni.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        # Linux / Noto
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in font_candidates:
        if os.path.exists(path):
            return path
    return None  # Pillow default will be used as last resort

# --- PDF export: image + invisible text layer ---
def save_searchable_pdf(img_pil, boxes, out_path, prefer_translated=True, visible_text=False):
    import io, fitz  # PyMuPDF

    W, H = img_pil.size
    doc = fitz.open()
    page = doc.new_page(width=W, height=H)

    # Put the composed image as the page background
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=95)   # or "PNG" if you want lossless
    page.insert_image(fitz.Rect(0, 0, W, H), stream=buf.getvalue())

    # Choose a font that supports your glyphs (CJK etc.)
    font_path = pick_font_path()  # you already have this
    # If your path is None or a TTC that PyMuPDF can’t embed, consider a TTF/OTF like NotoSansCJK.

    for b in boxes:
        try:
            x = float(b["x"]); y = float(b["y"])
            w = float(b["w"]); h = float(b["h"])
        except Exception:
            continue

        # Get text (prefer translated_text)
        txt = (b.get("translated_text") if prefer_translated else None) or b.get("text") or ""
        txt = txt.strip()
        if not txt:
            continue

        # Clamp to page
        x1 = max(0, x); y1 = max(0, y)
        x2 = min(W, x + w); y2 = min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        rect = fitz.Rect(x1, y1, x2, y2)

        # Font size doesn’t have to be exact for searchability. Keep it reasonable:
        fontsize = max(6, int(0.9 * (y2 - y1)))  # rough fit; PDF will wrap if needed

        # Make the text invisible but selectable (render_mode=3).
        # If you want it visible too, set visible_text=True.
        render_mode = 0 if visible_text else 3

        page.insert_textbox(
            rect,
            txt,
            fontsize=fontsize,
            fontfile=font_path if font_path else None,  # embed if available
            color=(0, 0, 0),        # color ignored when invisible
            align=fitz.TEXT_ALIGN_CENTER,
            render_mode=render_mode,
            overlay=True,
        )

    doc.save(out_path, deflate=True)
    doc.close()

# ---------- Helpers for measuring / wrapping / WCAG-ish contrast ----------

_CJK_RANGE = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u3040-\u30FF\uAC00-\uD7AF]"
)

def is_cjk_text(text: str) -> bool:
    return _CJK_RANGE.search(text) is not None

def measure(draw: ImageDraw.ImageDraw, txt: str, font: ImageFont.FreeTypeFont):
    l, t, r, b = draw.textbbox((0, 0), txt, font=font, stroke_width=0)
    return r - l, b - t

def wrap_by_width(draw, text, font, max_w):
    lines = []
    text = text.replace("\r", "").strip()
    if not text:
        return [""]
    # CJK with no spaces → wrap per character
    if is_cjk_text(text) and (" " not in text):
        cur = ""
        for ch in text:
            candidate = cur + ch
            w, _ = measure(draw, candidate, font)
            if w <= max_w or cur == "":
                cur = candidate
            else:
                lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
        return lines
    # Latin / mixed → greedy word wrapping
    words = text.split()
    cur = ""
    for w_ in words:
        candidate = w_ if cur == "" else f"{cur} {w_}"
        w, _ = measure(draw, candidate, font)
        if w <= max_w or cur == "":
            cur = candidate
        else:
            lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines

def fit_text_to_box(draw, text, font_path, box_w, box_h, padding=2, max_lines=None):
    """
    Returns (font, lines, line_height, total_h) for the largest font-size that fits.
    If font_path is None, falls back to ImageFont.load_default() without resizing.
    """
    avail_w = max(1, int(box_w - 2 * padding))
    avail_h = max(1, int(box_h - 2 * padding))

    # If we don't have a TTF path, do a best-effort with default font
    if not font_path:
        font = ImageFont.load_default()
        lines = wrap_by_width(draw, text, font, avail_w)
        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        total_h = line_h * len(lines)
        return font, lines, line_h, total_h

    low, high = 4, max(8, int(min(avail_h, 512)))  # reasonable search bounds
    best = None
    while low <= high:
        mid = (low + high) // 2
        font = ImageFont.truetype(font_path, size=mid)
        lines = wrap_by_width(draw, text, font, avail_w)

        if max_lines is not None and len(lines) > max_lines:
            # Too many lines → smaller font
            high = mid - 1
            continue

        ascent, descent = font.getmetrics()
        # A bit of extra leading improves readability
        line_h = ascent + descent + max(1, int(mid * 0.15))
        total_h = line_h * len(lines)

        # Check both height and width constraints
        if total_h <= avail_h and all(measure(draw, ln, font)[0] <= avail_w for ln in lines):
            best = (font, lines, line_h, total_h)
            low = mid + 1  # try bigger
        else:
            high = mid - 1  # too big → go smaller

    if best is None:
        # Fall back to the smallest size
        font = ImageFont.truetype(font_path, size=low)
        lines = wrap_by_width(draw, text, font, avail_w)
        ascent, descent = font.getmetrics()
        line_h = ascent + descent + max(1, int(low * 0.15))
        total_h = line_h * len(lines)
        return font, lines, line_h, total_h

    return best

def _srgb_to_lin(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _relative_luminance(rgb):
    # rgb: (R,G,B) 0..255
    r, g, b = _srgb_to_lin(np.array(rgb, dtype=float))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def _contrast_ratio(fg_rgb, bg_rgb):
    L1 = _relative_luminance(fg_rgb)
    L2 = _relative_luminance(bg_rgb)
    L1, L2 = (max(L1, L2), min(L1, L2))
    return (L1 + 0.05) / (L2 + 0.05)

def _ensure_min_contrast(fg_rgb, bg_rgb, min_ratio=3.0, steps=20):
    # iteratively nudge fg toward black/white to meet contrast
    fg = np.array(fg_rgb, dtype=float)
    bg = np.array(bg_rgb, dtype=float)
    if _contrast_ratio(fg, bg) >= min_ratio:
        return tuple(int(round(x)) for x in fg)

    # choose direction (toward black or white) that increases contrast faster
    toward_black = np.array([0, 0, 0], dtype=float)
    toward_white = np.array([255, 255, 255], dtype=float)

    def try_mix(target):
        f = fg.copy()
        for i in range(1, steps + 1):
            alpha = i / steps
            f_try = (1 - alpha) * fg + alpha * target
            if _contrast_ratio(f_try, bg) >= min_ratio:
                return f_try
        return f  # give up

    # pick the better direction
    candidate_b = try_mix(toward_black)
    candidate_w = try_mix(toward_white)
    if _contrast_ratio(candidate_b, bg) >= _contrast_ratio(candidate_w, bg):
        out = candidate_b
    else:
        out = candidate_w
    return tuple(int(round(x)) for x in out)

# --- Estimate text color from the cropped box ---
def _robust_median_color(pixels):
    # pixels: (N,3) RGB
    if pixels.size == 0:
        return (20, 20, 20)
    # clip out 5% extremes per channel for robustness
    lo = np.percentile(pixels, 5, axis=0)
    hi = np.percentile(pixels, 95, axis=0)
    keep = np.all((pixels >= lo) & (pixels <= hi), axis=1)
    return tuple(int(round(x)) for x in np.median(pixels[keep], axis=0))

def _estimate_text_and_bg_colors(box_rgb, overlay_bg_rgb=None):
    """
    box_rgb: np.uint8 HxWx3 from the *original* image (not blurred).
    overlay_bg_rgb: np.uint8 HxWx3 of the *composed* frosted patch (optional).
    Returns (fg_rgb, bg_rgb, mask) where mask is boolean HxW for text.
    """
    if box_rgb.size == 0:
        return (20, 20, 20), (255, 255, 255), None

    # LAB + Otsu to separate dark/bright
    lab = cv2.cvtColor(box_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    blur = cv2.GaussianBlur(L, (0, 0), 1.0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask_dark   = (L < th)   # text likely dark
    mask_bright = (L >= th)  # text likely bright (e.g., white on black)

    # Use edges to decide which mask hugs glyphs better
    edges = cv2.Canny(cv2.GaussianBlur(L, (0, 0), 1.2), 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    e_total = max(1, int(edges.sum() > 0) and int(np.count_nonzero(edges)))

    score_dark   = np.count_nonzero(edges & mask_dark.astype(np.uint8))
    score_bright = np.count_nonzero(edges & mask_bright.astype(np.uint8))

    mask = mask_dark if score_dark >= score_bright else mask_bright

    # Slight fatten to include glyph interiors
    mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.erode(mask, np.ones((2, 2), np.uint8), iterations=1).astype(bool)

    # Text color = median color of masked text pixels from the *original* patch
    text_pixels = box_rgb[mask]
    fg_rgb = _robust_median_color(text_pixels)

    # Background color = median of (overlay patch if provided) else inverse mask on original
    if overlay_bg_rgb is not None and overlay_bg_rgb.size:
        bg_pixels = overlay_bg_rgb[~mask]
    else:
        bg_pixels = box_rgb[~mask]
    if bg_pixels.size == 0:
        bg_rgb = (255, 255, 255)
    else:
        bg_rgb = _robust_median_color(bg_pixels)

    return fg_rgb, bg_rgb, mask

# ===== Draw box and small text =====
def draw_boxes(
    img_pil: Image.Image,
    boxes,
    line_thickness: int = 4,
    pad_ratio: float = 0.06,
    text_mode: str = "multiline",  # "multiline" or "single_line"
    align: str = "center",         # "left" | "center" | "right"
    valign: str = "center",        # "top" | "center" | "bottom"
    draw_box: bool = True,
    bg_style: str = "blur",        # "none" | "solid" | "blur"
    blur_radius: int = 8,
    bg_tint=(255, 255, 255),
    bg_tint_opacity: int = 80      # 0..255 (higher = milkier glass)
) -> Image.Image:
    draw = ImageDraw.Draw(img_pil)
    font_path = pick_font_path()

    # Pre-blur once for speed; we’ll cut patches from this instead of re-blurring per box
    preblur = img_pil.filter(ImageFilter.GaussianBlur(blur_radius)) if bg_style == "blur" else None

    for b in boxes:
        try:
            x = float(b["x"]); y = float(b["y"])
            w = float(b["w"]); h = float(b["h"])
            text = str(b.get("translated_text", ""))
        except Exception:
            continue

        x2, y2 = x + w, y + h
        pad = max(2, int(h * pad_ratio))

        # Draw red box on top later (so it isn’t blurred)
        # First: background treatment inside the box
        ix1, iy1 = max(0, int(round(x + 1))),      max(0, int(round(y + 1)))
        ix2, iy2 = min(img_pil.width,  int(round(x2 - 1))), min(img_pil.height, int(round(y2 - 1)))
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        if bg_style == "solid":
            draw.rectangle([ix1, iy1, ix2, iy2], fill=(255, 255, 255))
        elif bg_style == "blur":
            # 1) Take ORIGINAL patch (unblurred) + the preblurred patch
            base_region  = img_pil.crop((ix1, iy1, ix2, iy2)).convert("RGBA")  # original
            blur_region  = preblur.crop((ix1, iy1, ix2, iy2)).convert("RGBA")  # blurred

            # Optional white tint for frosted look
            if bg_tint_opacity > 0:
                tint = Image.new("RGBA", blur_region.size, (*bg_tint, bg_tint_opacity))
                blur_tinted = Image.alpha_composite(blur_region, tint)
            else:
                blur_tinted = blur_region

            # 2) **Estimate colors BEFORE** compositing/pasting
            box_rgb     = np.array(base_region.convert("RGB"))       # original pixels
            overlay_rgb = np.array(blur_tinted.convert("RGB"))       # what will be under text
            fg_rgb, bg_rgb, _ = _estimate_text_and_bg_colors(box_rgb, overlay_bg_rgb=overlay_rgb)

            # Enforce readable contrast against frosted background
            fg_draw = _ensure_min_contrast(fg_rgb, bg_rgb, min_ratio=3.0)
            stroke_draw = (255, 255, 255) if _relative_luminance(fg_draw) < 0.5 else (20, 20, 20)

            # 3) Now compose the frosted patch and paste it back
            composed = Image.alpha_composite(base_region, blur_tinted)
            img_pil.paste(composed.convert("RGB"), (ix1, iy1))

        # Now (optionally) the red outline so it sits above the blur
        #if draw_box:
            #draw.rectangle([x, y, x2, y2], outline=(255, 0, 0), width=line_thickness)

        # --- Text auto-fit (unchanged idea) ---
        if not text.strip():
            continue
        max_lines = 1 if text_mode == "single_line" else None
        font, lines, line_h, total_h = fit_text_to_box(
            draw, text, font_path, w, h, padding=pad, max_lines=max_lines
        )

        # Positioning
        avail_w = max(1, int(w - 2 * pad))
        avail_h = max(1, int(h - 2 * pad))
        if valign == "top":
            ty = int(y + pad)
        elif valign == "bottom":
            ty = int(y + pad + (avail_h - total_h))
        else:
            ty = int(y + pad + (avail_h - total_h) / 2)

        for ln in lines:
            lw, _ = measure(draw, ln, font)
            if align == "left":
                tx = int(x + pad)
            elif align == "right":
                tx = int(x + pad + (avail_w - lw))
            else:
                tx = int(x + pad + (avail_w - lw) / 2)

            draw.text(
                (tx, ty),
                ln,
                font=font,
                fill=tuple(fg_draw),
                stroke_width=max(1, int(font.size * 0.00)),
                stroke_fill=tuple(stroke_draw)
            )
            ty += line_h

    return img_pil


def main():
    if not os.path.exists(file_path):
        print(f"[ERROR] Image not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    # 1) Upload
    with open(file_path, "rb") as f:
        files = {"file": f}
        headers = {"Accept": "application/json"}
        try:
            response = requests.post(url, files=files, headers=headers, timeout=60)
        except requests.RequestException as e:
            print(f"[ERROR] Request failed: {e}", file=sys.stderr)
            sys.exit(2)

    print("status code:", response.status_code)

    # 2) Check HTTP and JSON
    if response.status_code != 200:
        print("response:", response.text[:500])
        sys.exit(3)

    try:
        data = response.json()
        print(data)
    except ValueError:
        print("[ERROR] Not JSON response")
        print("response:", response.text[:500])
        sys.exit(4)

    if not data.get("success", False):
        print("[ERROR] Server returned failure:", data)
        sys.exit(5)

    print("response ok")

    # 3) Load original image (using PIL)
    img_pil = Image.open(file_path).convert("RGB")

    # If server returns different dimensions (should usually match), use server dimensions
    W = int(data.get("image_width", img_pil.width))
    H = int(data.get("image_height", img_pil.height))
    if (W, H) != (img_pil.width, img_pil.height):
        img_pil = img_pil.resize((W, H), Image.BICUBIC)

    boxes = data.get("ocr_boxes", [])
    #img_pil = draw_boxes(img_pil, boxes)
    img_pil = draw_boxes(
        img_pil,
        boxes,
        bg_style="blur",        # <- frosted background
        blur_radius=8,          # bump up for softer blur
        bg_tint_opacity=80,     # 0 = no white tint; 80–120 looks nice
        line_thickness=4,
        text_mode="single_line", #or multiline
        align="center",
        valign="center"
    )
    out_pdf = "output.pdf"
    save_searchable_pdf(
        img_pil,
        boxes,
        out_path=out_pdf,
        prefer_translated=True,  # use per-box translated_text when available
        visible_text=False       # keep PDF text invisible but selectable
    )
    print(f"[OK] PDF saved to {out_pdf}")

    # 4) Display
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow("OCR Preview", img_cv)
    print("Press any key on the image window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()