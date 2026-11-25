"""
Face Pixelator

Drag a folder, or one or more photos, onto the window. Every face found is
pixelated or blurred (choose with the gear button), and the results are saved to
a "pixelated" or "blurred" folder next to the photos.
Runs fully offline: faces are found with the YuNet model file next to this script.
"""
import os
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD

try:  # lets iPhone .heic photos be opened
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "face_detection_yunet_2023mar.onnx")
SAMPLE = os.path.join(HERE, "sample.jpg")  # photo used for the before/after preview in Settings
OUTPUT_DIRS = {"pixelate": "pixelated", "blur": "blurred"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}

FADE = 1.4         # soft edges: full effect over the face oval, fading out by 1.4x its size
MIN_SCORE = 0.6     # detection confidence: lower finds more faces, but more false hits
# Each photo is scanned at these sizes: large sizes find small faces in group shots,
# small sizes find big close-up faces. Results are merged.
DETECT_SIZES = (2048, 1024, 640, 320)


def detect_faces(detector, rgb):
    """Return face boxes (x, y, w, h) in full-image pixels."""
    h, w = rgb.shape[:2]
    boxes, scores = [], []
    for scale in sorted({min(1.0, size / max(h, w)) for size in DETECT_SIZES}):
        small = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
        bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = detector.detect(bgr)
        for face in faces if faces is not None else []:
            boxes.append([float(v / scale) for v in face[:4]])
            scores.append(float(face[14]))
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, MIN_SCORE, 0.3)  # drop the same face found at several sizes
    return [boxes[i] for i in np.array(keep).flatten()]


def hide_face(rgb, box, mode, strength, area, soft):
    """Pixelate or blur one face in place.

    Strength 1-10 is measured against the face size, so it looks the same in any photo.
    Area 1-10 sets how much is covered: 1 = middle of the face (0.6x the detected face box),
    3 = the face box, 6 = face box plus hair, ears and chin (1.6x), 10 = whole head (2.4x).
    soft=True uses a round shape that fades out instead of a square box.
    """
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    half = (0.4 + 0.2 * area) / 2
    rx, ry = w * half, h * half  # half-size of the covered area
    reach = FADE if soft else 1.0
    img_h, img_w = rgb.shape[:2]
    x0, y0 = max(0, int(cx - rx * reach)), max(0, int(cy - ry * reach))
    x1, y1 = min(img_w, int(cx + rx * reach)), min(img_h, int(cy + ry * reach))
    if x1 <= x0 or y1 <= y0:
        return
    rw, rh = x1 - x0, y1 - y0
    region = rgb[y0:y1, x0:x1]
    # Pixelate: 20 blocks across the face at strength 1, down to 2 blocks at 10.
    # Blur: shrink so the face is 64px wide, then blur more with each strength step.
    scale = ((22 - 2 * strength) if mode == "pixelate" else 64) / max(w, h)
    small = cv2.resize(region, (max(1, round(rw * scale)), max(1, round(rh * scale))), interpolation=cv2.INTER_AREA)
    if mode == "pixelate":
        hidden = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    else:
        hidden = cv2.resize(cv2.GaussianBlur(small, (0, 0), strength), (rw, rh), interpolation=cv2.INTER_LINEAR)
    if soft:
        # Full effect inside the oval, fading smoothly to nothing at FADE x its size.
        dx = ((np.arange(x0, x1, dtype=np.float32) - cx) / rx) ** 2
        dy = ((np.arange(y0, y1, dtype=np.float32) - cy) / ry) ** 2
        t = np.clip((FADE - np.sqrt(dy[:, None] + dx[None, :])) / (FADE - 1), 0, 1)
        alpha = t * t * (3 - 2 * t)
        hidden = cv2.blendLinear(hidden, np.ascontiguousarray(region), alpha, 1 - alpha)
    rgb[y0:y1, x0:x1] = hidden


def find_images(paths):
    """Turn dropped paths into image files: a folder gives its images, an image file gives itself."""
    images = []
    for path in paths:
        candidates = [os.path.join(path, f) for f in sorted(os.listdir(path))] if os.path.isdir(path) else [path]
        images += [p for p in candidates if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMAGE_EXTS]
    return images


def process_images(images, mode, strength, area, soft, log):
    """Hide faces in each image. Results go to a "pixelated"/"blurred" folder next to each image."""
    out_name = OUTPUT_DIRS[mode]
    out_dirs = sorted({os.path.join(os.path.dirname(p), out_name) for p in images})
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)
    detector = cv2.FaceDetectorYN.create(MODEL, "", (320, 320), MIN_SCORE, 0.3, 5000)
    total_faces, no_face = 0, []

    for i, path in enumerate(images, 1):
        name = os.path.basename(path)
        prefix = f"[{i}/{len(images)}] {name}:"
        try:
            with Image.open(path) as im:
                icc = im.info.get("icc_profile")
                rgb = np.array(ImageOps.exif_transpose(im).convert("RGB"))

            faces = detect_faces(detector, rgb)
            for face in faces:
                hide_face(rgb, face, mode, strength, area, soft)

            stem, ext = os.path.splitext(name)
            if ext.lower() in (".heic", ".heif"):
                ext = ".jpg"
            # Saved fresh, so EXIF metadata (GPS location, camera, date) is not copied over.
            save_opts = {"quality": 95}
            if icc:
                save_opts["icc_profile"] = icc  # keeps colours accurate
            Image.fromarray(rgb).save(os.path.join(os.path.dirname(path), out_name, stem + ext), **save_opts)

            total_faces += len(faces)
            if faces:
                log(f"{prefix} {len(faces)} face(s) {out_name}")
            else:
                no_face.append(name)
                log(f"{prefix} no face found (copied as is)")
        except Exception as e:
            log(f"{prefix} skipped, could not process ({e})")

    log(f"\nDone. {total_faces} face(s) {out_name} across {len(images)} image(s).")
    for out_dir in out_dirs:
        log(f"Saved to: {out_dir}")
    if no_face:
        log(f"No face was found in {len(no_face)} image(s). Check these by eye before sharing:")
        for name in no_face:
            log(f"   {name}")


def main():
    root = TkinterDnD.Tk()
    root.title("Face Pixelator")
    root.geometry("640x520")
    messages = queue.Queue()
    busy = threading.Event()
    mode = tk.StringVar(value="pixelate")
    strength = tk.IntVar(value=8)
    area = tk.IntVar(value=6)
    soft = tk.BooleanVar(value=False)

    def open_settings():
        win = tk.Toplevel(root)
        win.title("Settings")
        win.resizable(False, False)
        win.transient(root)
        win.grab_set()

        options = tk.Frame(win)
        options.pack(fill="x", padx=16, pady=(14, 0))
        tk.Label(options, text="Hide faces by:", font=("Segoe UI", 11)).pack(side="left")
        radios = [tk.Radiobutton(options, text=text, value=value, variable=mode, font=("Segoe UI", 11))
                  for value, text in (("pixelate", "Pixelate"), ("blur", "Blur"))]
        for radio in radios:
            radio.pack(side="left", padx=8)
        soft_box = tk.Checkbutton(win, text="Soft round edges (fade out instead of a square box)",
                                  variable=soft, font=("Segoe UI", 11))
        soft_box.pack(anchor="w", padx=12, pady=(6, 0))
        sliders = [tk.Scale(win, label=label, from_=1, to=10, orient="horizontal", variable=var, font=("Segoe UI", 10))
                   for label, var in (("Strength", strength),
                                      ("Area   (1 = middle of the face only,  10 = whole head)", area))]
        for slider in sliders:
            slider.pack(fill="x", padx=16, pady=(8, 0))
        preview = tk.Frame(win)
        preview.pack(padx=16, pady=10)
        tk.Button(win, text="Done", width=10, command=win.destroy).pack(pady=(0, 14))

        if not (os.path.exists(SAMPLE) and os.path.exists(MODEL)):
            tk.Label(preview, text="Preview not available: sample.jpg must be next to this script.").pack()
            return

        # Before/after preview on the sample photo, updated live as the settings change.
        sample_rgb = np.array(Image.open(SAMPLE).convert("RGB"))
        detector = cv2.FaceDetectorYN.create(MODEL, "", (320, 320), MIN_SCORE, 0.3, 5000)
        sample_faces = detect_faces(detector, sample_rgb)
        before_photo = ImageTk.PhotoImage(Image.fromarray(sample_rgb))
        before_label = tk.Label(preview, text="Before", image=before_photo, compound="top",
                                font=("Segoe UI", 10, "bold"))
        before_label.image = before_photo  # keep a reference so Tk keeps showing it
        before_label.pack(side="left", padx=(0, 6))
        after_label = tk.Label(preview, text="After", compound="top", font=("Segoe UI", 10, "bold"))
        after_label.pack(side="left", padx=(6, 0))

        def refresh(*_):
            after = sample_rgb.copy()
            for face in sample_faces:
                hide_face(after, face, mode.get(), strength.get(), area.get(), soft.get())
            photo = ImageTk.PhotoImage(Image.fromarray(after))
            after_label.configure(image=photo)
            after_label.image = photo

        for toggle in radios + [soft_box]:
            toggle.configure(command=refresh)
        for slider in sliders:
            slider.configure(command=refresh)
        refresh()

    header = tk.Frame(root)
    header.pack(fill="x", padx=12, pady=(8, 0))
    summary = tk.Label(header, font=("Segoe UI", 10), fg="#555")
    summary.pack(side="left")
    tk.Button(header, text="⚙", font=("Segoe UI Symbol", 14), relief="flat", cursor="hand2",
              command=open_settings).pack(side="right")

    def update_summary(*_):
        edges = "Soft round edges" if soft.get() else "Square edges"
        summary.configure(text=f"Mode: {mode.get().capitalize()}   Strength: {strength.get()}/10   "
                               f"Area: {area.get()}/10   {edges}")

    for var in (mode, strength, area, soft):
        var.trace_add("write", update_summary)
    update_summary()

    drop_zone = tk.Label(root, text="Drag a folder or photos here", font=("Segoe UI", 16),
                         relief="groove", bd=3, height=5, bg="#eef3fb")
    drop_zone.pack(fill="x", padx=12, pady=(4, 6))
    log_box = scrolledtext.ScrolledText(root, font=("Consolas", 10), state="disabled")
    log_box.pack(fill="both", expand=True, padx=12, pady=(6, 12))

    def write(msg):
        log_box.configure(state="normal")
        log_box.insert("end", msg + "\n")
        log_box.see("end")
        log_box.configure(state="disabled")

    def poll_messages():
        while not messages.empty():
            write(messages.get_nowait())
        root.after(100, poll_messages)

    def run(images, settings):
        try:
            process_images(images, *settings, messages.put)
        except Exception as e:
            messages.put(f"Error: {e}")
        finally:
            busy.clear()

    def on_drop(event):
        if busy.is_set():
            write("Still working on the last drop, please wait.")
            return
        images = find_images(os.path.normpath(p) for p in root.tk.splitlist(event.data))
        if not images:
            write("No images found. Drag a folder of photos, or select photos and drag them here.")
            return
        busy.set()
        settings = (mode.get(), strength.get(), area.get(), soft.get())
        edges = "soft edges" if soft.get() else "square edges"
        write(f"\nProcessing {len(images)} image(s): {mode.get()}, strength {strength.get()}, "
              f"area {area.get()}, {edges}...")
        threading.Thread(target=run, args=(images, settings), daemon=True).start()

    for widget in (drop_zone, log_box):
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind("<<Drop>>", on_drop)

    if not os.path.exists(MODEL):
        write(f"Face model file is missing. It must be next to this script:\n{MODEL}")
    poll_messages()
    root.mainloop()


if __name__ == "__main__":
    main()
