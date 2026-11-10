import argparse

import cv2
import numpy as np
import pandas as pd

import config as C

def segment_ball(frame):        # dtects yellow ball in image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, C.HSV_LOW, C.HSV_HIGH)
    k = np.ones((C.MORPH_KERNEL, C.MORPH_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= C.MIN_BLOB_AREA]
    if not contours:
        return None, mask
    return max(contours, key=cv2.contourArea), mask


def measure(frame, contour):      # calculates balls properties from the contour
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    (cx, cy), r = cv2.minEnclosingCircle(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    (_, _), axes, _ = cv2.fitEllipse(contour)      # needs >= 5 contour points
    return dict(
        cx=cx, cy=cy, w=w, h=h, area=area,
        aspect=h / w,                              # scale-invariant squash measure
        elong=min(axes) / max(axes),               # minor/major axis: 1 = round, rotation-invariant
        fill=area / (np.pi * r * r),               # 1.0 = perfect disc
        solidity=area / hull_area if hull_area > 0 else np.nan,   # <1 = dents
    ), (x, y, w, h)


def contact_sheet(frames, idxs, out_path, cols=6, width=320):    # creates a cooolection of video frames in a single image
    tiles = []
    for f, i in zip(frames, idxs):
        s = width / f.shape[1]
        t = cv2.resize(f, None, fx=s, fy=s)
        cv2.putText(t, str(i), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    cv2.imwrite(str(out_path), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="video name, e.g. train or test")
    ap.add_argument("--show", action="store_true", help="show live tracking window")
    args = ap.parse_args()

    src = C.video_path(args.name)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{src.name}: {n_total} frames, {fps:.2f} fps, {W0}x{H0}")
    scale = min(1.0, C.PROC_WIDTH / W0)            # downscale 4K etc. for speed
    W, H = int(round(W0 * scale)), int(round(H0 * scale))

    C.TRACK_DIR.mkdir(parents=True, exist_ok=True)
    overlay_path = C.TRACK_DIR / f"{args.name}_overlay.mp4"
    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    sheet_every = max(1, n_total // 36)
    sheet_frames, sheet_idx = [], []
    rows, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        contour, _ = segment_ball(frame)
        row = dict(frame=i, t=i / fps)
        vis = frame.copy()
        if contour is None:
            row.update(dict.fromkeys(["cx", "cy", "w", "h", "area", "aspect", "elong",
                                      "fill", "solidity"], np.nan))
        else:
            m, (x, y, w, h) = measure(frame, contour)
            row.update(m)
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(vis, f"elong={m['elong']:.3f} aspect={m['aspect']:.3f}",
                        (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"frame {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        rows.append(row)
        writer.write(vis)
        if i % sheet_every == 0:
            sheet_frames.append(vis)
            sheet_idx.append(i)
        if args.show:
            cv2.imshow("track", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        i += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(rows)
    missing = df["aspect"].isna().sum()
    for col in ("aspect", "elong"):
        df[col] = df[col].interpolate(limit_direction="both")   # fill short dropouts
    out = C.track_path(args.name)
    df.to_csv(out, index=False)
    contact_sheet(sheet_frames, sheet_idx, C.TRACK_DIR / f"{args.name}_frames.jpg")

    print(f"wrote {out}  ({len(df)} rows, {missing} frames without detection)")
    print(f"wrote {overlay_path}")
    print(f"wrote {C.TRACK_DIR / (args.name + '_frames.jpg')}  <- pick RELEASE_FRAME from this")


if __name__ == "__main__":
    main()
