"""Step 6 - Demo video: real ball (left) | PyBullet box proxy (right) | curves (bottom).

Usage:
    python src/make_demo.py               # on the fresh 'test' press
    python src/make_demo.py --name train  # on the training press
    python src/make_demo.py --gain 3      # exaggerate the proxy's deformation 3x (display only)

Before release the proxy follows the tracked deformation (hand in contact, force unknown).
From the release frame on, the proxy is driven ONLY by the fitted spring-damper model.

Output: results/demo_<name>.mp4
"""
import argparse
import json

import cv2
import numpy as np

import config as C
import dynamics as D
import proxy as P

PANEL_W, PANEL_H, STRIP_H = 640, 360, 220


def draw_strip(t, x_obs, x_pred, r, i, y_range):
    """Small time-series plot drawn with OpenCV, with a cursor at frame i."""
    img = np.full((STRIP_H, 2 * PANEL_W, 3), 255, np.uint8)
    pad = 40
    W, H = img.shape[1] - 2 * pad, STRIP_H - 2 * pad
    lo, hi = y_range

    def pts(ts, xs):
        u = pad + (ts - t[0]) / (t[-1] - t[0]) * W
        v = pad + (hi - xs) / (hi - lo) * H
        return np.stack([u, v], 1).astype(np.int32)

    zero = pts(np.array([t[0], t[-1]]), np.zeros(2))
    cv2.line(img, tuple(zero[0]), tuple(zero[1]), (180, 180, 180), 1)
    for u, v in pts(t, x_obs):
        cv2.circle(img, (u, v), 2, (200, 120, 30), -1)               # observed (blue)
    cv2.polylines(img, [pts(t[r:], x_pred[r:])], False, (40, 40, 220), 2)  # model (red)
    ur = pts(t[r:r + 1], np.zeros(1))[0][0]
    cv2.line(img, (ur, pad), (ur, pad + H), (0, 0, 0), 1)
    ui = pts(t[i:i + 1], np.zeros(1))[0][0]
    cv2.line(img, (ui, pad), (ui, pad + H), (0, 160, 0), 2)
    cv2.putText(img, "deformation x:  blue = tracked from video,  red = model prediction after release",
                (pad, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return img


def label(img, text, color=(255, 255, 255)):
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="test")
    ap.add_argument("--gain", type=float, default=1.0, help="visual exaggeration of the proxy")
    args = ap.parse_args()

    prm = json.loads(C.PARAMS_FILE.read_text())
    df, rest, fps = D.load_track(args.name)
    t, x_obs = df["t"].to_numpy(), df["x"].to_numpy()
    r = D.release_frame(args.name, x_obs)
    x0, v0 = D.initial_state(x_obs[r:])
    x_pred = x_obs.copy()
    x_pred[r:] = D.simulate(prm["k"], prm["c"], x0, v0, len(x_obs) - r, fps)
    y_range = (min(x_obs.min(), x_pred.min()) - 0.01, max(x_obs.max(), x_pred.max()) + 0.01)

    P.VIS_GAIN = args.gain
    scene = P.ProxyScene(gui=False, width=PANEL_W, height=PANEL_H)
    cap = cv2.VideoCapture(str(C.video_path(args.name)))
    out_path = C.RESULTS_DIR / f"demo_{args.name}.mp4"
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (2 * PANEL_W, PANEL_H + STRIP_H))

    for i in range(len(df)):
        ok, frame = cap.read()
        if not ok:
            break
        real = label(cv2.resize(frame, (PANEL_W, PANEL_H)), f"real video  (frame {i})")
        scene.set_deformation(x_pred[i])
        sim = cv2.cvtColor(scene.render(), cv2.COLOR_RGB2BGR)
        if i < r:
            txt, col = "proxy: follows tracking (hand in contact)", (200, 200, 200)
        else:
            txt, col = f"proxy: MODEL PREDICTION  k={prm['k']:.2f} c={prm['c']:.2f}", (80, 255, 80)
        if args.gain != 1.0:
            txt += f"  [x{args.gain:g} visual]"
        sim = label(sim, txt, col)
        frame_out = np.vstack([np.hstack([real, sim]), draw_strip(t, x_obs, x_pred, r, i, y_range)])
        out.write(frame_out)
        if i % 50 == 0:
            print(f"  frame {i}/{len(df)}")

    cap.release()
    out.release()
    scene.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
