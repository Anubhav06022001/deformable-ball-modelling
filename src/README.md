# src/ — code

Run every script from the **repository root** (`python src/<script>.py ...`), in this order:

| # | File | What it does | Reads | Writes |
|---|------|--------------|-------|--------|
| 0 | `config.py` | All settings: paths, HSV colour range, signal choice, release frames, fit bounds. Not run directly. | – | – |
| 1 | `track.py` | OpenCV: segments the yellow ball in every frame (HSV threshold → largest contour) and measures bounding box, area, aspect, ellipse elongation (minor/major), fill, solidity. Frames are downscaled to 1280 px wide. | `data/videos/press_<name>.mp4` | `data/tracks/<name>.csv`, `<name>_overlay.mp4`, `<name>_frames.jpg` |
| 2 | `dynamics.py` | 1-DOF spring-mass-damper `m x'' = F − k x − c x'`, explicit Euler `v += (F/m)dt; x += v dt` with automatic sub-stepping for stability. Also loads a track and converts it to deformation `x`. `python src/dynamics.py` plots example responses. | track CSV | – |
| 3 | `fit.py` | Fits `k`, `c` on the free-recovery part (after release) of the training video: log-grid search, then `scipy.optimize.minimize` (Nelder-Mead) on the mean-squared error. | `data/tracks/train.csv` | `results/params.json`, `results/fit_plot.png` |
| 4 | `proxy.py` | PyBullet 3D proxy: an 8-corner box mesh deformed volume-preservingly by `x`. `python src/proxy.py` plays the fitted recovery in the GUI; `--direct` saves a preview image. | `results/params.json` | `results/proxy_preview.png` |
| 5 | `validate.py` | Predicts the recovery of a **new, unseen press** with the fitted `k`, `c` and compares it with the video against two baselines. | `data/tracks/test.csv`, `results/params.json` | `results/validation.json`, `results/validation_plot.png` |
| 6 | `make_demo.py` | Side-by-side video: real ball, PyBullet proxy, and the observed and predicted curves. | video + track + params | `results/demo_<name>.mp4` |

## Full run

```bash
python src/track.py train
python src/fit.py
python src/proxy.py                      # optional GUI check
python src/track.py test
python src/validate.py
python src/make_demo.py --name test
python src/make_demo.py --name train
```

## Deformation signal

```
elong = minor_axis / major_axis of cv2.fitEllipse(contour)   (scale- and rotation-invariant)
x     = (elong_rest − elong) / elong_rest                     (elong_rest = median of the last 15 frames)
```

`x > 0` means squeezed and `x = 0` means rest. The release frame is the end of the hold (the last frame where x ≥ 90 % of its peak). It can be overridden in `config.RELEASE_FRAME`. With mass fixed at `m = 1`, only `k/m` and `c/m` can be identified.
