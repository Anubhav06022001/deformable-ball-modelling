# data/videos/ — input recordings

| File | Role |
|------|------|
| `press_train.mp4` | Press → release → recovery. **Used to fit** `k` and `c`. |
| `press_test.mp4` | A second, separate press of the same ball. **Never used for fitting.** It is the "fresh instance" for validation. |

Names must follow `press_<name>.mp4`; `<name>` is the argument passed to the scripts.

## Capture guidelines (for the test video, ideally both)

- **Phone on a tripod or propped up, side-on, ball on a table. Do not move or zoom the camera.**
- A plain, non-yellow background. The tracker thresholds yellow (`HSV_LOW/HIGH` in `src/config.py`).
- Press from the **top** with one or two fingers, hold for about 1 s, then lift the hand **straight up and away** so it stops covering the ball.
- **Keep recording for at least 10 s after release**, until the ball has clearly stopped changing.
- Use 30 or 60 fps if available. Good lighting reduces motion blur.

| File | Resolution | fps | Frames | Duration |
|------|-----------|-----|--------|----------|
| `press_train.mp4` | 3840×2160 | 60 | 454 | 7.6 s (side-on view) |
| `press_test.mp4` | 3840×2160 | 60 | 361 | 6.0 s (top-down view) |
