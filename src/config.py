"""Shared settings for every script. Edit values here, not inside the scripts."""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "videos"
TRACK_DIR = ROOT / "data" / "tracks"
RESULTS_DIR = ROOT / "results"
PARAMS_FILE = RESULTS_DIR / "params.json"


def video_path(name: str) -> Path:
    return VIDEO_DIR / f"press_{name}.mp4"


def track_path(name: str) -> Path:
    return TRACK_DIR / f"{name}.csv"


# ---------tracking -----------

# HSV range for the yellow stress ball (OpenCV hue is 0-179).
# Skin is roughly hue 0-17, so a lower bound of 18 keeps the hand out of the mask.
HSV_LOW = (18, 90, 90)
HSV_HIGH = (40, 255, 255)
MORPH_KERNEL = 5          # px, opening kernel to remove speckle
MIN_BLOB_AREA = 2000      # px^2, smaller blobs are ignored
PROC_WIDTH = 1280         # frames are downscaled to this width before tracking (4K is slow)

# ---------------------------------------------------------------- signal definition
# Which CSV column is the observable shape signal.
# 'elong' = minor / major axis of an ellipse fitted to the ball contour.
#   - scale-invariant  -> camera distance / zoom does not matter
#   - rotation-invariant -> works for side-on AND top-down views and any squeeze direction
#   1.0 = round ball, < 1 = squeezed.   ('aspect' = bbox h/w is also in the CSV.)
SIGNAL = "elong"
REST_TAIL_FRAMES = 15     # rest shape = median of signal over the last N frames

# Release frame per video = end of the hold, where the fingers start to open.
# None -> auto: last frame where deformation >= 90 % of its peak (dynamics.guess_release).
# Set an integer to override (check data/tracks/<name>_overlay.mp4 / _frames.jpg).
# Note: the fingers open over ~0.3 s while the ball springs back, so the fitted k, c are
# EFFECTIVE parameters of "ball + letting go". Starting at thumb-off instead (train 265,
# test 226) leaves only ~5 % deformation to fit, too little signal.
RELEASE_FRAME = {
    "train": None,
    "test": None,
}

# ---------------------------------------------------------------- dynamics
MASS = 1.0                # only k/m and c/m are identifiable from video; fix m = 1
SUBSTEPS = 10             # Euler sub-steps per video frame (stability)

# search range for the fit (units: 1/s^2 for k, 1/s for c, with m=1)
K_BOUNDS = (1e-2, 1e4)
C_BOUNDS = (1e-3, 1e3)
