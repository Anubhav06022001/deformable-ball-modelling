"""Step 2 - Spring-mass-damper model of the ball's shape, integrated with explicit Euler.

State x = deformation (strain) of the ball relative to its rest shape:
    x = (aspect_rest - aspect) / aspect_rest       x > 0 : squashed, x = 0 : rest

Equation of motion (eq. 15.5 style, one degree of freedom):
    m x'' = F_ext - k x - c x'
Euler step (sub-stepped for stability):
    a = (F_ext - k*x - c*v) / m
    v += a * dt
    x += v * dt

Can be run on its own as a quick demo:
    python src/dynamics.py
"""
import numpy as np
import pandas as pd

import config as C


# ------------------------------------------------------------------ simulator
def simulate(k, c, x0, v0, n_frames, fps, m=C.MASS, substeps=C.SUBSTEPS, f_ext=None):
    """Return x at each video frame (length n_frames), starting from (x0, v0).

    f_ext: optional array (n_frames,) of external force per frame (default 0 = free recovery).
    """
    # explicit Euler is only stable if dt is small vs. both time scales (c/m and sqrt(k/m)),
    # so increase sub-steps automatically when the optimiser tries stiff/heavily-damped values
    need = int(np.ceil((c / m + np.sqrt(k / m)) / fps))
    substeps = max(substeps, need)
    dt = 1.0 / (fps * substeps)
    x, v = float(x0), float(v0)
    out = np.empty(n_frames)
    for i in range(n_frames):
        out[i] = x
        F = 0.0 if f_ext is None else f_ext[i]
        for _ in range(substeps):
            a = (F - k * x - c * v) / m
            v += a * dt
            x += v * dt
    return out


# ------------------------------------------------------------------ observation helpers
def load_track(name):
    """Load the CSV from track.py and add the deformation column 'x'."""
    df = pd.read_csv(C.track_path(name))
    sig = df[C.SIGNAL].to_numpy()
    rest = float(np.median(sig[-C.REST_TAIL_FRAMES:]))
    df["x"] = (rest - sig) / rest
    fps = 1.0 / float(df["t"].iloc[1] - df["t"].iloc[0])
    return df, rest, fps


def guess_release(x, frac=0.9, win=5):
    """Guess the release frame = END OF THE HOLD: the last frame where the
    (median-smoothed) deformation is still >= frac * its peak. After this the
    fingers open and the ball springs back. Override with config.RELEASE_FRAME."""
    xs = pd.Series(x).rolling(win, center=True, min_periods=1).median().to_numpy()
    return int(np.where(xs >= frac * np.nanmax(xs))[0][-1])


def release_frame(name, x):
    r = C.RELEASE_FRAME.get(name)
    return int(r) if r is not None else guess_release(x)


def initial_state(x_obs, n=3):
    """x0 = mean of the first n observed frames after release (reduces pixel noise);
    v0 = 0 (the hand holds the ball still just before letting go)."""
    return float(np.mean(x_obs[:n])), 0.0


# ------------------------------------------------------------------ analytic info
def describe(k, c, m=C.MASS):
    wn = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(k * m))
    regime = "under-damped" if zeta < 1 else ("critically damped" if np.isclose(zeta, 1) else "over-damped")
    return dict(k=k, c=c, m=m, omega_n=wn, zeta=zeta, regime=regime)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t = np.arange(120) / 24
    for k, c in [(50, 4), (50, 14), (50, 40)]:
        info = describe(k, c)
        plt.plot(t, simulate(k, c, 0.05, 0.0, 120, 24), label=f"k={k} c={c} ({info['regime']})")
    plt.xlabel("time [s]"); plt.ylabel("deformation x"); plt.legend(); plt.title("Euler spring-mass-damper")
    plt.show()
