
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import config as C
import dynamics as D


def loss(log_params, x_obs, fps, x0, v0):
    k, c = np.exp(log_params)          # optimise in log-space -> k, c stay positive
    x_sim = D.simulate(k, c, x0, v0, len(x_obs), fps)
    return float(np.mean((x_sim - x_obs) ** 2))


def fit(x_obs, fps):
    x0, v0 = D.initial_state(x_obs)
    # The loss surface has long flat valleys, so gradient methods (L-BFGS) get stuck.
    # 1) coarse log-spaced grid search to find the right basin
    ks = np.logspace(*np.log10(C.K_BOUNDS), 40)
    cs = np.logspace(*np.log10(C.C_BOUNDS), 40)
    grid = [(loss(np.log([k, c]), x_obs, fps, x0, v0), k, c) for k in ks for c in cs]
    _, k_init, c_init = min(grid)
    # 2) refine with derivative-free Nelder-Mead (scipy.optimize.minimize)
    best = minimize(loss, np.log([k_init, c_init]), args=(x_obs, fps, x0, v0),
                    method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-10, maxiter=2000))
    k, c = np.clip(np.exp(best.x), [C.K_BOUNDS[0], C.C_BOUNDS[0]], [C.K_BOUNDS[1], C.C_BOUNDS[1]])
    return k, c, x0, v0, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="train")
    args = ap.parse_args()

    df, rest, fps = D.load_track(args.name)
    x_all = df["x"].to_numpy()
    r = D.release_frame(args.name, x_all)
    x_obs = x_all[r:]
    print(f"rest {C.SIGNAL} = {rest:.4f}, release frame = {r} "
          f"({'config' if C.RELEASE_FRAME.get(args.name) is not None else 'auto-guessed'}), "
          f"{len(x_obs)} frames of free recovery")

    k, c, x0, v0, res = fit(x_obs, fps)
    x_sim = D.simulate(k, c, x0, v0, len(x_obs), fps)
    rmse = float(np.sqrt(np.mean((x_sim - x_obs) ** 2)))
    ss_res = np.sum((x_obs - x_sim) ** 2)
    ss_tot = np.sum((x_obs - x_obs.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)

    info = D.describe(k, c)
    params = dict(**info, x0=x0, v0=v0, fps=fps, rest_signal=rest, signal=C.SIGNAL,
                  release_frame=r, rmse=rmse, r2=r2, train_video=args.name,
                  optimizer_success=bool(res.success))
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.PARAMS_FILE.write_text(json.dumps(params, indent=2))

    print(f"k = {k:.3f} 1/s^2   c = {c:.3f} 1/s   (m = {C.MASS})")
    print(f"omega_n = {info['omega_n']:.3f} rad/s   zeta = {info['zeta']:.3f}  -> {info['regime']}")
    print(f"train RMSE = {rmse:.5f} strain   R^2 = {r2:.3f}")
    print(f"wrote {C.PARAMS_FILE}")

    # ---- plot
    t_all = df["t"].to_numpy()
    t = t_all[r:]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t_all, x_all, ".", color="0.7", ms=4, label="observed (whole video)")
    ax.plot(t, x_obs, ".", color="C0", ms=5, label="observed (free recovery, used for fit)")
    ax.plot(t, x_sim, "-", color="C3", lw=2, label=f"fitted model  k={k:.1f}, c={c:.1f}")
    ax.axvline(t_all[r], color="k", ls="--", lw=1, label="release")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(f"deformation x  (from {C.SIGNAL})")
    ax.set_title(f"Fit on '{args.name}':  zeta={info['zeta']:.2f}, RMSE={rmse:.4f}, R²={r2:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = C.RESULTS_DIR / "fit_plot.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
