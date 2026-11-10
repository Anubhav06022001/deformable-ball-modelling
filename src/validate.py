import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
import dynamics as D


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="test", help="tracked video to validate on")
    args = ap.parse_args()

    if not C.PARAMS_FILE.exists():
        raise SystemExit("results/params.json missing - run src/fit.py first")
    if not C.track_path(args.name).exists():
        raise SystemExit(f"{C.track_path(args.name)} missing - run: python src/track.py {args.name}")
    prm = json.loads(C.PARAMS_FILE.read_text())
    if args.name == prm["train_video"]:
        print("WARNING: validating on the training video - this is NOT a fresh instance.")

    df, rest, fps = D.load_track(args.name)
    x_all = df["x"].to_numpy()
    r = D.release_frame(args.name, x_all)
    x_obs = x_all[r:]
    x0, v0 = D.initial_state(x_obs)
    x_sim = D.simulate(prm["k"], prm["c"], x0, v0, len(x_obs), fps)

    e_model = rmse(x_sim, x_obs)
    e_stay = rmse(np.full_like(x_obs, x0), x_obs)
    e_instant = rmse(np.zeros_like(x_obs), x_obs)
    r2 = float(1 - np.sum((x_obs - x_sim) ** 2) / np.sum((x_obs - x_obs.mean()) ** 2))

    # time for the deformation to fall below 50 % of x0: observed vs. predicted
    def half_time(x):
        idx = np.where(np.abs(x) < 0.5 * abs(x0))[0]
        return float(idx[0] / fps) if len(idx) else float("nan")

    out = dict(video=args.name, release_frame=r, x0=x0, k=prm["k"], c=prm["c"],
               rmse_model=e_model, rmse_stay=e_stay, rmse_instant=e_instant, r2=r2,
               half_time_obs_s=half_time(x_obs), half_time_sim_s=half_time(x_sim),
               train_rmse=prm["rmse"])
    (C.RESULTS_DIR / "validation.json").write_text(json.dumps(out, indent=2))

    print(f"release frame {r}, x0 = {x0:.4f}, {len(x_obs)} frames")
    print(f"RMSE  model = {e_model:.5f}   (train was {prm['rmse']:.5f})")
    print(f"RMSE  baseline 'stay' = {e_stay:.5f}   baseline 'instant' = {e_instant:.5f}")
    print(f"R^2 = {r2:.3f}")
    print(f"50% recovery time: observed {out['half_time_obs_s']:.2f}s, predicted {out['half_time_sim_s']:.2f}s")
    print(f"wrote {C.RESULTS_DIR / 'validation.json'}")

    t_all = df["t"].to_numpy()
    t = t_all[r:]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t_all, x_all, ".", color="0.7", ms=4, label="observed (whole video)")
    ax.plot(t, x_obs, ".", color="C0", ms=5, label="observed recovery (unseen)")
    ax.plot(t, x_sim, "-", color="C3", lw=2, label="predicted by fitted model")
    ax.axvline(t_all[r], color="k", ls="--", lw=1, label="release")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(f"deformation x  (from {C.SIGNAL})")
    ax.set_title(f"Validation on '{args.name}':  RMSE model {e_model:.4f} | "
                 f"stay {e_stay:.4f} | instant {e_instant:.4f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = C.RESULTS_DIR / "validation_plot.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
