
import argparse
import json
import time

import numpy as np
import pybullet as p
import pybullet_data

import config as C

BALL_SIZE = 0.07     # m, rest edge length of the proxy cube (~ stress-ball diameter)
VIS_GAIN = 1.0       # >1 exaggerates deformation for display only (the physics is unchanged)

# 8 corners of a unit cube centred on x/y, sitting on z = 0
_UNIT = np.array([[sx, sy, sz] for sz in (0, 1) for sy in (-.5, .5) for sx in (-.5, .5)])
# 12 triangles (2 per face), counter-clockwise seen from outside
_FACES = [0, 2, 1, 1, 2, 3,   4, 5, 6, 5, 7, 6,   0, 1, 4, 1, 5, 4,
          2, 6, 3, 3, 6, 7,   0, 4, 2, 2, 4, 6,   1, 3, 5, 3, 7, 5]


def box_corners(x, size=BALL_SIZE, gain=VIS_GAIN):
    """Return the 8 deformed corner positions (8, 3) for deformation x."""
    xs = float(np.clip(x * gain, -0.9, 0.9))
    sz = (1.0 - xs) ** (2.0 / 3.0)
    sxy = 1.0 / np.sqrt(sz)
    return _UNIT * size * np.array([sxy, sxy, sz])


class ProxyScene:
    """Owns a PyBullet client with a ground plane and the deformable box proxy."""

    def __init__(self, gui=False, width=640, height=480):
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            p.resetDebugVisualizerCamera(0.25, 35, -20, [0, 0, 0.035], physicsClientId=self.client)
        self.body = None
        self.w, self.h = width, height
        self.view = p.computeViewMatrixFromYawPitchRoll([0, 0, 0.035], 0.25, 35, -20, 0, 2)
        self.proj = p.computeProjectionMatrixFOV(45, width / height, 0.01, 2.0)

    def set_deformation(self, x):
        """Rebuild the box mesh with the corners for state x.
        (PyBullet cannot edit rigid mesh vertices in place, so we recreate the body;
        it is visual-only, so this is cheap.)"""
        if self.body is not None:
            p.removeBody(self.body, physicsClientId=self.client)
        verts = box_corners(x).tolist()
        vis = p.createVisualShape(p.GEOM_MESH, vertices=verts, indices=_FACES,
                                  rgbaColor=[1.0, 0.85, 0.1, 1], physicsClientId=self.client)
        self.body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                      physicsClientId=self.client)
        return verts

    def render(self):
        """RGB image (h, w, 3) uint8 of the scene, works headless."""
        img = p.getCameraImage(self.w, self.h, self.view, self.proj,
                               renderer=p.ER_TINY_RENDERER, physicsClientId=self.client)
        return np.reshape(img[2], (self.h, self.w, 4))[:, :, :3].astype(np.uint8)

    def close(self):
        p.disconnect(self.client)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", action="store_true", help="headless, save a preview image")
    ap.add_argument("--x0", type=float, default=None, help="initial deformation (default: fitted x0)")
    args = ap.parse_args()

    import dynamics as D
    prm = json.loads(C.PARAMS_FILE.read_text())
    x0 = args.x0 if args.x0 is not None else prm["x0"]
    fps = prm["fps"]
    xs = D.simulate(prm["k"], prm["c"], x0, 0.0, int(8 * fps), fps)
    print(f"k={prm['k']:.3f} c={prm['c']:.3f}  x0={x0:.3f}  -> playing {len(xs)} frames")

    scene = ProxyScene(gui=not args.direct)
    if args.direct:
        import cv2
        tiles = []
        for x in (x0, xs[len(xs) // 2], xs[-1]):
            scene.set_deformation(x)
            img = cv2.cvtColor(scene.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(img, f"x={x:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            tiles.append(img)
        out = C.RESULTS_DIR / "proxy_preview.png"
        cv2.imwrite(str(out), np.hstack(tiles))
        print(f"wrote {out}")
    else:
        for x in xs:
            scene.set_deformation(x)
            time.sleep(1.0 / fps)
        input("done - press Enter to close")
    scene.close()


if __name__ == "__main__":
    main()
