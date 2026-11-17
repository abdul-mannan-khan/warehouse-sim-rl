"""Dual-arm picking module mounted on each AMR (numpy only, kinematic).

A mast stands on the rear deck of the iw.hub. A torso rides up and down the mast (lift) and turns about it
(waist); two UR5e arms with Robotiq 2F-85 grippers are mounted on the torso's left and right sides, base axes
pointing outwards. To pick, the torso turns to the rack and rises to the shelf, the hands come in on either side
of the parcel and squeeze it between the closed finger pads, lift it a little, pull it back to the chest, and the
torso turns forward and sinks so the hands can set the parcel down on the robot's own tray.

Frames: robot frame (x forward along the robot, y left, z up from the robot root prim); torso frame (origin on the
mast axis at shoulder height, x the facing direction). Arm IK is arm.ik in each arm's base frame.
"""
import math

import numpy as np

import arm as armk

MAST_X = -0.62                  # mast on the rear deck, behind the tray (robot frame)
MAST_TOP = 1.80
SHOULDER_Y = 0.21               # arm bases either side of the torso
H_MIN, H_PARK = 0.42, 0.90      # lowest shoulder height (torso clears the deck) and travel height
PSI_PARK = -math.pi / 2         # travels facing its right side (the rack), elbows spread along the robot
C_BELOW = 0.13                  # parcel centre below shoulder height while working
HAND_GAP = 0.02                 # tcp to parcel face when the pads touch it
SPREAD = 0.07                   # extra clearance of the open hands
HAND_DX = -0.03                 # hands grip the front part of the parcel's sides
CLOSED = math.radians(40.0)     # finger angle: pads together, used as a flat palm


MOUNT_TILT = math.radians(45.0)  # arm base axes point outwards and 45 deg down, so the elbows hang out and down


def _mount(side):
    """Arm base pose in the torso frame; side +1 left, -1 right. Base z points outwards and down, base x forwards."""
    T = np.eye(4)
    z = np.array([0.0, side * math.cos(MOUNT_TILT), -math.sin(MOUNT_TILT)])
    x = np.array([1.0, 0.0, 0.0])
    T[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    T[:3, 3] = [0.0, side * SHOULDER_Y, 0.0]
    return T


MOUNT = {+1: _mount(+1), -1: _mount(-1)}
MOUNT_INV = {k: np.linalg.inv(v) for k, v in MOUNT.items()}


def torso_pose(psi, h):
    T = armk.rz(psi)
    T[:3, 3] = [MAST_X, 0.0, h]
    return T


def hand_target(c, side, spread):
    """Tcp pose (torso frame) of one hand pressing on the parcel side; tool axis inward, jaw axis vertical."""
    T = np.eye(4)
    z = np.array([0.0, -side, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    T[:3, :3] = np.stack([np.cross(y, z), y, z], axis=1)
    T[:3, 3] = np.asarray(c) + [HAND_DX, side * (CART_W / 2 + HAND_GAP + spread), 0.0]
    return T


CART_W = 0.21                   # parcel width between the hands (set by the replay from its carton size)
READY = np.array([0.36, 0.0, -C_BELOW])
SEED = {+1: np.radians([-33.7, -61.7, 122.2, 58.5, 54.0, -43.3]),      # mirror-image postures, elbows out
        -1: np.radians([-146.3, -118.3, -122.2, 121.5, -54.0, -136.7])}   # and down (searched offline)


N_SAMPLES = 16
_READY_Q = {}


def ready_q(side):
    """Joints of one hand at the ready pose, solved once from the offline seed."""
    if side not in _READY_Q:
        q, res = armk.ik(MOUNT_INV[side] @ hand_target(READY, side, SPREAD), SEED[side], iters=300)
        if res > 1e-4:
            raise ValueError(f"picker hand {side}: ready pose not reached ({res:.5f})")
        _READY_Q[side] = q
    return _READY_Q[side]


def mjk(u):
    return armk.minjerk(u)


# (name, start fraction, end fraction) of one pick; see Pick.at
PHASES = (("rise", 0.00, 0.14), ("reach", 0.14, 0.26), ("squeeze", 0.26, 0.35), ("lift", 0.35, 0.40),
          ("pull", 0.40, 0.50), ("turn", 0.50, 0.64), ("over", 0.64, 0.76), ("place", 0.76, 0.82),
          ("release", 0.82, 0.86), ("back", 0.86, 1.00))


class Pick:
    """One shelf-to-tray pick. carton, slot: parcel centres in the robot frame."""

    def __init__(self, carton, slot):
        carton, slot = np.asarray(carton, float), np.asarray(slot, float)
        self.torso = {"park": (PSI_PARK, H_PARK),
                      "shelf": (-math.pi / 2, carton[2] + C_BELOW),
                      "tray": (0.0, max(slot[2] + C_BELOW, H_MIN))}
        Ts, Tt = torso_pose(*self.torso["shelf"]), torso_pose(*self.torso["tray"])
        cs = (np.linalg.inv(Ts) @ np.append(carton, 1.0))[:3]
        ct = (np.linalg.inv(Tt) @ np.append(slot, 1.0))[:3]
        self.carton_rf, self.slot_rf = carton, slot
        # hand keys: (parcel centre in the torso frame, spread)
        self.keys = {"ready": (READY, SPREAD), "pre": (cs, SPREAD), "grasp": (cs, 0.0),
                     "lift": (cs + [0, 0, 0.025], 0.0), "carry": (READY + [0, 0, 0.03], 0.0),
                     "over": (ct + [0, 0, 0.05], 0.0), "place": (ct, 0.0), "release": (ct, SPREAD)}
        # each hand move is sampled densely along its Cartesian path, every sample's IK seeded by the previous
        # one (continuation), so the arms stay on one smooth posture; frames interpolate the joint samples
        self.paths = {}
        q = {side: ready_q(side) for side in (+1, -1)}
        for k0, k1 in (("ready", "pre"), ("pre", "grasp"), ("grasp", "lift"), ("lift", "carry"), ("carry", "over"),
                       ("over", "place"), ("place", "release"), ("release", "ready")):
            (c0, s0), (c1, s1) = self.keys[k0], self.keys[k1]
            samples = {}
            for side in (+1, -1):
                qs, qq = [q[side]], q[side]
                for w in np.linspace(0, 1, N_SAMPLES + 1)[1:]:
                    G = MOUNT_INV[side] @ hand_target(c0 + (c1 - c0) * w, side, s0 + (s1 - s0) * w)
                    qn, res = armk.ik(G, qq, iters=100)
                    if res > 2e-3:
                        raise ValueError(f"picker hand {side} lost the path {k0}->{k1}: residual {res:.4f}")
                    qq = qq + (qn - qq + math.pi) % (2 * math.pi) - math.pi        # unwrap next to the last sample
                    qs.append(qq)
                samples[side] = np.array(qs)
                q[side] = qq
            self.paths[(k0, k1)] = samples
        self.loop_err = max(float(np.abs(q[s] - ready_q(s)).max()) for s in (+1, -1))

    def hands(self, k0, k1, u):
        """Parcel centre (torso frame) and both arms' joints at fraction u of the move k0 -> k1."""
        (c0, _), (c1, _) = self.keys[k0], self.keys[k1]
        w = mjk(u)
        c = c0 + (c1 - c0) * w
        if k0 == k1:
            if k0 == "ready":
                return c, {side: ready_q(side) for side in (+1, -1)}
            return c, {side: self._hold(k0, side) for side in (+1, -1)}
        x = w * N_SAMPLES
        i = min(int(x), N_SAMPLES - 1)
        f = x - i
        P = self.paths[(k0, k1)]
        return c, {side: P[side][i] + (P[side][i + 1] - P[side][i]) * f for side in (+1, -1)}

    def _hold(self, k, side):
        for (a, b), P in self.paths.items():
            if b == k:
                return P[side][-1]
        raise KeyError(k)

    def at(self, u):
        """Fraction u of the pick -> dict(psi, h, q {side: joints}, parcel: 'shelf' | 'held' | 'placed',
        parcel_tf: parcel centre in the torso frame while held)."""
        u = min(max(u, 0.0), 1.0)
        for name, a, b in PHASES:
            if u <= b or name == "back":
                v = (u - a) / (b - a)
                break
        tp = self.torso
        torso = tp["shelf"]
        parcel = "held"
        if name == "rise":
            torso, (c, q) = _lerp_t(tp["park"], tp["shelf"], v), self.hands("ready", "ready", 0.0)
            parcel = "shelf"
        elif name == "reach":
            c, q = self.hands("ready", "pre", v)
            parcel = "shelf"
        elif name == "squeeze":
            c, q = self.hands("pre", "grasp", v)
            parcel = "shelf"
        elif name == "lift":
            c, q = self.hands("grasp", "lift", v)
        elif name == "pull":
            c, q = self.hands("lift", "carry", v)
        elif name == "turn":
            torso, (c, q) = _lerp_t(tp["shelf"], tp["tray"], v), self.hands("carry", "carry", 0.0)
        elif name == "over":
            torso, (c, q) = tp["tray"], self.hands("carry", "over", v)
        elif name == "place":
            torso, (c, q) = tp["tray"], self.hands("over", "place", v)
        elif name == "release":
            torso, (c, q) = tp["tray"], self.hands("place", "release", v)
            parcel = "placed"
        else:
            torso, (c, q) = _lerp_t(tp["tray"], tp["park"], v), self.hands("release", "ready", v)
            parcel = "placed"
        return {"psi": torso[0], "h": torso[1], "q": q, "parcel": parcel, "c": c}


def _lerp_t(a, b, v):
    w = mjk(v)
    return (a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w)


def park():
    """Travel pose: torso facing the robot's right side, hands open in front of the chest."""
    return {"psi": PSI_PARK, "h": H_PARK, "q": {side: ready_q(side) for side in (+1, -1)}, "parcel": None, "c": READY}
