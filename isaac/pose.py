"""Procedural work motions for the Isaac Sim characters: reach, grip, carry, place.

NVIDIA's motion library has walking, idling and carrying clips but no pick or place motion, so the
work is posed procedurally on top of a retargeted idle clip:

* a forward bend distributed over the three spine joints,
* two-bone IK on each arm (shoulder, elbow, wrist) towards a hand target, elbows kept down and out,
* leg rotations blended in from the walk-in-place clip while the character turns, so feet step instead
  of sliding,
* a root height correction (`last_dz`) that keeps the lowest foot on the floor in every pose.

Targets are given in the character's body frame (metres: forward, left, up from the point between the
feet), converted to skeleton space with the character's own skeleton-to-root transform.
"""
import math

import numpy as np
from pxr import Vt

from retarget import to_quat


def rot(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * k + (1 - math.cos(ang)) * k @ k


def align(u, v):
    """Smallest rotation taking direction u to direction v."""
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    c = np.cross(u, v)
    s, d = np.linalg.norm(c), float(u @ v)
    if s < 1e-9:
        return np.eye(3)
    return rot(c, math.atan2(s, d))


def m2q(m):
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return np.array(q)


def q2m(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def blend(a, b, w):
    """Normalised quaternion blend of two rotation matrices."""
    if w <= 0:
        return a
    if w >= 1:
        return b
    qa, qb = m2q(a), m2q(b)
    if qa @ qb < 0:
        qb = -qb
    return q2m(qa * (1 - w) + qb * w)


def smooth(f):
    f = min(max(f, 0.0), 1.0)
    return f * f * (3 - 2 * f)


class Poser:
    def __init__(self, rt, skel_to_root):
        """rt: Retargeter for this character; skel_to_root: Gf.Matrix4d (row-vector) skeleton -> root xform."""
        self.rt = rt
        self.par = rt.tpar
        self.trans = np.array([[float(c) for c in t] for t in rt.t_trans])
        j = {n: i for i, n in enumerate(rt.tj)}
        self.j = j
        a = np.array([[skel_to_root[r][c] for c in range(4)] for r in range(4)])
        self.A = a[:3, :3].T                       # column convention, includes scale
        self.b = a[3, :3]
        self.Ai = np.linalg.inv(self.A)
        self.spine = [j[n] for n in ("Waist", "Spine01", "Spine02") if n in j]
        self.legs = [j[f"{s}_{n}"] for s in "LR" for n in ("Thigh", "Calf", "Foot", "ToeBase") if f"{s}_{n}" in j]
        self.arm = {s: (j[f"{s}_Upperarm"], j[f"{s}_Forearm"], j[f"{s}_Hand"]) for s in "LR"}
        self.hip = j["Hip"]
        # body axes in skeleton space, from the rest pose
        L0 = np.stack([r for r, _ in rt.t_local_rest])
        G, P = self.fk(L0)
        up = P[j["Head"]] - P[j["Hip"]]
        left = P[j["L_Upperarm"]] - P[j["R_Upperarm"]]
        up = up / np.linalg.norm(up)
        left = left - up * (left @ up)
        left = left / np.linalg.norm(left)
        self.up_s, self.left_s, self.fwd_s = up, left, np.cross(left, up)
        # the same axes in the root frame (the root is the character's xform; z up)
        self.fwd_r = self._dir_to_root(self.fwd_s)
        self.left_r = self._dir_to_root(self.left_s)
        self.fwd_r[2] = self.left_r[2] = 0.0
        self.fwd_r /= np.linalg.norm(self.fwd_r)
        self.left_r /= np.linalg.norm(self.left_r)
        self.unit = 1.0 / np.linalg.norm(self.A[:, 0])           # skeleton units per metre
        th, ca, fo = (j[f"L_{n}"] for n in ("Thigh", "Calf", "Foot"))
        self.leg_len = np.linalg.norm(P[ca] - P[th]) + np.linalg.norm(P[fo] - P[ca])
        s0, e0, w0 = self.arm["L"]
        self.arm_len = (np.linalg.norm(P[e0] - P[s0]) + np.linalg.norm(P[w0] - P[e0])) / self.unit
        self.L0 = L0
        # feet: joints that touch the ground, and their height in the rest pose (the mesh stands on z=0 there)
        self.feet = [j[f"{s}_{n}"] for s in "LR" for n in ("Foot", "ToeBase") if f"{s}_{n}" in j]
        self.foot_h0 = min(self.skel_to_body(P[i])[2] for i in self.feet)
        self.last_dz = 0.0
        self._build_reach()

    # skeleton <-> root frame
    def _dir_to_root(self, v):
        d = self.A @ v
        return d / np.linalg.norm(d)

    def to_root(self, p):
        return self.A @ p + self.b

    def body_to_skel(self, f, l, h):
        p = self.fwd_r * f + self.left_r * l + np.array([0.0, 0.0, h])
        return self.Ai @ (p - self.b)

    def skel_to_body(self, p):
        r = self.to_root(p)
        return np.array([r @ self.fwd_r, r @ self.left_r, r[2]])

    def fk(self, L, T=None):
        T = self.trans if T is None else T
        n = len(L)
        G = np.zeros((n, 3, 3))
        P = np.zeros((n, 3))
        for i in range(n):
            p = self.par[i]
            if p < 0:
                G[i], P[i] = L[i], T[i]
            else:
                G[i] = G[p] @ L[i]
                P[i] = P[p] + G[p] @ T[i]
        return G, P

    def _set_global(self, L, G, i, Gi_new):
        p = self.par[i]
        L[i] = (G[p].T if p >= 0 else np.eye(3)) @ Gi_new

    def _ik(self, L, T, side, target, pole):
        s, e, w = self.arm[side]
        G, P = self.fk(L, T)
        S, E, Wr = P[s], P[e], P[w]
        a, b = np.linalg.norm(E - S), np.linalg.norm(Wr - E)
        d = target - S
        dist = float(np.clip(np.linalg.norm(d), 0.05 * (a + b), 0.995 * (a + b)))
        dh = d / max(np.linalg.norm(d), 1e-9)
        ca = (a * a + dist * dist - b * b) / (2 * a * dist)
        ang = math.acos(float(np.clip(ca, -1, 1)))
        pp = pole - dh * (pole @ dh)
        pp = pp / max(np.linalg.norm(pp), 1e-9)
        E2 = S + a * (math.cos(ang) * dh + math.sin(ang) * pp)
        self._set_global(L, G, s, align(E - S, E2 - S) @ G[s])
        G, P = self.fk(L, T)
        T2 = S + dh * dist
        self._set_global(L, G, e, align(P[w] - P[e], T2 - P[e]) @ G[e])

    def pose(self, L_base, bend=0.0, hands=None, legs=None, leg_w=0.0, squat=0.0):
        """L_base: (J,3,3) local rotations of the underlying clip frame.
        bend: forward bend of the spine (rad); squat: knee bend (rad), the hips drop to keep the feet planted.
        hands: {"L"/"R": (body-frame wrist target (f, l, h), weight)}; weight 0 keeps the clip's arm.
        legs: (J,3,3) local rotations of a walk frame, blended into the legs with leg_w.
        Returns local rotations and local translations."""
        L = np.array(L_base, copy=True)
        T = self.trans.copy()
        if legs is not None and leg_w > 0:
            for i in self.legs:
                L[i] = blend(L[i], legs[i], leg_w)
        if squat > 1e-4:
            for sd in "LR":
                for name, k in (("Thigh", -1.0), ("Calf", 2.0), ("Foot", -1.0)):
                    i = self.j[f"{sd}_{name}"]
                    G, _ = self.fk(L, T)
                    self._set_global(L, G, i, rot(self.left_s, k * squat) @ G[i])
            G, _ = self.fk(L, T)
            p = self.par[self.hip]
            up_local = (G[p].T if p >= 0 else np.eye(3)) @ self.up_s
            T[self.hip] = T[self.hip] - up_local * self.leg_len * (1 - math.cos(squat))
        # keep the lowest foot on the floor: root height correction (metres) for this pose
        _, Pf = self.fk(L, T)
        dz = self.foot_h0 - min(self.skel_to_body(Pf[i])[2] for i in self.feet)
        self.last_dz = dz
        if abs(bend) > 1e-4:
            for i, frac in zip(self.spine, (0.4, 0.35, 0.25)):
                G, _ = self.fk(L, T)
                self._set_global(L, G, i, rot(self.left_s, bend * frac) @ G[i])
        if hands:
            _, P0 = self.fk(L, T)
            for side, (tgt, w) in hands.items():
                if tgt is None or w <= 1e-4:
                    continue
                wrist = P0[self.arm[side][2]]
                t = self.body_to_skel(tgt[0], tgt[1], tgt[2] - dz)   # the root is raised by dz
                t = wrist + (t - wrist) * w if w < 1 else t
                out = self.left_s * (1.0 if side == "L" else -1.0)
                pole = -self.up_s * 0.8 + out * 0.6 - self.fwd_s * 0.2
                self._ik(L, T, side, t, pole)
        return L, T

    def _build_reach(self):
        """Shoulder positions for a grid of (bend, squat), to pick the least effort posture for a reach."""
        self.grid = []
        for q in np.radians([0, 10, 20, 30, 40, 50]):
            for b in np.radians(np.arange(0, 80, 5)):
                L, T = self.pose(self.L0, b, None, squat=q)
                _, P = self.fk(L, T)
                sh = {sd: self.skel_to_body(P[self.arm[sd][0]]) + np.array([0.0, 0.0, self.last_dz]) for sd in "LR"}
                self.grid.append((b + 0.8 * q, b, q, sh))
        self.grid.sort(key=lambda g: g[0])

    def solve(self, wrists, slack=0.95):
        """(bend, squat) that brings both shoulders within slack * arm length of their wrist targets."""
        best, bd = (0.0, 0.0), 1e9
        for _, b, q, sh in self.grid:
            d = max(np.linalg.norm(sh[sd] - np.asarray(wrists[sd])) for sd in wrists) / self.arm_len
            if d <= slack:
                return b, q
            if d < bd:
                best, bd = (b, q), d
        return best

    def hand_mid(self, L, T):
        """Body-frame point between the palms (where a held carton sits), relative to the floor point."""
        _, P = self.fk(L, T)
        unit = self.unit
        pts = []
        for side in "LR":
            _, e, w = self.arm[side]
            d = P[w] - P[e]
            pts.append(P[w] + d / np.linalg.norm(d) * 0.07 * unit)   # palm centre, just past the wrist
        return self.skel_to_body((pts[0] + pts[1]) / 2) + np.array([0.0, 0.0, self.last_dz])

    @staticmethod
    def quats(L):
        return Vt.QuatfArray([to_quat(m) for m in L])
