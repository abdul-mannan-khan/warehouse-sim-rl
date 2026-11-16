"""Retarget Mixamo motion-capture clips onto the Reallusion skeleton of the Isaac Sim characters (numpy only).

Mixamo clips (FBX, converted to USD and dumped by export_mocap.py) use the 65-joint `mixamorig` skeleton in a
Y-up T-pose; the characters use a 101-joint Reallusion skeleton. For every mapped joint we transfer the
joint's global rotation away from its rest pose:

    G_tgt(t) = C (G_src(t) G_src_rest^T) C^T Q G_tgt_rest

C rotates the source skeleton frame into the target frame (from hips->head and shoulder->shoulder axes of the
two rest poses). Q is a per-joint correction that first turns the target bone onto the source rest bone
direction, so differences between the two rest poses (e.g. A-pose vs T-pose arms) do not leak into the motion.
Unmapped joints (twist and helper bones) keep their rest local rotation.

The hips' horizontal travel (scaled to the character's leg length) and its turn about the vertical axis are
returned separately, so the caller can move and turn the character root instead of the skeleton; height is
left to the sole-grounding in pose.py.
"""
import math

import numpy as np

SRC = "mixamorig_"
MAP = {"Hips": "Hip", "Spine": "Waist", "Spine1": "Spine01", "Spine2": "Spine02", "Neck": "NeckTwist01",
       "Head": "Head"}
CHILD = {"Hips": "Spine", "Spine": "Spine1", "Spine1": "Spine2", "Spine2": "Neck", "Neck": "Head"}
for S, s in (("Left", "L"), ("Right", "R")):
    MAP.update({f"{S}Shoulder": f"{s}_Clavicle", f"{S}Arm": f"{s}_Upperarm", f"{S}ForeArm": f"{s}_Forearm",
                f"{S}Hand": f"{s}_Hand", f"{S}UpLeg": f"{s}_Thigh", f"{S}Leg": f"{s}_Calf",
                f"{S}Foot": f"{s}_Foot", f"{S}ToeBase": f"{s}_ToeBase"})
    CHILD.update({f"{S}Shoulder": f"{S}Arm", f"{S}Arm": f"{S}ForeArm", f"{S}ForeArm": f"{S}Hand",
                  f"{S}Hand": f"{S}HandMiddle1", f"{S}UpLeg": f"{S}Leg", f"{S}Leg": f"{S}Foot",
                  f"{S}Foot": f"{S}ToeBase"})
    for src, dst in (("Thumb", "Thumb"), ("Index", "Index"), ("Middle", "Mid"), ("Ring", "Ring"),
                     ("Pinky", "Pinky")):
        for k in (1, 2, 3):
            MAP[f"{S}Hand{src}{k}"] = f"{s}_{dst}{k}"
            CHILD[f"{S}Hand{src}{k}"] = f"{S}Hand{src}{k + 1}"


def rot(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * k + (1 - math.cos(ang)) * k @ k


def align(u, v):
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    c = np.cross(u, v)
    s, d = np.linalg.norm(c), float(u @ v)
    if s < 1e-9:
        return np.eye(3)
    return rot(c, math.atan2(s, d))


def qmats(q):
    """(N,4) wxyz quaternions -> (N,3,3)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def body_frame(pos, idx, up, a, b):
    u = pos[idx[up[1]]] - pos[idx[up[0]]]
    v = pos[idx[a]] - pos[idx[b]]
    u = u / np.linalg.norm(u)
    v = v - u * (v @ u)
    v = v / np.linalg.norm(v)
    return np.stack([u, v, np.cross(u, v)], axis=1)


def mirror_data(d):
    """Left-right mirror image of a dumped Mixamo clip (Mixamo's lateral axis is x): swap Left/Right joint
    names and reflect every rotation and translation through the x = 0 plane."""
    def swap(names):
        out = []
        for n in names:
            n = str(n)
            out.append(n.replace("Left", "#").replace("Right", "Left").replace("#", "Right"))
        return np.array(out)
    M = np.diag([-1.0, 1.0, 1.0])
    e = dict(d)
    e["sj"], e["aj"] = swap(d["sj"]), swap(d["aj"])
    q = np.array(d["rot"], float)
    e["rot"] = q * np.array([1.0, 1.0, -1.0, -1.0])
    e["trans"] = np.array(d["trans"], float) * np.array([-1.0, 1.0, 1.0])
    e["bind_r"] = M @ np.asarray(d["bind_r"]) @ M
    e["bind_t"] = np.asarray(d["bind_t"]) * np.array([-1.0, 1.0, 1.0])
    return e


class MocapClip:
    """One Mixamo clip retargeted onto one character skeleton.

    tj, tpar: target joint names and parent indices; t_bind_r/t_bind_t: target bind (skeleton-space) pose;
    t_loc_r: target rest local rotations; up_s: the character's up axis in skeleton space; leg: leg length in
    skeleton units."""

    def __init__(self, data, tj, tpar, t_bind_r, t_bind_t, t_loc_r, up_s, leg):
        sj = [str(n) for n in data["sj"]]
        aj = [str(n) for n in data["aj"]]
        si = {n: i for i, n in enumerate(sj)}
        ti = {n: i for i, n in enumerate(tj)}
        self.tj, self.tpar, self.t_loc_r = list(tj), list(tpar), np.asarray(t_loc_r)
        self.up_s = up_s / np.linalg.norm(up_s)
        s_bind_r, s_bind_t = np.asarray(data["bind_r"]), np.asarray(data["bind_t"])
        fs = body_frame(s_bind_t, si, (SRC + "Hips", SRC + "Head"), SRC + "LeftArm", SRC + "RightArm")
        ft = body_frame(t_bind_t, ti, ("Hip", "Head"), "L_Upperarm", "R_Upperarm")
        self.C = ft @ fs.T
        # mapped joints: target index -> (source index, rest correction Q)
        self.map = {}
        for s, t in MAP.items():
            if SRC + s not in si or t not in ti:
                continue
            Q = np.eye(3)
            c = CHILD.get(s)
            if c and SRC + c in si and MAP.get(c) in ti:
                ds = self.C @ (s_bind_t[si[SRC + c]] - s_bind_t[si[SRC + s]])
                dt = t_bind_t[ti[MAP[c]]] - t_bind_t[ti[t]]
                if np.linalg.norm(ds) > 1e-6 and np.linalg.norm(dt) > 1e-6:
                    Q = align(dt, ds)
            self.map[ti[t]] = (si[SRC + s], Q @ np.asarray(t_bind_r)[ti[t]])
        self.s_bind_r = s_bind_r
        # source animation -> source skeleton-space global rotations per frame
        apar = list(data["apar"])
        a2s = [si.get(n, -1) for n in aj]
        R = qmats(np.asarray(data["rot"]))                       # (F, A, 3, 3) local
        F = len(R)
        self.n = F
        times = np.asarray(data["times"], float)
        self.fps = float(data["tcps"]) / float(np.median(np.diff(times))) if F > 1 else 30.0
        G = np.zeros((F, len(sj), 3, 3))
        g = np.zeros((F, len(aj), 3, 3))
        for i in range(len(aj)):
            g[:, i] = R[:, i] if apar[i] < 0 else g[:, apar[i]] @ R[:, i]
            if a2s[i] >= 0:
                G[:, a2s[i]] = g[:, i]
        self.G_src = G
        # hips travel in the target frame, in target skeleton units (scaled by leg length)
        hips = a2s.index(si[SRC + "Hips"])
        tr = np.asarray(data["trans"])[:, hips]                    # source local == skeleton space for the root
        s_leg = (np.linalg.norm(s_bind_t[si[SRC + "LeftLeg"]] - s_bind_t[si[SRC + "LeftUpLeg"]]) +
                 np.linalg.norm(s_bind_t[si[SRC + "LeftFoot"]] - s_bind_t[si[SRC + "LeftLeg"]]))
        self.hip_t = (tr - tr[0]) @ self.C.T * (leg / s_leg)
        self.hip = ti["Hip"]

    def frame_index(self, t):
        """Clip time in seconds -> fractional frame."""
        return min(max(t * self.fps, 0.0), self.n - 1.0)

    def globals_at(self, k):
        """Target skeleton-space global rotations of the mapped joints at fractional frame k (dict)."""
        k0 = int(math.floor(k))
        k1 = min(k0 + 1, self.n - 1)
        w = k - k0
        out = {}
        for ti_, (si_, rest) in self.map.items():
            a = self.G_src[k0, si_]
            if w > 1e-6:
                a = slerp_m(a, self.G_src[k1, si_], w)
            out[ti_] = self.C @ (a @ self.s_bind_r[si_].T) @ self.C.T @ rest
        return out

    def locals_at(self, k, strip_yaw=True):
        """Local rotations (J,3,3) at fractional frame k, the hips' yaw about up (rad, relative to frame 0) and
        horizontal hips travel (skeleton units, skeleton space). With strip_yaw the turn is removed from the
        skeleton so the caller can apply it to the character root."""
        g = self.globals_at(k)
        yaw = 0.0
        if strip_yaw:
            yaw = self.yaw(g[self.hip]) - self.yaw0()
            Rm = rot(self.up_s, -yaw)
            g = {i: Rm @ m for i, m in g.items()}
        n = len(self.tj)
        G = [None] * n
        L = np.zeros((n, 3, 3))
        for i in range(n):
            p = self.tpar[i]
            Gp = G[p] if p >= 0 else np.eye(3)
            G[i] = g[i] if i in g else Gp @ self.t_loc_r[i]
            L[i] = Gp.T @ G[i]
        k0 = int(math.floor(k))
        k1 = min(k0 + 1, self.n - 1)
        ht = self.hip_t[k0] + (self.hip_t[k1] - self.hip_t[k0]) * (k - k0)
        ht = ht - self.up_s * (ht @ self.up_s)
        if strip_yaw:
            ht = rot(self.up_s, -yaw) @ ht
        return L, yaw, ht

    def yaw(self, Ghip):
        f = Ghip @ self._hip_fwd_local()
        f = f - self.up_s * (f @ self.up_s)
        return math.atan2(float(np.cross(self._ref, f) @ self.up_s), float(self._ref @ f))

    def _hip_fwd_local(self):
        if not hasattr(self, "_hf"):
            g0 = self.globals_at(0)[self.hip]
            # character forward in skeleton space = left x up
            ref = self._fwd_s
            self._hf = g0.T @ ref
            self._ref = ref
        return self._hf

    def yaw0(self):
        if not hasattr(self, "_y0"):
            self._y0 = self.yaw(self.globals_at(0)[self.hip])
        return self._y0

    def set_forward(self, fwd_s):
        self._fwd_s = fwd_s / np.linalg.norm(fwd_s)
        for a in ("_hf", "_ref", "_y0"):
            if hasattr(self, a):
                delattr(self, a)


def slerp_m(a, b, w):
    from pose import m2q, q2m
    qa, qb = m2q(a), m2q(b)
    if qa @ qb < 0:
        qb = -qb
    d = float(np.clip(qa @ qb, -1, 1))
    if d > 0.9995:
        q = qa + (qb - qa) * w
    else:
        th = math.acos(d)
        q = (math.sin((1 - w) * th) * qa + math.sin(w * th) * qb) / math.sin(th)
    return q2m(q)
