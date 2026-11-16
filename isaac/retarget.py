"""Retarget NVIDIA's biped animation clips onto the Reallusion skeleton used by the Isaac Sim characters.

The stock clips (Isaac/People/Animations, MotionLibrary) are authored for the 81-joint biped in
Characters/Biped_Setup.usd, while the characters are rigged with a 101-joint Reallusion skeleton, so
binding a clip directly leaves them in T-pose. We transfer each mapped joint's rotation away from its rest
pose in skeleton space:  R_tgt(t) = C (R_src(t) R_src_rest^T) C^T R_tgt_rest,  where C rotates the biped's
skeleton frame (Y-up) into the character's (Z-up), measured from the two rest poses. Unmapped joints keep
their rest pose; joint translations keep the character's proportions.
"""
import numpy as np
from pxr import Gf, Usd, UsdSkel, Vt

MAP = {"Pelvis": "Hip", "Spine1": "Waist", "Spine2": "Spine01", "Chest": "Spine02", "Neck1": "NeckTwist01",
       "Neck2": "NeckTwist02", "Head": "Head"}
for s in "LR":
    MAP.update({f"{s}_UpLeg": f"{s}_Thigh", f"{s}_LoLeg": f"{s}_Calf", f"{s}_Ankle": f"{s}_Foot",
                f"{s}_Ball": f"{s}_ToeBase", f"{s}_Clavicle": f"{s}_Clavicle", f"{s}_UpArm": f"{s}_Upperarm",
                f"{s}_LoArm": f"{s}_Forearm", f"{s}_Wrist": f"{s}_Hand"})
    for k in (1, 2, 3):
        MAP[f"{s}_ThumbFinger{k}"] = f"{s}_Thumb{k}"
    for src, dst in (("Index", "Index"), ("Middle", "Mid"), ("Ring", "Ring"), ("Pinky", "Pinky")):
        for k in (1, 2, 3):
            MAP[f"{s}_{src}Finger{k + 1}"] = f"{s}_{dst}{k}"


def leaf(p):
    return str(p).rsplit("/", 1)[-1]


def parents(joints):
    idx = {str(j): i for i, j in enumerate(joints)}
    return [idx.get(str(j).rsplit("/", 1)[0], -1) if "/" in str(j) else -1 for j in joints]


def qmat(q):
    w, (x, y, z) = q.GetReal(), q.GetImaginary()
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_rot(m):
    """Column-convention rotation and translation of a (row-vector) Gf.Matrix4d, scale removed."""
    a = np.array([[m[i][j] for j in range(4)] for i in range(4)])
    r = a[:3, :3].T
    u, _, vt = np.linalg.svd(r)
    return u @ vt, a[3, :3]


def to_quat(r):
    q = Gf.Matrix3d(*r.T.flatten().tolist()).ExtractRotation().GetQuat()
    return Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary()))


def frame(pos, names, up, a, b):
    i = {n: k for k, n in enumerate(names)}
    u = pos[i[up[1]]] - pos[i[up[0]]]
    v = pos[i[a]] - pos[i[b]]
    u = u / np.linalg.norm(u)
    v = v - u * (v @ u)
    v = v / np.linalg.norm(v)
    return np.stack([u, v, np.cross(u, v)], axis=1)


class Retargeter:
    def __init__(self, src_skel_url, tgt_skel):
        st = Usd.Stage.Open(src_skel_url)
        sk = UsdSkel.Skeleton(next(p for p in st.Traverse() if p.IsA(UsdSkel.Skeleton)))
        self.sj = [leaf(j) for j in sk.GetJointsAttr().Get()]
        s_rest = [mat_rot(m) for m in sk.GetBindTransformsAttr().Get()]
        self.s_rest = np.stack([r for r, _ in s_rest])
        s_pos = np.stack([t for _, t in s_rest])
        tg = UsdSkel.Skeleton(tgt_skel)
        self.tjoints = tg.GetJointsAttr().Get()
        self.tj = [leaf(j) for j in self.tjoints]
        self.tpar = parents(self.tjoints)
        t_bind = [mat_rot(m) for m in tg.GetBindTransformsAttr().Get()]
        self.t_rest = np.stack([r for r, _ in t_bind])
        t_pos = np.stack([t for _, t in t_bind])
        self.t_bind_t = t_pos
        rest_local = tg.GetRestTransformsAttr().Get()
        self.t_local_rest = [mat_rot(m) for m in rest_local]
        self.t_trans = Vt.Vec3fArray([Gf.Vec3f(*map(float, t)) for _, t in self.t_local_rest])
        self.t_scale = Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * len(self.tj))
        fs = frame(s_pos, self.sj, ("Pelvis", "Head"), "L_UpArm", "R_UpArm")
        ft = frame(t_pos, self.tj, ("Hip", "Head"), "L_Upperarm", "R_Upperarm")
        self.C = ft @ fs.T
        self.tmap = {self.tj.index(d): self.sj.index(s) for s, d in MAP.items() if s in self.sj and d in self.tj}

    def clip(self, url, fps=30.0, max_sec=None):
        """Retargeted local rotations (one Vt.QuatfArray per output frame) for a clip, looped at `fps`."""
        return [Vt.QuatfArray([to_quat(m) for m in L]) for L in self.clip_mats(url, fps, max_sec)]

    def clip_mats(self, url, fps=30.0, max_sec=None):
        """Retargeted local rotation matrices, shape (frames, joints, 3, 3)."""
        st = Usd.Stage.Open(url)
        an = UsdSkel.Animation(next(p for p in st.Traverse() if p.IsA(UsdSkel.Animation)))
        aj = an.GetJointsAttr().Get()
        apar = parents(aj)
        aidx = [self.sj.index(leaf(j)) if leaf(j) in self.sj else -1 for j in aj]
        ts = an.GetRotationsAttr().GetTimeSamples() or [0.0]
        tcps = st.GetTimeCodesPerSecond() or 30.0
        n_out = max(int(min(len(ts) / tcps, max_sec or 1e9) * fps), 1)
        out = np.zeros((n_out, len(self.tj), 3, 3))
        for k in range(n_out):
            tc = ts[0] + k / fps * tcps
            rots = an.GetRotationsAttr().Get(tc)
            g_src = np.zeros((len(self.sj), 3, 3))
            gl = [None] * len(aj)
            for i in range(len(aj)):
                r = qmat(rots[i])
                gl[i] = r if apar[i] < 0 else gl[apar[i]] @ r
                if aidx[i] >= 0:
                    g_src[aidx[i]] = gl[i]
            g_t = [None] * len(self.tj)
            for i in range(len(self.tj)):
                p = self.tpar[i]
                if i in self.tmap:
                    s = self.tmap[i]
                    g_t[i] = self.C @ (g_src[s] @ self.s_rest[s].T) @ self.C.T @ self.t_rest[i]
                else:
                    g_t[i] = (g_t[p] if p >= 0 else np.eye(3)) @ self.t_local_rest[i][0]
                out[k, i] = (g_t[p].T if p >= 0 else np.eye(3)) @ g_t[i]
        return out
