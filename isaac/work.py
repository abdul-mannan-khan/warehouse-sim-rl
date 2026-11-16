"""Pick-and-place work cycles for the Isaac Sim characters, built on motion-capture clips (numpy only).

Clips (Adobe Mixamo, retargeted by mocap.py; not redistributed, see README):
  turn       "Box Turn": a stepping turn of ~134 deg to the left; its mirror image gives right turns
  put_down   "Putting Down": one-handed put-down with knee bend and weight shift, then standing back up
  pick_up    "Picking Up": one-handed pick-up from low; its early frames give a natural forward lean
Legs, hips and spine come from the clips. The right arm is guided onto the parcel and the tote slot by
two-bone IK on top of the clip (pose.py), the head looks at what the hand is about to touch, and consecutive
segments are cross-faded so the motion never stops dead. The parcels are small (17 x 21 x 14 cm), so they
are handled one-handed, as in the clips.
"""
import math

import numpy as np

from pose import m2q

FADE = 0.22                       # cross-fade between segments (s)


# ------------------------------------------------------------------ quaternion arrays
def mats_to_q(L):
    return np.array([m2q(m) for m in L])


def q_to_mats(Q):
    Q = Q / np.linalg.norm(Q, axis=-1, keepdims=True)
    w, x, y, z = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def nlerp(a, b, w):
    if w <= 0:
        return a
    if w >= 1:
        return b
    s = np.sign(np.sum(a * b, axis=1, keepdims=True))
    s[s == 0] = 1
    q = a * (1 - w) + b * s * w
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def minjerk(f):
    f = min(max(f, 0.0), 1.0)
    return f * f * f * (10 - 15 * f + 6 * f * f)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------ clip cache
class Clip:
    """A retargeted clip sampled at its own frames: quaternions (F,J,4), hips yaw (F,), hips travel (F,2) in
    the body frame (forward, left; metres)."""

    def __init__(self, mclip, ps, strip_yaw=False):
        self.fps, self.n = mclip.fps, mclip.n
        Q, Y, H = [], [], []
        for k in range(mclip.n):
            L, yaw, ht = mclip.locals_at(k, strip_yaw=strip_yaw)
            Q.append(mats_to_q(L))
            Y.append(yaw)
            r = ps.A @ ht
            H.append((float(r @ ps.fwd_r), float(r @ ps.left_r)))
        self.Q, self.yaw, self.ht = np.array(Q), np.array(Y), np.array(H)

    def at(self, k):
        k = min(max(k, 0.0), self.n - 1.0)
        k0 = int(k)
        k1 = min(k0 + 1, self.n - 1)
        w = k - k0
        return (nlerp(self.Q[k0], self.Q[k1], w), self.yaw[k0] + (self.yaw[k1] - self.yaw[k0]) * w,
                self.ht[k0] + (self.ht[k1] - self.ht[k0]) * w)


def upper_mask(ps):
    """Arm joints (clavicle down: twist bones and fingers too) and neck/head."""
    roots = {ps.j[n] for n in ("L_Clavicle", "R_Clavicle", "NeckTwist01") if n in ps.j}
    m = np.zeros(len(ps.par), bool)
    for i in range(len(ps.par)):
        p = i
        while p >= 0:
            if p in roots:
                m[i] = True
                break
            p = ps.par[p]
    return m


def hand_body(ps, Q, side="R"):
    """Body-frame wrist position (floor at 0) of a clip pose."""
    L, T = q_to_mats(Q), ps.trans
    _, P = ps.fk(L, T)
    return ps.skel_to_body(P[ps.arm[side][2]]) + np.array([0.0, 0.0, ps.ground_dz(L, T)])


def palm(ps, L, T, side="R"):
    """Body-frame palm centre of a posed skeleton (after Poser.pose, which sets last_dz)."""
    _, P = ps.fk(L, T)
    _, e, w = ps.arm[side]
    d = P[w] - P[e]
    p = P[w] + d / np.linalg.norm(d) * 0.08 * ps.unit
    return ps.skel_to_body(p) + np.array([0.0, 0.0, ps.last_dz])


def upright(ps, q, stance):
    """Keep a clip's legs (thigh globals, so the feet land where the clip puts them) but stand the pelvis and
    spine up as in `stance`, turned to the clip's hip heading. Mixamo's box turn leans back against a heavy
    box, which looks wrong with a small parcel."""
    L, Ls = q_to_mats(q), q_to_mats(stance)
    G, _ = ps.fk(L)
    Gs, _ = ps.fk(Ls)
    hip, pel = ps.j["Hip"], ps.j.get("Pelvis")
    f = G[hip] @ (Gs[hip].T @ ps.fwd_s)
    f = f - ps.up_s * (f @ ps.up_s)
    yaw = math.atan2(float(np.cross(ps.fwd_s, f) @ ps.up_s), float(ps.fwd_s @ f))
    from pose import rot
    Ry = rot(ps.up_s, yaw)
    out = Ls.copy()
    for i in range(len(L)):                          # everything below the thighs as in the clip
        out[i] = L[i]
    p = ps.par[hip]
    Gp = G[p] if p >= 0 else np.eye(3)
    Ghip = Ry @ Gs[hip]
    out[hip] = Gp.T @ Ghip
    Gpel = Ghip
    if pel is not None:
        Gpel = Ry @ Gs[pel]
        out[pel] = Ghip.T @ Gpel
    for n in ("Waist", "Spine01", "Spine02"):
        if n in ps.j:
            out[ps.j[n]] = Ls[ps.j[n]]
    for s_ in "LR":
        th = ps.j[f"{s_}_Thigh"]
        out[th] = (Gpel if ps.par[th] == pel else Ghip).T @ G[th]
    return mats_to_q(out)


class Worker:
    """One character's clips and the key poses/frames derived from them."""

    def __init__(self, ps, clips, idle_frames):
        self.ps = ps
        self.turnL, self.turnR = clips["turn"], clips["turn_m"]
        self.put, self.pick = clips["put_down"], clips["pick_up"]
        self.upper = upper_mask(ps)
        self.stance = self.put.Q[0].copy()          # relaxed standing (also how pick-up ends)
        for c in (self.turnL, self.turnR):
            if not getattr(c, "upright", False):
                c.Q = np.array([upright(ps, q, self.stance) for q in c.Q])
                c.upright = True
        self.idle = [mats_to_q(L) for L in idle_frames]
        self.lean = self.pick.Q[33]                  # forward lean, right hand reaching low and forward
        self.rest_hand = hand_body(ps, self.stance)
        # put-down: start of the bend, the right hand's lowest point (release), upright again
        hs = [hand_body(ps, self.put.Q[k])[2] for k in range(self.put.n)]
        self.put_rel = int(np.argmin(hs))
        self.put_k0 = next((k for k in range(self.put_rel, -1, -1) if hs[k] > hs[0] - 0.01), 0)
        self.put_k0 = max(self.put_k0 - 8, 0)
        self.put_k1 = next((k for k in range(self.put_rel, self.put.n) if hs[k] > hs[-1] - 0.03), self.put.n - 1)
        self.put_k1 = min(self.put_k1 + 10, self.put.n - 1)
        self.put_hand = hand_body(ps, self.put.Q[self.put_rel])


# ------------------------------------------------------------------ segments
def seg_turn(wk, f, psi0, psi1, carrying):
    d = wrap(psi1 - psi0)
    clip = wk.turnL if d > 0 else wk.turnR           # the clip turns left; its mirror image turns right
    k = f * (clip.n - 1)
    q, yaw, _ = clip.at(k)
    y_end = clip.yaw[-1]
    psi = psi0 + d * (yaw / y_end if abs(y_end) > 1e-6 else f)
    off = np.zeros(2)                                # root travel comes from the planted foot (plan_root)
    ahead = wrap(psi1 - psi) * min(1.0, 0.5 + f)     # eyes lead the turn
    return {"q": np.where(wk.upper[:, None], wk.stance, q), "psi": psi, "off": off, "lean": 0.0,
            "hand": None, "look": (2.0 * math.cos(ahead), 2.0 * math.sin(ahead), 1.2), "look_w": 0.8,
            "parcel": "hand" if carrying else "src", "placed": False}


def seg_reach(wk, f, psi, target, grip_at=0.5, lean_max=0.10, lean_q=0.15, look_w=0.9):
    """Reach the right hand to `target` (body frame), take or drop the parcel there at grip_at, bring the hand
    back to its rest place at the side."""
    rest = wk.rest_hand
    a, b = grip_at - 0.08, grip_at + 0.08
    if f < a:
        g = minjerk(f / a)
        hand, w = rest + (target - rest) * g, 1.0
    elif f < b:
        g, hand, w = 1.0, target, 1.0
    else:
        g = 1.0 - minjerk((f - b) / (1 - b))
        hand, w = target + (rest - target) * (1 - g), g
    q = nlerp(wk.stance, wk.lean, lean_q * g)
    return {"q": q, "psi": psi, "off": np.zeros(2), "lean": lean_max * g, "hand": (hand, w),
            "look": target, "look_w": look_w * (1.0 if f < b else g), "parcel": None, "placed": False}


def seg_put(wk, f, psi, B):
    """The motion-capture put-down, its right hand corrected onto the tote slot B (body frame)."""
    k = wk.put_k0 + (wk.put_k1 - wk.put_k0) * f
    q, _, ht = wk.put.at(k)
    fr = (wk.put_rel - wk.put_k0) / max(wk.put_k1 - wk.put_k0, 1)
    if f < fr:
        w = minjerk((f / fr - 0.2) / 0.8)
    else:
        w = 1.0 - minjerk((f - fr) / (0.45 * (1 - fr)))
    corr = (B - wk.put_hand) * w
    hand = (hand_body(wk.ps, q) + corr, 1.0) if w > 1e-3 else None
    return {"q": q, "psi": psi, "off": np.zeros(2), "lean": 0.0, "hand": hand, "look": B,
            "look_w": 0.9 if f < fr + 0.1 else max(0.0, 0.9 - (f - fr - 0.1) * 4),
            "parcel": "hand" if f < fr else None, "placed": f >= fr}


# ------------------------------------------------------------------ cycles
# A cycle starts facing where the last parcel went (dst): turn to the source, take a parcel, turn back,
# deliver it. Pickers deliver with the motion-capture put-down into the robot's tote; packers drop the parcel
# into the shipping box on the table with a reach.
PICK_SEG = (("turn_src", 1.25), ("take", 1.30), ("turn_dst", 1.25), ("put", 2.90))
PACK_SEG = (("turn_src", 0.95), ("take", 1.25), ("turn_dst", 0.95), ("drop", 1.25))
PICK_CYCLE = sum(d for _, d in PICK_SEG)
PACK_CYCLE = sum(d for _, d in PACK_SEG)


def plan(segs, D):
    """Start/end of each segment in a cycle of length D: compressed if D is short, padded at the end with
    standing still if it is long."""
    s = min(1.0, D / sum(d for _, d in segs))
    out, t = [], 0.0
    for n, d in segs:
        out.append((n, t, t + d * s))
        t += d * s
    return out


def station_desc(wk, name, f, ctx):
    """ctx: psiS / psiR facing the source / destination, A / B the parcel there (body frame, floor at 0)."""
    lean = ctx.get("lean", (0.10, 0.15))
    if name == "turn_src":
        return seg_turn(wk, f, ctx["psiR"], ctx["psiS"], carrying=False)
    if name == "take":
        d = seg_reach(wk, f, ctx["psiS"], ctx["A"], lean_max=lean[0], lean_q=lean[1])
        d["parcel"] = "src" if f < 0.5 else "hand"
        return d
    if name == "turn_dst":
        return seg_turn(wk, f, ctx["psiS"], ctx["psiR"], carrying=True)
    if name == "put":
        return seg_put(wk, f, ctx["psiR"], ctx["B"])
    if name == "drop":
        d = seg_reach(wk, f, ctx["psiR"], ctx["B"], lean_max=lean[0], lean_q=lean[1])
        d["parcel"] = "hand" if f < 0.5 else None
        d["placed"] = f >= 0.5
        return d
    raise ValueError(name)


def blend_desc(a, b, w):
    """Cross-fade two segment descriptions (a -> b by w)."""
    if w >= 1:
        return b
    out = dict(b)
    out["q"] = nlerp(a["q"], b["q"], w)
    out["psi"] = a["psi"] + wrap(b["psi"] - a["psi"]) * w
    out["off"] = np.asarray(a["off"]) * (1 - w) + np.asarray(b["off"]) * w
    out["lean"] = a["lean"] * (1 - w) + b["lean"] * w
    ha, hb = a["hand"], b["hand"]
    if ha is not None or hb is not None:
        pa = ha[0] if ha is not None else hb[0]
        pb = hb[0] if hb is not None else ha[0]
        wa = ha[1] if ha is not None else 0.0
        wb = hb[1] if hb is not None else 0.0
        out["hand"] = (pa * (1 - w) + pb * w, wa * (1 - w) + wb * w)
    return out


def cycle_desc(wk, segs, desc_fn, t, D, ctx):
    """Description at time t of a cycle of length D, cross-fading out of the previous segment."""
    p = plan(segs, D)
    for i, (n, a, b) in enumerate(p):
        if t < b or i == len(p) - 1:
            f = min(max((t - a) / max(b - a, 1e-6), 0.0), 1.0)
            d = desc_fn(wk, n, f, ctx)
            if i > 0 and t - a < FADE:
                d = blend_desc(desc_fn(wk, p[i - 1][0], 1.0, ctx), d, minjerk((t - a) / FADE))
            return d


def apply_desc(wk, d):
    """Pose the skeleton: local rotations, translations; root height correction is left in ps.last_dz."""
    ps = wk.ps
    hands = {"R": d["hand"]} if d.get("hand") is not None else None
    L, T = ps.pose(q_to_mats(d["q"]), d.get("lean", 0.0), hands)
    if d.get("look") is not None and d.get("look_w", 0) > 1e-3:
        ps.look(L, T, d["look"], d["look_w"])
    return L, T


# ------------------------------------------------------------------ planted feet
FEET = ("L_Foot", "R_Foot")


def feet_body(ps, q):
    """Body-frame ankle positions (xy) and heights of a clip pose."""
    L, T = q_to_mats(q), ps.trans
    _, P = ps.fk(L, T)
    dz = ps.ground_dz(L, T)
    out = []
    for n in FEET:
        b = ps.skel_to_body(P[ps.j[n]])
        out.append((b[:2], b[2] + dz))
    return out


def rotz(psi, v):
    c, s_ = math.cos(psi), math.sin(psi)
    return np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1]])


def plan_root(wk, segs, desc_fn, D, ctx, fps=30.0, lift=0.03):
    """Root travel (world xy, relative to the worker's home spot) over one cycle, sampled at fps, chosen so the
    supporting foot never slides: each frame the foot that is lower (within `lift`) and moves least is held
    in place and the root follows. Whatever drift is left at the end of the cycle is removed gradually, so
    the next cycle starts on the home spot again."""
    n = max(int(round(D * fps)), 1)
    root = np.zeros(2)
    roots = [root.copy()]
    prev = None
    for i in range(n + 1):
        d = cycle_desc(wk, segs, desc_fn, i / fps, D, ctx)
        fb = feet_body(wk.ps, d["q"])
        w = [root + rotz(d["psi"], p) for p, _ in fb]
        if prev is not None:
            h = [z for _, z in fb]
            mv = [np.linalg.norm(w[k] - prev[k]) for k in range(2)]
            cand = [k for k in range(2) if h[k] <= min(h) + lift]
            k = min(cand, key=lambda k_: mv[k_])
            root = root + (prev[k] - w[k])
            w = [root + rotz(d["psi"], p) for p, _ in fb]
            roots.append(root.copy())
        prev = w
    roots = np.array(roots)
    drift = roots[-1] - roots[0]
    ramp = np.array([minjerk(i / n) for i in range(n + 1)])
    return roots - ramp[:, None] * drift


def root_at(roots, t, fps=30.0):
    k = min(max(t * fps, 0.0), len(roots) - 1.0)
    k0 = int(k)
    k1 = min(k0 + 1, len(roots) - 1)
    return roots[k0] + (roots[k1] - roots[k0]) * (k - k0)
