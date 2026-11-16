"""Kinematic UR5e + Robotiq 2F-85 for the Isaac Sim replay (numpy only, no physics).

The Isaac Sim asset Robots/UniversalRobots/ur5e/ur5e.usd (Gripper variant "Robotiq_2f_85") keeps every link as
a flat child prim of /ur5e, joined by physics joints. The replay does not step physics, so we pose the links
ourselves: forward kinematics over the joint frames below (copied from the asset), a damped least-squares IK
for a top-down grasp, and the gripper's parallelogram finger linkage.

All poses are 4x4 matrices in the frame of the robot's root prim (z up, metres).
"""
import math

import numpy as np

# (child link, parent link, localPos0, localRot0 wxyz) for the six revolute joints; localPos1/Rot1 are identity
UR5E = (
    ("shoulder_link", "base_link", (0, 0, 0.1625), (-4.371139e-8, 0, 0, 1)),
    ("upper_arm_link", "shoulder_link", (0, 0, 0), (0.70710677, 0.70710677, 0, 0)),
    ("forearm_link", "upper_arm_link", (-0.425, 0, 0), (1, 0, 0, 0)),
    ("wrist_1_link", "forearm_link", (-0.3922, 0, 0.1333), (1, 0, 0, 0)),
    ("wrist_2_link", "wrist_1_link", (0, -0.0997, 0), (0.70710677, 0.70710677, 0, 0)),
    ("wrist_3_link", "wrist_2_link", (0, 0.0996, 0), (0.70710677, -0.70710677, -6.181724e-8, 0)),
)
LINKS = ["base_link"] + [j[0] for j in UR5E]
# rest pose of the Gripper prim (child of the root) while wrist_3 is at its rest pose
GRIP_REST = ((0.8172, 0.2329, 0.0628), (0.5, -0.5, -0.5, -0.5))
TCP = np.array([0.0, 0.0, 0.130])          # grasp centre between the finger pads, gripper frame (z = approach)
FINGER_MAX = math.radians(47.0)

# finger linkage in the gripper frame (pivots in the y-z plane, all joints turn about x). Each side is a
# parallelogram: both knuckles turn by the same angle and the inner finger (with its pad) only translates.
P_OUTER, P_INNER = np.array([0.0306, 0.05466]), np.array([0.0127, 0.06118])
F_OUTER = np.array([0.06776, 0.09809])                       # outer finger -> inner finger pivot


def qmat(q):
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mquat(R):
    """3x3 -> wxyz."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)
    q = np.array(q)
    return q if q[0] >= 0 else -q


def tf(p, q):
    T = np.eye(4)
    T[:3, :3] = qmat(q)
    T[:3, 3] = p
    return T


def rz(a):
    T = np.eye(4)
    c, s = math.cos(a), math.sin(a)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def rx_about(a, py, pz):
    """Rotation by a about the x axis through (0, py, pz)."""
    T = np.eye(4)
    c, s = math.cos(a), math.sin(a)
    T[1:3, 1:3] = [[c, -s], [s, c]]
    T[1:3, 3] = np.array([py, pz]) - T[1:3, 1:3] @ np.array([py, pz])
    return T


J0 = [tf(p, q) for _, _, p, q in UR5E]


def fk(q):
    """Joint angles (6,) -> {link: pose}, plus 'gripper' (the Gripper prim) and 'tcp'."""
    out = {"base_link": np.eye(4)}
    for i, (child, parent, _, _) in enumerate(UR5E):
        out[child] = out[parent] @ J0[i] @ rz(q[i])
    out["gripper"] = out["wrist_3_link"] @ W3G
    out["tcp"] = out["gripper"].copy()
    out["tcp"][:3, 3] = out["gripper"][:3, :3] @ TCP + out["gripper"][:3, 3]
    return out


_rest = {"base_link": np.eye(4)}
for _i, (_c, _p, _, _) in enumerate(UR5E):
    _rest[_c] = _rest[_p] @ J0[_i]
W3G = np.linalg.inv(_rest["wrist_3_link"]) @ tf(*GRIP_REST)
REST = _rest


def fingers(a):
    """Finger closing angle a (0 open .. FINGER_MAX) -> {gripper link: pose in the Robotiq_2F_85 frame}."""
    out = {"base_link": np.eye(4)}
    v = F_OUTER - P_OUTER
    for side, sg in (("right", 1.0), ("left", -1.0)):
        po, pi = P_OUTER * [sg, 1], P_INNER * [sg, 1]
        Ro = rx_about(sg * a, *po)
        out[f"{side}_outer_knuckle"] = Ro
        out[f"{side}_outer_finger"] = Ro
        out[f"{side}_inner_knuckle"] = rx_about(sg * a, *pi)
        c, s = math.cos(sg * a), math.sin(sg * a)
        vv = v * [sg, 1]
        d = np.array([c * vv[0] - s * vv[1], s * vv[0] + c * vv[1]]) - vv
        T = np.eye(4)
        T[1:3, 3] = d
        out[f"{side}_inner_finger"] = T
    return out


def pad_gap(a):
    """Distance between the finger pads (m) at closing angle a (85 mm open, about 0 at FINGER_MAX)."""
    v = F_OUTER - P_OUTER
    y = P_OUTER[0] + v[0] * math.cos(a) - v[1] * math.sin(a)
    return 0.085 + 2 * (y - F_OUTER[0])


def grip_angle(width):
    """Closing angle at which the pads touch an item of this width."""
    lo, hi = 0.0, FINGER_MAX
    for _ in range(40):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if pad_gap(m) > width else (lo, m)
    return (lo + hi) / 2


def target(p, yaw):
    """Top-down grasp pose: approach axis straight down, jaw axis horizontal at heading yaw."""
    T = np.eye(4)
    y = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    T[:3, :3] = np.stack([np.cross(y, z), y, z], axis=1)
    T[:3, 3] = p
    return T


def _err(T, G):
    e = np.zeros(6)
    e[:3] = G[:3, 3] - T[:3, 3]
    q = mquat(G[:3, :3] @ T[:3, :3].T)                     # rotation vector of the remaining turn
    s = np.linalg.norm(q[1:])
    if s > 1e-12:
        e[3:] = q[1:] / s * 2 * math.atan2(s, q[0])
    return e


def ik(G, seed, iters=200, tol=1e-5):
    """Damped least squares from `seed` (deterministic) to tcp pose G. Returns (q, residual)."""
    q = np.array(seed, float)
    lam = 0.05
    for _ in range(iters):
        e = _err(fk(q)["tcp"], G)
        if np.linalg.norm(e) < tol:
            break
        J = np.zeros((6, 6))
        for k in range(6):
            dq = q.copy()
            dq[k] += 1e-6
            J[:, k] = (_err(fk(dq)["tcp"], G) - e) / -1e-6
        q = q + J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), e)
    return q, float(np.linalg.norm(_err(fk(q)["tcp"], G)))


def minjerk(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (10 - 15 * u + 6 * u * u)


class PickPlace:
    """One looping bin-to-box cycle for an arm. src/dst: item centres in the robot root frame; item: (l, w, h)
    of the item (w across the jaw). Joint keyframes are solved once; straight vertical moves are re-solved per
    frame from the keyframe so the gripper goes straight down into the bin and the box. item: (size across the
    jaw, size along the jaw, height) in metres; the jaw closes across item[0]."""

    CLEAR = 0.26                     # height of the approach points above the grasp / release points
    # (name, duration s); moves between keyframes, the gripper closes during "close" and opens during "open"
    STEPS = (("to_src", 0.85), ("down_src", 0.45), ("close", 0.30), ("up_src", 0.45), ("to_dst", 0.95),
             ("down_dst", 0.45), ("open", 0.25), ("up_dst", 0.40), ("home", 0.70))
    SEED = (math.radians(90), math.radians(-60), math.radians(100), math.radians(-130), math.radians(-90), 0.0)

    def __init__(self, src, dst, item, yaw_src=0.0, yaw_dst=0.0):
        self.src, self.dst = np.asarray(src, float), np.asarray(dst, float)
        self.item = item
        self.a_open, self.a_close = grip_angle(item[0] + 0.03), grip_angle(item[0])
        lift = np.array([0, 0, self.CLEAR])
        g_src = self.src + [0, 0, item[2] * 0.15]
        g_dst = self.dst + [0, 0, item[2] * 0.15]
        home = (self.src + self.dst) / 2 + [0, -0.12, self.CLEAR + 0.12]
        self.poses = {"home": target(home, (yaw_src + yaw_dst) / 2), "src_up": target(g_src + lift, yaw_src),
                      "src": target(g_src, yaw_src), "dst_up": target(g_dst + lift, yaw_dst),
                      "dst": target(g_dst, yaw_dst)}
        self.q, seed = {}, self.SEED
        for k in ("home", "src_up", "src", "dst_up", "dst"):
            self.q[k], res = ik(self.poses[k], seed)
            if res > 1e-3:
                raise ValueError(f"arm cannot reach {k}: residual {res:.4f}")
            seed = self.q["home"]
        self.T = sum(d for _, d in self.STEPS)
        self.g_src, self.g_dst = g_src, g_dst
        self._cache = {}

    def at(self, t):
        """Cycle time t (s) -> (joint angles, finger angle, item state) where item state is 'src' (in the bin),
        'held' or 'dst' (released in the box)."""
        t = t % self.T
        k = round(t, 4)
        if k in self._cache:
            return self._cache[k]
        t0 = 0.0
        for name, d in self.STEPS:
            if t < t0 + d or name == "home":
                u = minjerk((t - t0) / d)
                break
            t0 += d
        q, Q = None, self.q
        a = self.a_open
        state = "src"
        if name == "to_src":
            q = Q["home"] + (Q["src_up"] - Q["home"]) * u
        elif name in ("down_src", "up_src"):
            h = (1 - u) if name == "down_src" else u
            q = ik(target(self.g_src + [0, 0, self.CLEAR * h], _yaw(self.poses["src"])), Q["src"])[0]
            if name == "up_src":
                a, state = self.a_close, "held"
        elif name == "close":
            q, a = Q["src"], self.a_open + (self.a_close - self.a_open) * u
            state = "held" if u > 0.6 else "src"
        elif name == "to_dst":
            q, a, state = Q["src_up"] + (Q["dst_up"] - Q["src_up"]) * u, self.a_close, "held"
        elif name in ("down_dst", "up_dst"):
            h = (1 - u) if name == "down_dst" else u
            q = ik(target(self.g_dst + [0, 0, self.CLEAR * h], _yaw(self.poses["dst"])), Q["dst"])[0]
            a, state = (self.a_close, "held") if name == "down_dst" else (self.a_open, "dst")
        elif name == "open":
            q, a, state = Q["dst"], self.a_close + (self.a_open - self.a_close) * u, ("held" if u < 0.3 else "dst")
        else:
            q, state = Q["dst_up"] + (Q["home"] - Q["dst_up"]) * u, "dst"
        out = (q, a, state)
        self._cache[k] = out
        return out

    def held_pose(self, q):
        """Item centre pose while held: hangs from the tcp, offset back up by the grasp height."""
        T = fk(q)["tcp"].copy()
        T[:3, 3] -= np.array([0, 0, self.item[2] * 0.15])
        return T


def _yaw(T):
    y = T[:3, 1]
    return math.atan2(y[1], y[0])
