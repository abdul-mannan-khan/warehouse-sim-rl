"""Draw titles, clock, KPIs and labels onto the Isaac Sim frames (in place).

    python overlay.py frames/            # reads every hud_*.json in the folder
"""
import glob
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

d = sys.argv[1]
FR, META = {}, None
for p in sorted(glob.glob(os.path.join(d, "hud_*.json"))):
    j = json.load(open(p))
    META = j["meta"]
    for h in j["frames"]:
        FR[h["frame"]] = h
W, H, FPS, N = META["W"], META["H"], META["fps"], META["n_frames"]
s = W / 1920
SHOT_START, acc = [], 0
for sh in META["shots"]:
    SHOT_START.append(acc)
    acc += int(round(sh[1] * FPS))

REG = next((p for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                        "C:/Windows/Fonts/segoeui.ttf"] if os.path.exists(p)), None)
BOLD = next((p for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                         "C:/Windows/Fonts/segoeuib.ttf"] if os.path.exists(p)), None)


def font(px, bold=False):
    f = BOLD if bold else REG
    return ImageFont.truetype(f, int(px * s)) if f else ImageFont.load_default(size=int(px * s))


F = {"title": font(26, True), "sub": font(19), "clock": font(44, True), "rate": font(18, True),
     "cap": font(30, True), "capsub": font(19), "kpi": font(30, True), "kpil": font(16), "lab": font(20, True),
     "big": font(62, True), "bigsub": font(28), "end": font(34), "endb": font(40, True)}
WHITE, GREY, ACCENT = (255, 255, 255, 255), (190, 198, 210, 255), (118, 185, 0, 255)   # NVIDIA-ish green accent
PANEL = (10, 13, 18, 175)
TOTE = {"normal": (40, 110, 245, 255), "rush": (240, 50, 40, 255), "pick": (40, 215, 90, 255), "idle": (140, 145, 150, 255)}


def S(v):
    return int(v * s)


def panel(dr, box, r=12, fill=PANEL):
    dr.rounded_rectangle([S(box[0]), S(box[1]), S(box[2]), S(box[3])], radius=S(r), fill=fill)


def text(dr, xy, t, f, fill=WHITE, anchor="la"):
    dr.text((S(xy[0]), S(xy[1])), t, font=f, fill=fill, anchor=anchor)


def clock(t):
    m = int(round(t))
    return f"{m // 60:02d}:{m % 60:02d}"


def draw(im, h):
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    f = h["frame"]
    # header
    t1 = "Warehouse picking simulation  \u2022  NVIDIA Isaac Sim replay"
    t2 = f"SimPy discrete-event model  \u2022  dispatch policy: {META['policy']}"
    hw = max(dr.textlength(t1, font=F["title"]), dr.textlength(t2, font=F["sub"])) / s
    panel(dr, (28, 24, 52 + hw + 28, 108))                # sized to the text
    dr.rectangle([S(28), S(24), S(34), S(108)], fill=ACCENT)
    text(dr, (52, 36), t1, F["title"])
    text(dr, (52, 74), t2, F["sub"], GREY)
    # clock + time-lapse rate
    panel(dr, (1640, 24, 1892, 128))
    text(dr, (1866, 32), "SHIFT TIME", F["kpil"], GREY, "ra")
    text(dr, (1866, 52), clock(h["t"]), F["clock"], WHITE, "ra")
    rate = h["rate"]
    rtxt = f"{rate:,.0f}x time-lapse" if rate >= 1.5 else "real time"
    text(dr, (1866, 104), rtxt, F["rate"], ACCENT, "ra")
    # KPI panel
    panel(dr, (1400, 820, 1892, 1052))
    k = [("Orders done", f"{h['done']}"), ("Late", f"{h['late']}"), ("Mean cycle", f"{h['ct']:.1f} min")]
    for i, (lab, val) in enumerate(k):
        x = 1424 + i * 150
        text(dr, (x, 836), lab, F["kpil"], GREY)
        text(dr, (x, 856), val, F["kpi"])
    text(dr, (1424, 912), "Orders waiting per zone", F["kpil"], GREY)
    mx = max(12, max(h["waiting"]))
    for z, w in enumerate(h["waiting"]):
        y = 938 + z * 26
        text(dr, (1424, y), f"Z{z + 1}", F["kpil"], GREY)
        dr.rounded_rectangle([S(1460), S(y + 3), S(1460 + 370), S(y + 17)], radius=S(5), fill=(255, 255, 255, 40))
        if w:
            dr.rounded_rectangle([S(1460), S(y + 3), S(1460 + 370 * w / mx), S(y + 17)], radius=S(5),
                                 fill=(255, 180, 40, 230))
        text(dr, (1846, y), str(w), F["kpil"], WHITE)
    # caption (lower third) + tote legend
    cap = h["caption"]
    fc, px = F["cap"], 30
    while px > 20 and dr.textlength(cap, font=fc) / s > 1300:   # shrink a long caption to stay clear of the KPIs
        px -= 1
        fc = font(px, True)
    tw = dr.textlength(cap, font=fc) / s
    panel(dr, (28, 950, 28 + max(tw, 640) + 48, 1052))
    dr.rectangle([S(28), S(950), S(34), S(1052)], fill=ACCENT)
    text(dr, (52, 962 + (30 - px) / 2), cap, fc)
    lx = 52
    for key, lab in [("normal", "normal order"), ("rush", "rush order"), ("pick", "being picked"), ("idle", "empty")]:
        dr.rounded_rectangle([S(lx), S(1014), S(lx + 22), S(1032)], radius=S(4), fill=TOTE[key])
        text(dr, (lx + 30, 1012), lab, F["capsub"], GREY)
        lx += 30 + dr.textlength(lab, font=F["capsub"]) / s + 28
    # labels
    for lab, (x, y) in h["labels"].items():
        tw = dr.textlength(lab, font=F["lab"])
        x0, y0 = x - tw / 2 - S(12), y - S(40)
        if y0 < S(140) or y > S(930):                  # would overlap the header, clock or lower panels
            continue
        dr.rounded_rectangle([x0, y0, x0 + tw + S(24), y0 + S(32)], radius=S(8), fill=(10, 13, 18, 200))
        dr.line([x, y0 + S(32), x, y - S(4)], fill=ACCENT, width=max(1, S(2)))
        dr.ellipse([x - S(5), y - S(5), x + S(5), y + S(5)], fill=ACCENT)
        dr.text((x, y0 + S(16)), lab, font=F["lab"], fill=WHITE, anchor="mm")
    out = Image.alpha_composite(im, ov)

    # title card over the first seconds, end card over the last seconds, dips to black between shots
    fx = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dx = ImageDraw.Draw(fx)
    title_a = max(0.0, min(1.0, (3.2 * FPS - f) / (0.8 * FPS)))
    if title_a > 0:
        dx.rectangle([0, 0, W, H], fill=(6, 8, 12, int(200 * title_a)))
        a = int(255 * title_a)
        dx.text((W / 2, H / 2 - S(70)), "Robot-assisted warehouse picking", font=F["big"], fill=(255, 255, 255, a), anchor="mm")
        dx.text((W / 2, H / 2 + S(10)), "One 8-hour shift from a SimPy simulator, replayed in NVIDIA Isaac Sim",
                font=F["bigsub"], fill=(200, 208, 220, a), anchor="mm")
        dx.text((W / 2, H / 2 + S(60)), f"{META['n_robots']} iw.hub robots  \u2022  {META['nz']} picking zones  \u2022  "
                f"dispatch policy: {META['policy']}", font=F["bigsub"], fill=(118, 185, 0, a), anchor="mm")
    end_a = max(0.0, min(1.0, (f - (N - 3.5 * FPS)) / (0.8 * FPS)))
    if end_a > 0:
        dx.rectangle([0, 0, W, H], fill=(6, 8, 12, int(215 * end_a)))
        a = int(255 * end_a)
        late_pct = 100.0 * h["late"] / max(h["done"], 1)
        dx.text((W / 2, H / 2 - S(110)), "End of shift", font=F["big"], fill=(255, 255, 255, a), anchor="mm")
        lines = [f"{h['done']} orders completed  \u2022  {late_pct:.1f}% late  \u2022  mean cycle time {h['ct']:.1f} min",
                 "Every robot position comes from the simulator's event trace",
                 "github.com/abdul-mannan-khan/warehouse-sim-rl"]
        for i, ln in enumerate(lines):
            dx.text((W / 2, H / 2 - S(10) + S(56) * i), ln, font=F["endb"] if i == 0 else F["end"],
                    fill=(255, 255, 255, a) if i == 0 else (200, 208, 220, a), anchor="mm")
    dip = 0.0
    for st in SHOT_START[1:]:
        dd = abs(f - (st - 0.5))
        if dd < 6:
            dip = max(dip, 1.0 - dd / 6)
    if dip > 0:
        dx.rectangle([0, 0, W, H], fill=(0, 0, 0, int(255 * dip ** 1.5)))
    return Image.alpha_composite(out, fx)


for f in sorted(FR):
    p = os.path.join(d, f"{f:04d}.png")
    if not os.path.exists(p):
        continue
    im = Image.open(p).convert("RGBA")
    draw(im, FR[f]).convert("RGB").save(os.path.join(d, f"o{f:04d}.png") if "--keep" in sys.argv else p)
print("overlay done", len(FR))
