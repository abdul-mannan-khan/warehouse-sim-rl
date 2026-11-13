"""Add clock, KPIs and labels to the raw Isaac Sim frames (in place).  python overlay.py frames/ [hud.json]"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

d = sys.argv[1]
H = json.load(open(os.path.join(d, sys.argv[2] if len(sys.argv) > 2 else "hud.json")))
L = H["labels"]
FONT = next((p for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if os.path.exists(p)), None)


def font(px, bold=False):
    if FONT:
        return ImageFont.truetype(FONT.replace("Sans.ttf", "Sans-Bold.ttf") if bold else FONT, px)
    return ImageFont.load_default(size=px)


s = L["W"] / 1920
F_T, F_S, F_K, F_L, F_Z = font(int(44 * s), True), font(int(26 * s)), font(int(30 * s)), font(int(21 * s)), font(int(34 * s), True)


def shadowed(dr, xy, text, f, fill=(255, 255, 255)):
    dr.text((xy[0] + 2, xy[1] + 2), text, font=f, fill=(0, 0, 0, 200))
    dr.text(xy, text, font=f, fill=fill)


for h in H["frames"]:
    name = "preview.png" if len(H["frames"]) == 1 and os.path.exists(os.path.join(d, "preview.png")) else f"{h['frame']:04d}.png"
    p = os.path.join(d, name)
    im = Image.open(p).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    W, Hh = im.size
    dr.rectangle([0, 0, W, int(118 * s)], fill=(8, 10, 14, 170))
    dr.rectangle([0, Hh - int(92 * s), W, Hh], fill=(8, 10, 14, 190))
    shadowed(dr, (int(40 * s), int(18 * s)), "SimPy warehouse simulator, replayed in NVIDIA Isaac Sim", F_T)
    shadowed(dr, (int(40 * s), int(72 * s)), f"Dispatch policy: {L['policy']}   |   {L['n_robots']} iw.hub robots, {L['nz']} zones", F_S)
    t = h["t"]
    shadowed(dr, (W - int(300 * s), int(24 * s)), f"t = {int(t) // 60:02d}:{int(t) % 60:02d}", F_T)
    shadowed(dr, (int(40 * s), Hh - int(84 * s)),
             f"done {h['done']}    late {h['late']}    mean cycle time {h['ct']:4.1f} min    waiting {h['waiting']}", F_K)
    shadowed(dr, (int(40 * s), Hh - int(38 * s)),
             "blue tote = normal order    red = rush order    green = being picked    cubes = orders waiting", F_L,
             (215, 220, 230))
    for z, (x, y) in enumerate(L["zones"]):
        shadowed(dr, (x, y), f"Zone {z + 1}", F_Z)
    shadowed(dr, tuple(L["depot"]), "Depot + packing", F_Z)
    Image.alpha_composite(im, ov).convert("RGB").save(p)
print("overlay done", len(H["frames"]))
