"""Draw the USE + binary-feedback architecture figure in the style of the USE figure.

The input stage (prompt, classes, weak/strong views, frozen encoders, view
selection) and the stock glyphs (Sigma node, math labels, legend) are taken
from the original USE figure so the style matches exactly. The sort-entropy
panel is dropped, and the confidence gate, binary-feedback query and
REINFORCE loss are drawn in the same visual language.

Usage:
    python utils/make_feedback_arch.py report/figures/use_architecture.png \
        report/figures/use_feedback_architecture.png
Fonts: Comic Neue (SIL OFL) in utils/fonts/, a stand-in for the Comic Sans
used in the original figure.
"""
import math
import os
import sys
from PIL import Image, ImageDraw, ImageFont

src, out = sys.argv[1], sys.argv[2]
orig = Image.open(src).convert("RGB")          # 4034 x 1491
H = orig.size[1]
W = 4500

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
comic = lambda s: ImageFont.truetype(os.path.join(FONT_DIR, "ComicNeue-Bold.ttf"), s)
math_it = lambda s: ImageFont.truetype("/usr/share/fonts/gsfonts/NimbusRoman-Italic.otf", s)
math_rm = lambda s: ImageFont.truetype("/usr/share/fonts/gsfonts/NimbusRoman-Regular.otf", s)

PANEL = (242, 242, 242)
GRAY = (127, 127, 127)
PURPLE = (112, 48, 160)
GREEN = (59, 95, 33)
GREEN_FILL = (227, 242, 217)
ORANGE = (214, 96, 20)
ORANGE_FILL = (252, 228, 204)
BLACK = (0, 0, 0)

img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

# ---------------------------------------------------------------- background
d.rounded_rectangle([1300, 20, W - 22, 1463], radius=40, fill=PANEL)
img.paste(orig.crop((0, 0, 2470, H)), (0, 0))          # input stage + selection


def paste(box, at):
    """Copy a region of the original figure to position `at` (top-left)."""
    img.paste(orig.crop(box), at)


# ---------------------------------------------------------------- drawing kit
def head(p_from, p_to, color, width=6, size=30):
    """Open V arrow head, as in the original figure."""
    ang = math.atan2(p_to[1] - p_from[1], p_to[0] - p_from[0])
    for s in (+1, -1):
        a = ang + s * math.radians(145)
        d.line([p_to, (p_to[0] + size * math.cos(a), p_to[1] + size * math.sin(a))],
               fill=color, width=width)


def path(pts, color, width=6, dash=None, arrow=True):
    for p, q in zip(pts, pts[1:]):
        if dash is None:
            d.line([p, q], fill=color, width=width)
            continue
        on, off = dash
        L = math.dist(p, q)
        ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
        t = 0.0
        while t < L:
            t1 = min(t + on, L)
            d.line([(p[0] + ux * t, p[1] + uy * t), (p[0] + ux * t1, p[1] + uy * t1)],
                   fill=color, width=width)
            t += on + off
    if arrow:
        head(pts[-2], pts[-1], color, width)


GREEN_DASH = (21, 14)
PURPLE_DASH = (28, 19)


def text(xy, s, font, fill=BLACK, anchor="mm"):
    d.text(xy, s, font=font, fill=fill, anchor=anchor)


def rich(cx, cy, segs, fill=BLACK):
    """segs: (text, font, dy) pieces laid out left to right, centred on cx."""
    widths = [d.textlength(t, font=f) for t, f, _ in segs]
    x = cx - sum(widths) / 2
    for (t, f, dy), w in zip(segs, widths):
        d.text((x, cy + dy), t, font=f, fill=fill, anchor="lm")
        x += w


# ---------------------------------------------------------------- SE node
SX, SY, SR = 2800, 1035, 88
paste((3424, 945, 3610, 1126), (SX - 91, SY - 90))        # Sigma node
d.rectangle([SX - 8, SY - SR - 20, SX + 8, SY - SR + 2], fill=PANEL)  # stub of the old arrow
for x in range(SX + 60, SX + 100):                         # stub of the old prediction arrow
    for y in range(SY - 20, SY + 30):
        r, g, b_ = img.getpixel((x, y))
        if b_ - g > 40:
            img.putpixel((x, y), PANEL)
path([(2462, 1003), (SX - SR - 8, 1003)], GRAY)
path([(2462, 1068), (SX - SR - 8, 1068)], PURPLE, dash=PURPLE_DASH)

# Prediction (inference only), down and to the right of the Sigma node.
PY = 1330
path([(SX, SY + SR + 4), (SX, PY), (3010, PY)], PURPLE, dash=PURPLE_DASH)
paste((3628, 945, 3892, 1008), (3035, PY - 75))           # "Prediction"
paste((3690, 1075, 3835, 1160), (3080, PY - 5))           # \hat p(x)

# ---------------------------------------------------------------- confidence gate
GX0, GX1, GY0, GY1 = 2990, 3550, 915, 1155
GCX = (GX0 + GX1) / 2
d.rounded_rectangle([GX0, GY0, GX1, GY1], radius=18, fill="white", outline=GRAY, width=5)
text((GCX, GY0 + 62), "Confidence Gate", comic(62))
rich(GCX, GY0 + 160, [("H", math_it(64), 0), ("(", math_rm(64), 0), ("q", math_it(64), 0),
                      (") / log ", math_rm(64), 0), ("C", math_it(64), 0),
                      ("  vs.  ", math_rm(64), 0), ("τ", math_it(64), 0),
                      ("fb", math_rm(42), 22)])
text((GCX, GY1 + 45), "each sample takes exactly one branch", comic(46), GRAY)
path([(SX + SR + 6, SY), (GX0 - 6, SY)], GRAY)

# ---------------------------------------------------------------- confident: RCE
RCE = (GCX - 340, 213)
paste((3178, 213, 3858, 422), (int(RCE[0]), RCE[1]))      # Reverse Cross Entropy box
path([(2462, 318), (RCE[0] - 4, 318)], GRAY)              # selected views
paste((2625, 222, 3025, 295), (2505, 222))                # "Selected Views"
paste((2755, 340, 2930, 420), (2620, 340))                # \bar p_theta(x)

path([(GCX, GY0 - 4), (GCX, 427)], GRAY)                  # pseudo label q(x) to RCE
paste((3330, 570, 3468, 662), (int(GCX) - 160, 575))      # q(x)
paste((3540, 552, 3725, 688), (int(GCX) + 25, 555))       # "Pseudo Label"
text((GCX + 30, 745), "confident", comic(56), BLACK, anchor="lm")
rich(GCX + 30 + 110, 815, [("≤ ", math_rm(60), 0), ("τ", math_it(60), 0), ("fb", math_rm(40), 18)])

# ---------------------------------------------------------------- uncertain: feedback
FX0, FX1, FY0, FY1 = 3800, 4380, 915, 1155
FCX = (FX0 + FX1) / 2
d.rounded_rectangle([FX0, FY0, FX1, FY1], radius=18, fill=ORANGE_FILL, outline=ORANGE, width=5)
text((FCX, FY0 + 58), "Binary Feedback", comic(62))
rich(FCX, FY0 + 130, [("Is ", comic(58), 0), ("ŷ", math_it(66), 0), (" correct?", comic(58), 0)])
rich(FCX, FY0 + 196, [("r", math_it(64), 0), (" = +1 / −1", math_rm(64), 0)])
path([(GX1 + 6, SY), (FX0 - 6, SY)], ORANGE)
text(((GX1 + FX0) / 2, SY - 48), "uncertain", comic(56), ORANGE)
rich((GX1 + FX0) / 2, SY + 50, [("> ", math_rm(60), 0), ("τ", math_it(60), 0), ("fb", math_rm(40), 18)], ORANGE)

# ---------------------------------------------------------------- REINFORCE loss
RX0, RX1, RY0, RY1 = 3800, 4380, 213, 421
d.rectangle([RX0, RY0, RX1, RY1], fill=GREEN_FILL, outline=GREEN, width=5)
text((FCX, RY0 + 62), "REINFORCE Loss", comic(62), GREEN)
rich(FCX, RY0 + 145, [("−", math_rm(62), 0), ("r", math_it(62), 0), (" log ", math_rm(62), 0),
                      ("p", math_it(62), 0), ("θ", math_it(42), 20), ("(", math_rm(62), 0),
                      ("ŷ", math_it(62), 0), (" | ", math_rm(62), 0), ("A", math_it(62), 0),
                      ("0", math_rm(42), 20), ("(", math_rm(62), 0), ("x", math_it(62), 0),
                      ("))", math_rm(62), 0)], GREEN)
path([(FCX, FY0 - 4), (FCX, RY1 + 6)], ORANGE)
text((FCX - 30, 620), "replaces", comic(54), ORANGE, anchor="rm")
text((FCX - 30, 690), "RCE", comic(54), ORANGE, anchor="rm")
text((FCX + 30, 650), "reward", comic(54), ORANGE, anchor="lm")
d.text((FCX + 40 + d.textlength("reward", font=comic(54)), 650), "r", font=math_it(60), fill=ORANGE, anchor="lm")

# ---------------------------------------------------------------- optimize paths
TOP = 48
path([(2468, TOP), (FCX, TOP)], GREEN, width=5, dash=GREEN_DASH, arrow=False)
path([(GCX, RY0), (GCX, TOP)], GREEN, width=5, dash=GREEN_DASH, arrow=False)
path([(FCX, RY0), (FCX, TOP)], GREEN, width=5, dash=GREEN_DASH, arrow=False)
paste((3570, 100, 3810, 170), (int(GCX) + 25, 100))       # "Optimize"
paste((3570, 100, 3810, 170), (int(FCX) + 25, 100))

# ---------------------------------------------------------------- legend
LX, LY = 3700, 1195
paste((3350, 1232, 3920, 1395), (LX, LY))                 # Adaptation / Inference
ly3 = LY + 36 + 2 * 87 - 2
path([(LX + 7, ly3), (LX + 240, ly3)], ORANGE)
text((LX + 285, ly3), "Feedback", comic(66), ORANGE, anchor="lm")

img.save(out, optimize=True)
print(out, img.size)
