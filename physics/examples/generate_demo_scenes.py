"""Generate deterministic, licence-free scenes for the physics-engine demo."""

from __future__ import annotations

from pathlib import Path
import random

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "demo_images"
WIDTH, HEIGHT = 960, 640


def _base() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), (19, 32, 53))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, WIDTH, 215], fill=(73, 103, 138))
    draw.rectangle([0, 215, WIDTH, HEIGHT], fill=(54, 65, 78))
    return image, draw


def perspective_consistent(path: Path) -> None:
    image, draw = _base()
    vp = (480, 220)
    draw.line([(0, 220), (WIDTH, 220)], fill=(225, 235, 245), width=3)

    for bottom_x in range(-100, WIDTH + 101, 90):
        draw.line([vp, (bottom_x, HEIGHT)], fill=(190, 200, 211), width=3)
    for y in (280, 345, 425, 515, 610):
        left_x = int(vp[0] - (y - vp[1]) * 1.35)
        right_x = int(vp[0] + (y - vp[1]) * 1.35)
        draw.line([(left_x, y), (right_x, y)], fill=(150, 164, 178), width=3)
    for x in (60, 145, 230, 730, 815, 900):
        draw.line([(x, 130), (x, 550)], fill=(235, 223, 190), width=4)
    for x in (60, 145, 730, 815):
        draw.rectangle([x, 245, x + 55, 520], outline=(205, 190, 155), width=3)
    image.save(path)


def perspective_inconsistent(path: Path) -> None:
    image, draw = _base()
    rng = random.Random(2026)
    draw.line([(0, 220), (WIDTH, 220)], fill=(225, 235, 245), width=3)
    # Deliberately fragmented structural lines sampled from many incompatible
    # local projections rather than three coherent vanishing-point families.
    for _ in range(46):
        x1 = rng.randint(25, WIDTH - 180)
        y1 = rng.randint(245, HEIGHT - 45)
        length = rng.randint(85, 230)
        slope = rng.uniform(-1.25, 1.25)
        x2 = min(WIDTH - 20, x1 + length)
        y2 = int(max(230, min(HEIGHT - 20, y1 + slope * (x2 - x1))))
        color = (210, 190 + rng.randint(0, 30), 155)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=4)
    for _ in range(12):
        x = rng.randint(40, WIDTH - 40)
        y = rng.randint(245, HEIGHT - 180)
        draw.line([(x, y), (x + rng.randint(-45, 45), y + rng.randint(100, 260))], fill=(170, 185, 200), width=4)
    image.save(path)


def annotated_scene(path: Path, inconsistent: bool) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), (225, 230, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 385, WIDTH, HEIGHT], fill=(184, 174, 155))
    draw.rectangle([520, 70, 900, 370], fill=(160, 205, 220), outline=(50, 75, 90), width=6)
    for x, y in ((150, 400), (300, 390), (430, 410), (610, 405)):
        draw.rectangle([x - 24, y - 100, x + 24, y], fill=(65, 95, 135))

    shadow_vectors = [((150, 400), (255, 465)), ((300, 390), (405, 455)), ((430, 410), (535, 475)), ((610, 405), (715, 470))]
    if inconsistent:
        shadow_vectors[1] = ((300, 390), (215, 475))
        shadow_vectors[3] = ((610, 405), (670, 315))
    for start, end in shadow_vectors:
        draw.polygon(
            [start, (start[0] + 24, start[1]), (end[0] + 22, end[1] + 8), end],
            fill=(95, 89, 80),
        )

    reflection_pairs = [((590, 135), (830, 135)), ((575, 195), (845, 195)), ((565, 255), (855, 255)), ((550, 320), (870, 320))]
    if inconsistent:
        reflection_pairs[1] = ((575, 195), (815, 285))
        reflection_pairs[3] = ((550, 320), (760, 105))
    for object_point, reflected_point in reflection_pairs:
        radius = 7
        draw.ellipse([object_point[0] - radius, object_point[1] - radius, object_point[0] + radius, object_point[1] + radius], fill=(16, 110, 190))
        draw.ellipse([reflected_point[0] - radius, reflected_point[1] - radius, reflected_point[0] + radius, reflected_point[1] + radius], outline=(16, 110, 190), width=3)
    image.save(path)


def _symbol(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    color: tuple[int, int, int],
    index: int,
) -> None:
    x, y = center
    draw.rounded_rectangle(
        [x - 24, y - 20, x + 24, y + 20],
        radius=7,
        fill=color,
        outline=(25, 36, 48),
        width=3,
    )
    if index % 2:
        draw.line([(x - 17, y - 12), (x + 17, y + 12)], fill=(250, 245, 225), width=5)
        draw.line([(x - 17, y + 12), (x + 17, y - 12)], fill=(250, 245, 225), width=5)
    else:
        draw.ellipse([x - 10, y - 10, x + 10, y + 10], fill=(250, 245, 225))


def automatic_scene(path: Path, inconsistent: bool) -> None:
    """Scene designed to exercise model-proposed regions and correspondences."""

    image = Image.new("RGB", (WIDTH, HEIGHT), (220, 228, 234))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 380, WIDTH, HEIGHT], fill=(188, 178, 157))
    draw.rectangle(
        [550, 50, 925, 345],
        fill=(151, 198, 216),
        outline=(39, 62, 75),
        width=9,
    )

    colors = ((210, 65, 72), (53, 145, 92), (220, 145, 45), (105, 75, 190))
    direct = ((130, 95), (225, 165), (320, 235), (415, 305))
    reflected = [(645, 95), (735, 165), (825, 235), (880, 305)]
    if inconsistent:
        reflected[1] = (735, 275)
        reflected[3] = (865, 105)
    for index, (source, destination) in enumerate(zip(direct, reflected)):
        _symbol(draw, source, colors[index], index)
        _symbol(draw, destination, colors[index], index)

    contacts = ((105, 430), (315, 425), (525, 435), (735, 425))
    tips = [(190, 500), (400, 495), (610, 505), (820, 495)]
    if inconsistent:
        tips[1] = (245, 515)
        tips[3] = (780, 350)
    for index, (contact, tip) in enumerate(zip(contacts, tips)):
        x, y = contact
        draw.rectangle([x - 25, y - 92, x + 25, y], fill=colors[index])
        draw.polygon(
            [contact, (contact[0] + 22, contact[1]), (tip[0] + 18, tip[1] + 8), tip],
            fill=(91, 84, 75),
        )
    image.save(path)


def no_geometry(path: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = image.load()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            pixels[x, y] = (
                int(45 + 75 * x / WIDTH),
                int(80 + 70 * y / HEIGHT),
                int(120 + 45 * (x + y) / (WIDTH + HEIGHT)),
            )
    image.save(path)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    perspective_consistent(OUTPUT / "perspective_consistent.png")
    perspective_inconsistent(OUTPUT / "perspective_inconsistent.png")
    annotated_scene(OUTPUT / "annotated_consistent.png", inconsistent=False)
    annotated_scene(OUTPUT / "annotated_inconsistent.png", inconsistent=True)
    automatic_scene(OUTPUT / "automatic_consistent.png", inconsistent=False)
    automatic_scene(OUTPUT / "automatic_inconsistent.png", inconsistent=True)
    no_geometry(OUTPUT / "no_geometry.png")
    print(f"Generated seven demo scenes in {OUTPUT}")


if __name__ == "__main__":
    main()
