from pathlib import Path
from PIL import Image, UnidentifiedImageError
import pillow_heif

pillow_heif.register_heif_opener()

root = Path(r"c:/Users/Lenovo/Desktop/动画图片")
convertible_exts = {".png", ".jpeg", ".jpg", ".webp", ".heif", ".heic", ".jfif", ".bmp"}
converted = []
skipped = []
failed = []

for path in root.iterdir():
    if not path.is_file():
        continue
    ext = path.suffix.lower()
    if ext not in convertible_exts:
        continue
    if ext == ".jpg":
        continue
    target = path.with_suffix(".jpg")
    if target.exists():
        skipped.append((path.name, "target exists"))
        continue
    try:
        with Image.open(path) as img:
            img.load()
            has_alpha = ("A" in img.getbands()) or ("transparency" in img.info)
            if has_alpha:
                if img.mode not in ("RGBA", "LA"):
                    img = img.convert("RGBA")
                else:
                    img = img.copy()
                alpha = img.getchannel("A") if "A" in img.getbands() else None
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=alpha)
                out = background
            else:
                out = img.convert("RGB")
            out.save(target, "JPEG", quality=95)
            converted.append((path.name, target.name))
    except UnidentifiedImageError as e:
        failed.append((path.name, f"unidentified image: {e}"))
    except Exception as e:
        failed.append((path.name, str(e)))

print("Converted:")
for src, dst in converted:
    print(f"  {src} -> {dst}")
print("\nSkipped:")
for item in skipped:
    print(f"  {item[0]} ({item[1]})")
print("\nFailed:")
for item in failed:
    print(f"  {item[0]} ({item[1]})")

print(f"\nSummary: converted {len(converted)}, skipped {len(skipped)}, failed {len(failed)}")
