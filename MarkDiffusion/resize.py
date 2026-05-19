#!/usr/bin/env python3
import os
from PIL import Image, ImageOps

TARGET_SIZE = (256, 256)

ROOTS = [
    "/home/nisp/Jie/security26/Attacked/Ctrlgen/SVD_VideoMark",
    "/home/nisp/Jie/security26/Attacked/Ctrlgen/SVD_VideoShield",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def is_image(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in IMG_EXTS

def resize_inplace(img_path: str) -> bool:
    """
    Resize image to 256x256 and overwrite.
    Returns True if modified, False otherwise.
    """
    try:
        with Image.open(img_path) as im:
            # fix EXIF orientation for JPEGs etc.
            im = ImageOps.exif_transpose(im)

            # Convert to RGB if needed (avoid issues for some formats)
            # Keep alpha for PNG/WebP by using RGBA
            has_alpha = (im.mode in ("RGBA", "LA")) or ("transparency" in im.info)
            if has_alpha:
                im = im.convert("RGBA")
            else:
                im = im.convert("RGB")

            if im.size == TARGET_SIZE:
                return False

            im = im.resize(TARGET_SIZE, resample=Image.Resampling.LANCZOS)

            # Preserve original format by saving with same extension
            ext = os.path.splitext(img_path.lower())[1]
            save_kwargs = {}

            if ext in (".jpg", ".jpeg"):
                # JPEG doesn't support alpha
                if im.mode == "RGBA":
                    im = im.convert("RGB")
                save_kwargs.update({"quality": 95, "subsampling": 0, "optimize": True})

            im.save(img_path, **save_kwargs)
            return True
    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")
        return False

def main():
    total = 0
    changed = 0

    for root in ROOTS:
        if not os.path.isdir(root):
            print(f"[WARN] Not a directory: {root}")
            continue

        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                if not is_image(fp):
                    continue
                total += 1
                if resize_inplace(fp):
                    changed += 1

    print(f"Done. Found {total} images, resized {changed} to {TARGET_SIZE[0]}x{TARGET_SIZE[1]}.")

if __name__ == "__main__":
    main()