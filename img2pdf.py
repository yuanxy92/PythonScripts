import os
from PIL import Image
from natsort import natsorted

def images_to_pdf(
    image_dir,
    output_pdf,
    dpi=300,
    extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff")
):
    image_files = natsorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith(extensions)
    )

    if not image_files:
        raise ValueError("No image files found.")

    pdf_pages = []

    for fname in image_files:
        path = os.path.join(image_dir, fname)
        img = Image.open(path)

        # 统一转为 RGB（PDF 不支持 RGBA / P）
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 设置 DPI（不改变像素，只是物理尺寸映射）
        img.info["dpi"] = (dpi, dpi)
        pdf_pages.append(img)

    # 保存 PDF
    pdf_pages[0].save(
        output_pdf,
        save_all=True,
        append_images=pdf_pages[1:],
        resolution=dpi
    )

    print(f"Saved PDF to: {output_pdf}")

images_to_pdf('/Users/yuanxy/Backup/祁比亚', '/Users/yuanxy/Backup/祁比亚.pdf')