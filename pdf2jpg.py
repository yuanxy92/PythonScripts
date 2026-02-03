import os
from pathlib import Path
from pdf2image import convert_from_path

def convert_pdfs_to_jpgs(folder_path, dpi=300):
    folder = Path(folder_path)
    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    for pdf_file in pdf_files:
        print(f"Converting: {pdf_file.name}")

        try:
            images = convert_from_path(str(pdf_file), dpi=dpi)
            for i, image in enumerate(images):
                output_filename = f"{pdf_file.stem}_page{i+1}.jpg"
                output_path = folder / output_filename
                image.save(output_path, "JPEG")
            print(f"Saved {len(images)} page(s) from {pdf_file.name}")
        except Exception as e:
            print(f"Failed to convert {pdf_file.name}: {e}")

# 用法
if __name__ == "__main__":
    folder_path = "/Users/yuanxy/Drives/aDriveSync/MydocumentsADrive/补充医疗/2025/Jun/新冠仁济南院"  # ← 修改为目标文件夹路径
    convert_pdfs_to_jpgs(folder_path)
