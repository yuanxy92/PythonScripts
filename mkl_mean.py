import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def adjust_brightness_hsv(img, brightness_scale=0.5):
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 2] *= brightness_scale
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return adjusted


# def compute_statistics(image_paths, sample_per_image=1000, brightness_scale=None, region_size=150):
#     all_pixels = []

#     for path in tqdm(image_paths, desc="Sampling colors"):
#         img = cv2.imread(path)[..., ::-1] / 255.0  # 转成RGB并归一化
#         img = cv2.resize(img,(400,400))

#         if brightness_scale is not None:
#             img = adjust_brightness_hsv(img, brightness_scale)

#         h, w, _ = img.shape

#         # 计算中心区域范围
#         center_x = w // 2
#         center_y = h // 2
#         half_size = region_size // 2

#         x_start = max(center_x - half_size, 0)
#         x_end = min(center_x + half_size, w)
#         y_start = max(center_y - half_size, 0)
#         y_end = min(center_y + half_size, h)

#         # 裁剪中心区域
#         region = img[y_start:y_end, x_start:x_end, :]

#         rh, rw, _ = region.shape
#         n = min(sample_per_image, rh * rw)

#         # 在中心区域内采样
#         idx = np.random.choice(rh * rw, n, replace=False)
#         coords = np.unravel_index(idx, (rh, rw))
#         pixels = region[coords]

#         all_pixels.append(pixels)

#     all_pixels = np.concatenate(all_pixels, axis=0)
#     mean = np.mean(all_pixels, axis=0)
#     cov = np.cov(all_pixels.T)
#     return mean, cov

def compute_statistics_with_pairs(input_paths, gt_paths, sample_per_image=5000, brightness_scale=None, region_size=380):
    assert len(input_paths) == len(gt_paths), "Input and GT image lists must be the same length."

    all_input = []
    all_gt = []

    for inp_path, gt_path in tqdm(zip(input_paths, gt_paths), total=len(input_paths), desc="Sampling color pairs"):
        img_in = cv2.imread(inp_path)[..., ::-1] / 255.0  # RGB
        img_gt = cv2.imread(gt_path)[..., ::-1] / 255.0

        img_in = cv2.resize(img_in, (400, 400))
        img_gt = cv2.resize(img_gt, (400, 400))

        if brightness_scale is not None:
            img_gt = adjust_brightness_hsv(img_gt, brightness_scale)

        h, w, _ = img_in.shape

        # 中心区域范围
        cx, cy = w // 2, h // 2
        hs = region_size // 2
        x0, x1 = max(cx - hs, 0), min(cx + hs, w)
        y0, y1 = max(cy - hs, 0), min(cy + hs, h)

        region_in = img_in[y0:y1, x0:x1, :]
        region_gt = img_gt[y0:y1, x0:x1, :]

        rh, rw, _ = region_in.shape
        n = min(sample_per_image, rh * rw)

        idx = np.random.choice(rh * rw, n, replace=False)
        coords = np.unravel_index(idx, (rh, rw))

        sampled_in = region_in[coords]
        sampled_gt = region_gt[coords]

        all_input.append(sampled_in)
        all_gt.append(sampled_gt)

    all_input = np.concatenate(all_input, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    mean_input = np.mean(all_input, axis=0)
    mean_gt = np.mean(all_gt, axis=0)
    cov_input = np.cov(all_input.T)
    cov_gt = np.cov(all_gt.T)

    return mean_input, cov_input, mean_gt, cov_gt



def monge_kantorovich_linear(mean_src, cov_src, mean_tgt, cov_tgt):
    # MKL transform: T = B^{1/2} A^{-1/2}
    eigvals_src, eigvecs_src = np.linalg.eigh(cov_src)
    eigvals_tgt, eigvecs_tgt = np.linalg.eigh(cov_tgt)

    sqrt_src = eigvecs_src @ np.diag(np.sqrt(eigvals_src)) @ eigvecs_src.T
    inv_sqrt_src = np.linalg.inv(sqrt_src)

    sqrt_tgt = eigvecs_tgt @ np.diag(np.sqrt(eigvals_tgt)) @ eigvecs_tgt.T

    T = sqrt_tgt @ inv_sqrt_src
    return T

def apply_color_transform(img, T, mean_src, mean_tgt):
    h, w, c = img.shape
    flat = img.reshape(-1, 3)
    flat = (flat - mean_src) @ T.T + mean_tgt
    flat = np.clip(flat, 0, 1)
    return flat.reshape(h, w, 3)

def filter_lists_by_common_basenames(list1, list2):
    # Get base names without extension
    base1 = {os.path.splitext(os.path.basename(p))[0] for p in list1}
    base2 = {os.path.splitext(os.path.basename(p))[0] for p in list2}

    # Find intersection of base names
    common_basenames = base1 & base2

    # Filter both lists to keep only common base names
    filtered_list1 = [p for p in list1 if os.path.splitext(os.path.basename(p))[0] in common_basenames]
    filtered_list2 = [p for p in list2 if os.path.splitext(os.path.basename(p))[0] in common_basenames]

    return filtered_list1, filtered_list2

def main(lr_dir, gt_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    lr_paths = sorted(glob(os.path.join(lr_dir, "*")))
    gt_paths = sorted(glob(os.path.join(gt_dir, "*")))

    lr_paths, gt_paths = filter_lists_by_common_basenames(lr_paths, gt_paths)

    print(f"Found {len(lr_paths)} LR images, {len(gt_paths)} GT images.")

    # Step 1: Compute color statistics
    print("Computing statistics...")
    # mean_lr, cov_lr = compute_statistics(lr_paths)
    # print(mean_lr)
    # mean_gt, cov_gt = compute_statistics(gt_paths,brightness_scale=0.8)
    # print(mean_gt)
    mean_lr,cov_lr,mean_gt,cov_gt=compute_statistics_with_pairs(lr_paths,gt_paths,brightness_scale=0.8)
    

    # Step 2: Compute MKL transform
    print("Computing MKL color transform...")
    T = monge_kantorovich_linear(mean_lr, cov_lr, mean_gt, cov_gt)

    # Step 3: Apply to LR images
    print("Applying color transform to LR images...")
    for path in tqdm(lr_paths, desc="Transforming"):
        img = cv2.imread(path)[..., ::-1] / 255.0
        corrected = apply_color_transform(img, T, mean_lr, mean_gt)
        fname = os.path.basename(path)
        cv2.imwrite(os.path.join(output_dir, fname), (corrected * 255).astype(np.uint8)[..., ::-1])

    print("Done! Results saved to:", output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_dir", default="/data/jianing/large_sensor/train")
    parser.add_argument("--gt_dir", default="/data/jianing/micro_sensor/dataset_73/train/gt")
    parser.add_argument("--output_dir", default="/data/jianing/corrected_output2", help="Where to save results")
    args = parser.parse_args()

    # main(args.lr_dir, args.gt_dir, args.output_dir)
    main('E:/Data/Metalens/Journal/lr_metalens_3mm_corrected', 
        'E:/Data/Metalens/Journal/hr_images512/hr_images512', 
        'E:/Data/Metalens/Journal/lr_metalens_3mm_corrected_color')
