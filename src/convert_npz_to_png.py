import numpy as np
import os
from PIL import Image
import argparse


def npz_to_png_and_npz(npz_path, output_dir):
    # Load .npz file with pickle for meta
    data = np.load(npz_path, allow_pickle=True)
    images = data['X']
    metas = data['meta']
    labels = data['y'] 
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, "images")
    npz_dir = os.path.join(output_dir, "npz")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)
    for i, (img, mask, meta) in enumerate(zip(images, labels, metas)):
        # Normalize image if needed
        if img.dtype != np.uint8:
            img = (255 * (img - img.min()) / (img.ptp() + 1e-8)).astype(np.uint8)

        # Handle channel-first: (2, H, W) → (H, W, 2)
        if img.ndim == 3 and img.shape[0] == 2:
            img = np.transpose(img, (1, 2, 0))

        # Add fake white 3rd channel
        if img.ndim == 3 and img.shape[2] == 2:
            h, w, _ = img.shape
            white_channel = np.full((h, w, 1), 255, dtype=np.uint8)
            img = np.concatenate((img, white_channel), axis=2)

        # Final check
        if img.shape[2] != 3:
            raise ValueError(f"Unexpected shape {img.shape} at index {i}")
        print(f"[{i}] Image size: {h} pixels high × {w} pixels wide")
        # Use metadata to create unique filename
        base_name = os.path.splitext(os.path.basename(meta[0]))[0] if meta[0] else f"sample_{i:04d}"
        specimen = str(meta[5]) if meta[5] else "unknown"

        # Sanitize specimen string (remove spaces, etc.)
        specimen = specimen.replace(" ", "_")

        # Create final filename
        sample_name = f"{base_name}_{specimen}" + "_" + str(i)
        
        # Save PNG
        img_path = os.path.join(image_dir, f"{sample_name}.png")
        Image.fromarray(img, 'RGB').save(img_path)
        sample_dir = output_dir + "/" + sample_name
        os.makedirs(sample_dir, exist_ok=True)
        dapi_dir = sample_dir + "/dapi_img"
        membrane_dir = sample_dir + "/membrane_img"
        biomarker_dir = sample_dir + "/biomarker_img"
        os.makedirs(dapi_dir, exist_ok=True)
        os.makedirs(membrane_dir, exist_ok=True)
        os.makedirs(biomarker_dir, exist_ok=True)        
        # Save each channel as individual PNG
        Image.fromarray(img[:, :, 0]).save(os.path.join(dapi_dir, f"{sample_name}_DAPI.png"))
        Image.fromarray(img[:, :, 1]).save(os.path.join(membrane_dir, f"{sample_name}_NaKATPase.png"))
        Image.fromarray(img[:, :, 2]).save(os.path.join(biomarker_dir, f"{sample_name}_marker.png"))

        # Save the image + metadata as .npz
        npz_path_i = os.path.join(npz_dir, f"{sample_name}.npz")
        np.savez_compressed(
            os.path.join(npz_dir, f"{sample_name}.npz"),
            X=img,
            y=mask,
            meta=meta
        )
        
        #np.savez_compressed(npz_path_i, X=img, meta=meta)

        print(f"Saved: {sample_name}.png and {sample_name}.npz")

    print(f"\nAll images and .npz files saved to: {output_dir}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npz arrays to PNG images")
    parser.add_argument('--input', required=True, help='Path to .npz file')
    parser.add_argument('--output-dir', required=True, help='Directory to save PNG images')
    args = parser.parse_args()

    npz_to_png_and_npz(args.input, args.output_dir)
