import tifffile # needs to be 2023 or higher due to OME.TIFF metadata
import numpy as np
import os
from PIL import Image
import cv2
import staintools
import argparse


def extract_channels(imgfile, prefix, output_dirs):
    if os.path.isfile(imgfile):
        print(f"Reading: {imgfile}")
        img = tifffile.imread(imgfile)

        im1 = img[0]
        dapi_file = os.path.join(output_dirs['dapi'], prefix + '_DAPI.png')
        Image.fromarray(im1).save(dapi_file)

        im2 = img[44]
        membrane_file = os.path.join(output_dirs['membrane'], prefix + '_NaKATPase.png')
        Image.fromarray(im2).save(membrane_file)

        im3 = img[45]
        biomarker_file = os.path.join(output_dirs['biomarker'], prefix + '_PCK.png')
        Image.fromarray(im3).save(biomarker_file)

        return dapi_file, membrane_file, biomarker_file
        
def normalize_grayscale(image_path, output_path):
    image = cv2.imread(image_path, 0)
    normalized = cv2.equalizeHist(image)
    cv2.imwrite(output_path, normalized)
    print(f"Saved normalized grayscale image to: {output_path}")


def stain_normalize(target_path, to_transform_path, output_path):
    target = staintools.read_image(target_path)
    to_transform = staintools.read_image(to_transform_path)

    target = staintools.LuminosityStandardizer.standardize(target)
    to_transform = staintools.LuminosityStandardizer.standardize(to_transform)

    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(target)

    transformed = normalizer.transform(to_transform)
    cv2.imwrite(output_path, transformed)
    print(f"Saved stain normalized image to: {output_path}")


def main(args):
    output_dirs = {
        'dapi': args.dapi_out,
        'membrane': args.membrane_out,
        'biomarker': args.biomarker_out,
    }

    os.makedirs(output_dirs['dapi'], exist_ok=True)
    os.makedirs(output_dirs['membrane'], exist_ok=True)
    os.makedirs(output_dirs['biomarker'], exist_ok=True)

    # Step 1: Extract TIFF channels
    dapi_img, membrane_img, biomarker_img = extract_channels(args.input, args.sample_prefix, output_dirs)
    print(dapi_img)
    print(biomarker_img)
    eq_out = os.path.join(output_dirs['membrane'], args.sample_prefix + '_normalized_region_ROI_01_NaKATPase.png')
    normalize_grayscale(membrane_img, eq_out)

    # Step 3: Normalize membrane to another target
    stain_normalize(
        args.membrane_training,
        eq_out,
        os.path.join(output_dirs['membrane'], args.sample_prefix + '_normalized_to_example_membrane_NaKATPase.png'))

    # Step 4: Normalize other DAPI samples
    stain_normalize(
        args.dapi_training,
        dapi_img,
        os.path.join(output_dirs['dapi'], args.sample_prefix + '_normalized_to_example_DAPI.png')
    )

    stain_normalize(
        args.biomarker_training,
        biomarker_img,
        os.path.join(output_dirs['biomarker'], args.sample_prefix + '_normalized_to_example_biomarker.png')
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract channels and apply stain normalization.")
    parser.add_argument('--input', required=True, help='Input OME-TIFF image file')
    parser.add_argument('--sample_prefix', required=True, help='perfix for sample')
    parser.add_argument('--dapi-out', help='Output directory for DAPI channel', default='dapi_img')
    parser.add_argument('--membrane-out', help='Output directory for membrane channel', default='membrane_img')
    parser.add_argument('--biomarker-out', help='Output directory for biomarker channel', default='biomarker_img')
    parser.add_argument('--dapi-training', help='spotfile for DAPI channel', required=True)
    parser.add_argument('--membrane-training', help='spotfile for membrane channel training', required=True)
    parser.add_argument('--biomarker-training', help='spotfile for biomarker training', required=True)    
    args = parser.parse_args()
    main(args)
