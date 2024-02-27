import numpy as np
import SimpleITK as sitk
import nibabel as nib
import argparse
import os
from pathlib import Path
from tqdm import tqdm

def resample_image(image, new_spacing, reference_image):
    """Resamples an image to a new spacing, using the reference image to determine the size of the output image.

    Args:
      image: The image to resample.
      new_spacing: The new spacing of the image.
      reference_image: The reference image, which is used to determine the size of the output image.

    Returns:
      The resampled image.
    """
    output_origin = reference_image.GetOrigin()
    output_spacing = new_spacing
    output_direction = reference_image.GetDirection()

    resampled_image = sitk.Resample(
        image,
        reference_image.GetSize(),
        sitk.Transform(),
        sitk.sitkBSpline,
        output_origin,
        output_spacing,
        output_direction,
    )

    resampled_array = sitk.GetArrayFromImage(resampled_image)

    return np.transpose(resampled_array, (2, 1, 0))

def preprocess_nifti_images(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "PET"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "CT"), exist_ok=True)

    all_files = os.listdir(in_dir)
    pet_files = sorted(list(filter(lambda s: "PET" in s, all_files)))
    ct_files = sorted(list(filter(lambda s: "CT" in s, all_files)))

    assert len(ct_files) == len(pet_files)

    for ct_filename, pet_filename in tqdm(zip(ct_files, pet_files)):
        assert ct_filename.replace("CT", "") == pet_filename.replace("PET", "")

        ct_path = os.path.join(in_dir, ct_filename)
        pet_path = os.path.join(in_dir, pet_filename)

        pet_nib = nib.load(pet_path)

        ct_image = sitk.ReadImage(ct_path)
        pet_image = sitk.ReadImage(pet_path)

        pet_spacing = pet_image.GetSpacing()

        ct_downsampled_image = resample_image(
            ct_image, pet_spacing, pet_image
        )

        ct_hu_min, ct_hu_max = -200, 1000
        pet_hu_min, pet_hu_max = 0, 100

        clipped_ct_data = np.clip(ct_downsampled_image, ct_hu_min, ct_hu_max)
        clipped_pet_data = np.clip(pet_nib.get_fdata(), pet_hu_min, pet_hu_max)

        ct_data_norm = (clipped_ct_data - np.min(clipped_ct_data)) / (np.max(clipped_ct_data) - np.min(clipped_ct_data))
        pet_data_norm = (clipped_pet_data - np.min(clipped_pet_data)) / (np.max(clipped_pet_data) - np.min(clipped_pet_data))

        final_ct_image = nib.Nifti1Image(
            ct_data_norm, pet_nib.affine, pet_nib.header
        )
        final_pet_image = nib.Nifti1Image(
            pet_data_norm, pet_nib.affine, pet_nib.header
        )

        if pet_data_norm.shape == ct_data_norm.shape:
            ct_output_path = os.path.join(out_dir, "CT", os.path.splitext(os.path.basename(ct_filename))[0] + ".nii.gz")
            pet_output_path = os.path.join(out_dir, "PET", os.path.splitext(os.path.basename(pet_filename))[0] + ".nii.gz")

            nib.save(final_ct_image, ct_output_path)
            nib.save(final_pet_image, pet_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Nifti Images")
    parser.add_argument("--in_dir", type=str, help="Input directory containing all the images (CT and PET, mixed)")
    parser.add_argument("--out_dir", type=str, help="Directory to save the images")
    args = parser.parse_args()
    
    preprocess_nifti_images(args.in_dir, args.out_dir)
