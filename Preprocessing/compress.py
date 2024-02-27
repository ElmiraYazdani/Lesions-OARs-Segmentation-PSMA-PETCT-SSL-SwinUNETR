import os
import argparse
import gzip
from tqdm import tqdm

def compress_nifti_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in tqdm(os.listdir(input_dir)):
        if not filename.endswith(".nii.gz"):
            with open(os.path.join(input_dir, filename), "rb") as input_file:
                uncompressed_data = input_file.read()
            compressed_data = gzip.compress(uncompressed_data)
            output_filename = filename + ".gz"
            with open(os.path.join(output_dir, output_filename), "wb") as output_file:
                output_file.write(compressed_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress NIfTI Files")
    parser.add_argument("--input_dir", type=str, help="Path to the directory containing NIfTI files")
    parser.add_argument("--output_dir", type=str, help="Path to the directory to save compressed NIfTI files")
    args = parser.parse_args()
    compress_nifti_files(args.input_dir, args.output_dir)
