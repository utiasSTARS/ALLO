"""Organize files, resize images, combine masks"""

import sys
import os
import os.path as osp
from pathlib import Path
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import argparse
import random
from tqdm import tqdm
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def combine_masks(anomaly_mask, fb_mask) -> np.ndarray:
    """Combine anomaly mask and fb mask into 3 class mask"""
    final_mask = np.zeros_like(anomaly_mask, dtype=np.uint8)  # space background is black
    final_mask[fb_mask == 255] = 1  # grayscasle colour of foreground (station and celestial bodies)
    final_mask[anomaly_mask == 255] = 2  # grayscale colour of anomaly
    return final_mask

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to input directory with camera folders')
    parser.add_argument('--output', type=str, help='path to output directory')
    parser.add_argument('--log_file', type=str, help='path to log file')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument("--anomaly", action="store_true", 
                        default=False, help="normal or anomaly dataset")
    parser.add_argument('--min_pixel', type=int, default=2000, help='minimum pixel size of anomaly')
    args = parser.parse_args()
    return args


def main(args):
    print(args.anomaly)
    # find input and output directories, make output directory if needed and log file
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file)
    logging.basicConfig(filename=str(log_file), level=logging.INFO)

    # iterate through all camera folders in input directory
    for cam in tqdm(input_dir.iterdir(), 
                    desc="Processing cameras", 
                    total=len(list(input_dir.iterdir()))):
        if cam.is_dir() and cam.name.startswith("Camera"):
            logging.info(f"Processing {cam.name}")
            cnt = 0
            # images are saved to numbered camera folder with two subfolders: images and masks
            cam_output_dir = output_dir / cam.name
            cam_output_dir.mkdir(parents=True, exist_ok=True)
            if args.anomaly:  # combine anomalous masks and save images and masks
                # anomaly images and anomaly mask input and output folders
                anomaly_input_dir = cam / "anomaly"
                anomaly_output_dir = cam_output_dir / "images"
                anomaly_output_dir.mkdir(parents=True, exist_ok=True)
                mask_input_dir = cam / "anomaly_mask"
                mask_output_dir = cam_output_dir / "masks"
                mask_output_dir.mkdir(parents=True, exist_ok=True)
                
                # iterate through saved anomaly masks
                for mask in mask_input_dir.iterdir():
                    done = False
                    if mask.suffix == ".png":
                        #* Read anomaly mask and verify shape
                        try:
                            mask_img = cv2.imread(str(mask))
                            if mask_img.shape != (1080, 1920, 3):
                                logging.error(f"Invalid shape {mask_img.shape} for {mask}")
                                continue
                        except Exception as e:
                            logging.error(f"Failed to read {mask}")
                            continue
                        #* read anomaly image and foreground/backgrond mask, combine masks
                        try:
                            anomaly_img = cv2.imread(str(anomaly_input_dir / mask.name))  # anomalous image
                            if anomaly_img.shape != (1080, 1920, 3):
                                logging.error(f"Invalid shape {anomaly_img.shape} for {mask.name}")
                                continue
                            else:
                                # Check number of anomalous pixels, if more than min pixels save
                                if (np.sum(mask_img == 255) // 3) <= args.min_pixel: 
                                    logging.info(f"Skipping {mask.name}")
                                    continue
                                # read foreground/background mask
                                try:
                                    fb_mask = cv2.imread(str(mask).replace("anomaly_mask", "fb_mask"))
                                except Exception as e:
                                    logging.error(f"Failed to read f/b mask: {mask}")
                                    continue
                                # combine anomaly mask with foreground/background mask
                                mask_img = combine_masks(mask_img, fb_mask)
                                # save anomalous image and combined masks
                                done = cv2.imwrite(str(mask_output_dir / mask.name), mask_img)
                                done = done and cv2.imwrite(str(anomaly_output_dir / mask.name), anomaly_img)
                                if not done:
                                    logging.error(f"Failed to write {mask.name}")
                                else:
                                    cnt += 1
                        except Exception as e:
                            logging.error(f"Failed to read {mask.name} \
                                or failed to write to {str(mask.name)}")
                            print(e)
            else:  # save normal images with their foreground/background masks
                # input and output directories for normal images and masks
                cam_input_dir = cam / "normal"
                fb_input_dir = cam / "fb_mask"
                normal_output_dir = cam_output_dir / "images"
                normal_output_dir.mkdir(parents=True, exist_ok=True)
                mask_output_dir = cam_output_dir / "masks"
                mask_output_dir.mkdir(parents=True, exist_ok=True)
                # iterate through all normal images in camera folder
                for f in cam_input_dir.iterdir():
                    done = False
                    if f.suffix == ".png":
                        #* read, resize and save normal image to images folder
                        try:
                            img = cv2.imread(str(f))
                            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
                            done = cv2.imwrite(str(normal_output_dir / f.name), img)
                        except Exception as e:
                            logging.error(f"Failed to process {f}")
                        # read, resize and save foreground/background mask to masks folder
                        try:
                            fb_mask = cv2.imread(str(fb_input_dir / f.name))
                            fb_mask = cv2.resize(fb_mask, (1920, 1080), interpolation=cv2.INTER_AREA)
                            done = done and cv2.imwrite(str(mask_output_dir / f.name), fb_mask)
                        except Exception as e:
                            logging.error(f"Failed to process {f}")
                        finally:
                            if not done:
                                logging.error(f"Failed to write {f} to \
                                    {str(normal_output_dir / f.name)}")
                            else:
                                cnt += 1
                    else:
                        logging.info(f"Skipping {f}")
                        
            logging.info(f"Processed {cnt} images")

if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    main(args)