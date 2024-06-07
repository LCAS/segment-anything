import cv2  # type: ignore

from segment_anything import sam_model_registry, SamPredictor

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image 

import pandas as pd

import gc

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def combine_image_and_mask(image, mask):

    mask_channels = mask.shape[-1] if mask.shape[-1] == 3 else mask.shape[-1] - 1

    # Convert mask to uint8 and resize it to match the dimensions of the image
    mask_resized = cv2.resize((mask[..., :mask_channels] * 255).astype(np.uint8), (image.shape[1], image.shape[0]))

    # Apply the condition to merge the images
    combined_image = np.where(mask_resized[..., None] == 0, image, 0)

    # Swap R and B channels in the combined image
    combined_image = combined_image[..., ::-1]

    combined_image = combined_image.astype(np.uint8)

    return combined_image

def show_points_on_image(image, coords, labels, path: str, fig_name, marker_size=20) -> None:
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    # Create a copy of the image to draw on
    image_with_points = image.copy()

    # Draw green points for positive labels
    for pos_point in pos_points:
        cv2.drawMarker(image_with_points, tuple(pos_point.astype(int)), (0, 255, 0), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=marker_size)

    # Draw red points for negative labels
    for neg_point in neg_points:
        cv2.drawMarker(image_with_points, tuple(neg_point.astype(int)), (0, 0, 255), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=marker_size)

    # Save the image with points
    cv2.imwrite(os.path.join(path, fig_name), cv2.cvtColor(image_with_points, cv2.COLOR_RGB2BGR))

    return

def write_masked_images_to_folder(image, masks: List[Dict[str, Any]], path: str, maskID: int) -> None:    
    for i, mask_data in enumerate(masks):
        mask = mask_data
        filename = f"{maskID}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)

        fig_name = f"Mask_{maskID}.png"

        combined_image = combine_image_and_mask(image, mask)
        
        cv2.imwrite(os.path.join(path, fig_name), combined_image)

    return

def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"

    predictor = SamPredictor(sam)

    save_images_ = False

    print(f"Model type: {args.model_type}")

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)

    csv_file_path = "output_metashape/final_dataset.csv"  # Replace with the actual path to your CSV file

    combined_image_data = pd.DataFrame()

    for t in targets:
        print(f"Processing '{t}'...")

        # Extract image name without extension from the path
        image_name = os.path.splitext(os.path.basename(t))[0]

        print(image_name)

        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)

        return_logits_ = False

        # Load CSV file into a pandas DataFrame
        csv_data = pd.read_csv(csv_file_path)

        # Filter rows based on the image name
        image_data = csv_data[csv_data['image_name'] == image_name]
        
        if image_data.empty:
            print(f"No data found for image '{image_name}' in the CSV file, skipping...")
            continue
        else:
            print(f"image_data length '{len(image_data)}'")
            # Add a new column 'mask_ID' to image_data and assign -1 initially to all elements
            image_data['mask_ID'] = -1
            # Add a new column 'score' to image_data and assign 0.0 initially to all elements
            image_data['score'] = 0.0000

        # Extract points from the CSV data
        
        input_points = image_data[['pixel_x', 'pixel_y']].values
        #print(input_points)
        input_labels = np.ones((1,input_points.shape[0]))[0]
        #print(input_labels)

        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)
        if output_mode == "binary_mask":
            os.makedirs(save_base, exist_ok=True)
            fig_name = "Input_AllPoints.png"
            if save_images_:
                show_points_on_image(image, input_points, input_labels, save_base, fig_name)

        #print(f"Input points x-y coordinate list {input_points}")
        
        input_label = np.array([1])

        mask_id = 0

        for i in range(len(image_data)):
            current_row = image_data.iloc[i]

            if current_row['mask_ID'] == -1:
                
                # Extract the row of points from the CSV data as a 2D array
                input_point = current_row[['pixel_x', 'pixel_y']].values.reshape(1, -1)

                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False,
                    return_logits=return_logits_,
                )

                # Check the score
                score_threshold = 0.95  # Adjust this threshold as needed
                if scores[0] >= score_threshold:
                    # Use the obtained mask
                    print(f"Mask obtained with a score of {scores[0]}, processing further for...")

                    print(f"Input point x-y coordinate {input_point}")
                    #print(f"Input point mask ID: {image_data['mask_ID'].values[i]}")

                    base = os.path.basename(t)
                    base = os.path.splitext(base)[0]
                    save_base = os.path.join(args.output, base)
                    if output_mode == "binary_mask":
                        os.makedirs(save_base, exist_ok=True)
                        fig_name = "Input_" + str(mask_id) + ".png"
                        if save_images_:
                            show_points_on_image(image, input_point, input_label, save_base, fig_name, 100)

                    # Find all pixels within the mask region
                    mask_region = masks[0]  # Assuming masks is a binary mask

                    #print("Mask Region")

                    # Get the coordinates of all pixels within the mask
                    mask_coords = np.argwhere(mask_region)

                    # Combine values in image_data
                    combined_image_coords = image_data['pixel_x'].round().astype(int) * 10000 + image_data['pixel_y'].round().astype(int)

                    # Combine values in mask_coords
                    combined_mask_coords = mask_coords[:, 1] * 10000 + mask_coords[:, 0]

                    # Find matching rows using combined values
                    matching_rows = image_data[np.isin(combined_image_coords, combined_mask_coords)]

                    #print("Matching Coordinates")
                    #print(len(matching_rows))
                    #print(matching_rows.index)
                    
                    image_data.loc[matching_rows.index, 'mask_ID'] = mask_id
                    image_data.loc[matching_rows.index, 'score'] = scores[0]

                    #print("Mask Assigned")

                    base = os.path.basename(t)
                    base = os.path.splitext(base)[0]
                    save_base = os.path.join(args.output, base)
                    if output_mode == "binary_mask":
                        os.makedirs(save_base, exist_ok=True)
                        if save_images_:
                            write_masked_images_to_folder(image, masks, save_base, mask_id)
                    else:
                        save_file = save_base + ".json"
                        with open(save_file, "w") as f:
                            json.dump(masks, f)
                    
                    print("Mask image saved!")
                    mask_id = mask_id + 1

        # Save the updated image_data to a new CSV file
        output_csv_path = "mask_output.csv"  # Replace with your desired output file path
        image_data.to_csv(os.path.join(save_base, output_csv_path), index=False)
        print(f"Updated image_data saved as '{output_csv_path}', processing next image")  

        # Append image_data to the combined DataFrame
        combined_image_data = pd.concat([combined_image_data, image_data], ignore_index=True)
             
    plt.close('all')

    # Save the combined DataFrame to a new CSV file
    combined_csv_file_path = "combined_mask_output.csv"
    combined_image_data.to_csv(os.path.join(args.output,combined_csv_file_path), index=False)

    print(f"Combined mask output data saved to {combined_csv_file_path}")
    
    print("All images are processed, done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
