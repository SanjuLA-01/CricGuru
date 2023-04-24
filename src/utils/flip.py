import os
import cv2

def flipImages(input_path, output_path):

    # Loop through all the files in the input directory
    for filename in os.listdir(input_path):
        # Check if the file is an image
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # Load the image
            image = cv2.imread(os.path.join(input_path, filename))
            # Flip the image horizontally
            flipped = cv2.flip(image, 1)
            # Save the flipped image
            cv2.imwrite(os.path.join(output_path, filename), flipped)
