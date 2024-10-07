import cv2
import numpy as np
import os

# Parameters for the dataset
num_classes = 5  # Number of different species (classes)
images_per_class = 100  # Number of images per class
image_size = (64, 64)  # Size of the generated images
dataset_dir = 'dummy_dataset'  # Directory to store the dataset

# Species (classes) names (replace with actual species names if available)
species_names = ['elephant', 'lion', 'zebra', 'giraffe', 'rhino']

# Function to create the dataset
def generate_dummy_dataset():
    # Create the main dataset directory if it doesn't exist
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Loop through each class and generate images
    for i, species in enumerate(species_names):
        class_dir = os.path.join(dataset_dir, species)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        for j in range(images_per_class):
            # Create a blank image with random colors
            image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            image[:] = color

            # Add a random shape to represent the species
            shape_type = np.random.choice(['circle', 'rectangle', 'line'])
            if shape_type == 'circle':
                center = (np.random.randint(0, image_size[0]), np.random.randint(0, image_size[1]))
                radius = np.random.randint(10, 30)
                cv2.circle(image, center, radius, (0, 0, 0), -1)
            elif shape_type == 'rectangle':
                pt1 = (np.random.randint(0, image_size[0]//2), np.random.randint(0, image_size[1]//2))
                pt2 = (pt1[0] + np.random.randint(10, 30), pt1[1] + np.random.randint(10, 30))
                cv2.rectangle(image, pt1, pt2, (0, 0, 0), -1)
            elif shape_type == 'line':
                pt1 = (np.random.randint(0, image_size[0]), np.random.randint(0, image_size[1]))
                pt2 = (np.random.randint(0, image_size[0]), np.random.randint(0, image_size[1]))
                cv2.line(image, pt1, pt2, (0, 0, 0), thickness=2)

            # Save the image
            image_path = os.path.join(class_dir, f'{species}_{j}.png')
            cv2.imwrite(image_path, image)

    print(f"Dummy dataset generated successfully in '{dataset_dir}' directory.")

# Generate the dummy dataset
generate_dummy_dataset()