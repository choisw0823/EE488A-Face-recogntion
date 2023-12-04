from PIL import Image
import os
import glob
import shutil
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]='4'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

mtcnn = MTCNN(keep_all=True,device = device)
crop_path = "/home/ug4/vggface_crop"  # Path to save the cropped images
data_dir = "/ee488a/vggface/train"  # Base directory for data



# Function to create the same folder structure in the destination directory
def create_folder_structure(src_dir, dest_dir):
    for subdir, dirs, files in os.walk(src_dir):
        for dir in dirs:
            os.makedirs(os.path.join(dest_dir, dir), exist_ok=True)

# Create the same folder structure from data_dir to crop_path
create_folder_structure(data_dir, crop_path)

# Function to crop faces and save the images
def crop_and_save_faces(data_dir, crop_path):
    for subdir in tqdm(os.listdir(data_dir)):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            # Process all jpg images in each sub-folder
            files = glob.glob(subdir_path + '/*.jpg')

            for fname in files:
                image = Image.open(fname).convert('RGB')
                bboxes, probs = mtcnn.detect(image)

                if bboxes is not None:
                    # Choose the most confident face detection
                    max_conf_index = np.argmax(probs)
                    bbox = bboxes[max_conf_index]
                    conf = probs[max_conf_index]

                    # Calculate center and size of the bounding box
                    sx, sy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                    ss = int(max(bbox[3] - bbox[1], bbox[2] - bbox[0]) / 1.5)
                    
                    # Crop the face area using the bounding box
                    face = image.crop((sx - ss, sy - ss, sx + ss, sy + ss))

                    # Check if the cropped face meets the size requirements
                    if face.size[0] == face.size[1] and face.size[1] >= 50 and conf >= 0.9:
                        outname = fname.replace(data_dir, crop_path)
                        os.makedirs(os.path.dirname(outname), exist_ok=True)
                        face = face.resize((256, 256))  # Resize the face
                        face.save(outname)  # Save the cropped face image
                    else:
                        print(f'[INFO] Invalid image {fname}')
                else:
                    print(f'[INFO] Skipping image {fname} with no or multiple faces detected')

# Crop faces and maintain the folder structure
crop_and_save_faces(data_dir, crop_path)

# Print completion message
print('Cropping completed.')
