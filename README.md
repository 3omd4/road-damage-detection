Road Damage Detection with YOLOv8
Overview
This project implements a road damage detection model using the YOLOv8 architecture to identify and classify road damages such as potholes, cracks, and manholes. The model is trained on the Road Damage Dataset: Potholes, Cracks, and Manholes, which contains annotated images of road surfaces captured in various conditions.
Dataset
The Road Damage Dataset: Potholes, Cracks, and Manholes from Kaggle is used for training and evaluation. Below are the detailed characteristics of the dataset:

Source: Road Damage Dataset: Potholes, Cracks, and Manholes by Lorenzo Arcioni.
Content:
Images: 2,009 images with annotated road damages captured in various environments.
Annotations: Provided in YOLO format (text files), including:
Filename of the image.
Bounding box coordinates (x_center, y_center, width, height, normalized) for road damages.
Class labels: pothole, crack, manhole.


Classes: The project focuses on three classes:
Pothole: Depressions in the road surface.
Crack: Surface cracks of varying types.
Manhole: Manhole covers on the road.


Image Format: JPEG files.
Resolution: Varies (typically 1280x720 or higher, resized to 640x640 for training).
Conditions: Images capture diverse conditions, including different lighting (day, dusk), weather (dry, wet), and road types (asphalt, concrete).


Size: The dataset occupies approximately 1.2 GB when extracted.
Directory Structure:

dataset/
├── images/
│   ├── train/
│   ├── val/
├── labels/
│   ├── train/
│   ├── val/


Usage: The dataset is split into training and validation sets. Images are resized to 640x640 pixels to match YOLOv8 input requirements.

Download the dataset from Kaggle and extract it into the dataset/ directory.
Install dependencies using:
pip install ultralytics

Project Structure
road-damage-detection/
├── dataset/                  # Training and validation dataset
├── models/                   # Trained YOLO models and weights (last.pt, best.pt)
├── notebooks/                # Jupyter notebooks for training and evaluation
├── visuals/                  # Plots and detection examples
├── README.md                 # Project documentation

Setup

Clone the Repository:

git clone https://github.com/3omd4/road-damage-detection.git
cd road-damage-detection


Download the Dataset:

Download the Road Damage Dataset from Kaggle.
Extract the dataset into the dataset/ directory.


Install Dependencies:


pip install -r requirements.txt

Usage

Preprocess the Data:The dataset is already structured in YOLO format. Ensure the dataset/ directory contains images/ and labels/ subdirectories for training and validation sets, along with a dataset.yaml file specifying paths and classes.

Train the Model:Train the YOLOv8 model using:


yolo train model=yolov8n.pt data=dataset.yaml epochs=30 imgsz=640 batch=16

   The script trains the YOLOv8n model, fine-tuning from yolov8n.pt, and saves the trained models (last.pt and best.pt) to the models/ directory.

Evaluate the Model:Evaluate the model's performance on the validation set:

yolo val model=models/best.pt data=dataset.yaml

   This generates metrics such as mAP (mean Average Precision), precision, recall, and a confusion matrix.
Model Details

Architecture: YOLOv8 (pre-trained on COCO using yolov8n.pt, fine-tuned for road damage detection).
Input: Images resized to 640x640 pixels.
Output: Bounding box predictions with class labels: pothole, crack, manhole.
Training:
Optimizer: SGD (default in YOLOv8)
Loss: YOLOv8 loss (combining box, objectness, and classification losses)
Batch Size: 16
Epochs: 30



Model Outputs

Trained Models:
last.pt: The final model checkpoint after 30 epochs.
best.pt: The best-performing model checkpoint based on validation mAP.



Training Curves
The following plots show the model's performance during training (available in the visuals/ directory):

See the visuals directory for all plots and detection examples.
Contributing
Contributions are welcome! Please:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Submit a pull request with a clear description of changes.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Road Damage Dataset: Potholes, Cracks, and Manholes by Lorenzo Arcioni.
YOLOv8 by Ultralytics for providing the model implementation.
