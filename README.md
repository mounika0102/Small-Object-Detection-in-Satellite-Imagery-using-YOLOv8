# Small-Object-Detection-in-Satellite-Imagery-using-YOLOv8

📄 Project Overview

This project focuses on detecting small objects in high-resolution satellite images using the YOLOv8 model. The **DIOR dataset** is used, which includes diverse object categories and varying scales, making it ideal for evaluating object detection performance, especially for small objects.



📁 Dataset: DIOR:

DIOR (Dataset for Object Detection in Optical Remote Sensing Images) contains 20 object categories across thousands of high-resolution satellite images. This project uses both training and test data provided in the dataset.


⚙️ Project Workflow

1. Library Installation
Install required libraries including `ultralytics` for YOLOv8 and `gdown` for downloading from Google Drive.
```bash
pip install -U -q gdown ultralytics
```

2. Download DIOR Dataset
Check for existing data and download the annotations and image data using `gdown`.

3. Extract Data
Extract zip files into a structured directory for annotations and images.

4. Prepare File Lists
Create lists of all image and annotation files from the dataset for further processing.

5. Exploratory Data Analysis (EDA)
Parse the XML annotation files to collect metadata and analyze object distribution in the dataset.

6. Annotation Parsing
Parse XML files to extract image size, object labels, and bounding box information.

7. Convert Annotations to YOLO Format
Normalize bounding boxes and convert annotations into YOLO format with the structure:
```
<class_id> <x_center> <y_center> <width> <height>
```

8. Save YOLO Format Annotations
Save YOLO-formatted annotations into a new directory for training.

9. Train YOLOv8 Model
Create a YAML config file defining dataset paths and class names. Train the model using:
```python
model = YOLO('yolov8n.yaml')
Results = model.train(data='config.yaml', imgsz=800, epochs=50, batch=16, name='yolov8n_epochs50_batch16')
```

 10. Evaluate on Test Set
Validate model performance using the test set with:
```python
model.val(data='test_config.yaml', imgsz=800, name='yolov8n_val_on_test')
```

11. Make Predictions
Run predictions on a random test image and visualize original vs predicted bounding boxes using `matplotlib`.



Files
- `config.yaml`: Defines training/validation paths and class labels.
- `dior_data/`: Contains extracted images and annotation files.
- `runs/detect/`: YOLOv8 training results and best model weights.


Key Features
- Accurate small object detection in satellite imagery.
- Uses DIOR dataset with 20 object classes.
- Full pipeline: download, preprocess, train, evaluate, predict.


Model Used
YOLOv8 Model: A lightweight yet powerful object detection model optimized for speed and efficiency.


Future Improvements
- Hyperparameter tuning for improved small object accuracy.
- Experimenting with YOLOv8 variants (e.g., YOLOv8s, YOLOv8m).
- Incorporating additional augmentation techniques.
