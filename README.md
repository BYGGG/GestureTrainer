# Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Known Issues](#known-issues)

## Overview
This software enables users to create, train, and deploy custom gesture recognition models. It streamlines the entire process—from dataset preprocessing to model deployment—making it accessible for users to explore and implement gesture recognition solutions.

## Features
- **User-Friendly Interface:** Built with MATLAB App Designer for an interactive user experience.  
- **Dataset Preprocessing:** Tools to clean, normalize, and prepare gesture data.  
- **Model Specification:** Choose from various machine learning algorithms tailored for gesture recognition.  
- **Training Pipeline:** Efficient training with real-time performance metrics.  
- **Deployment Options:** Export models for online inference on the headset.

## Prerequisites
- MATLAB R2024a or later  
- MATLAB App Designer  
- Relevant toolboxes (e.g., Deep Learning Toolbox)  
- TensorFlow  

## Usage

1. **Clone the Repository**  
   Clone this repository and install the prerequisites listed above.

3. **Set Up the Project**  
- Open `Project.mlapp` in MATLAB App Designer.  
- Create a new project or load an existing one by entering the project name and selecting the folder.  

![image](https://github.com/user-attachments/assets/fb5a6d54-dff6-4fa1-a231-6f59f35fc8a2)

3. **Overview**  
- Once you click **Create/Load**, an Overview page appears.  
- You can preprocess a new dataset or specify model configurations.  
- If there are existing preprocessed datasets or configured models, you can select one and click **View** to see its details.  

![image](https://github.com/user-attachments/assets/51ec5f38-0b04-4722-ac83-c3d79ebd38d2)

4. **Data Preprocessing**  
- To preprocess a new dataset, select **Preprocess new dataset** in the Overview page.  
- You can configure various design decisions, such as joint selection, degrees of freedom feature selection, data augmentation (currently under development), window size, and train/validation/test split ratios.  
- Click the **Process** button (top right) to start preprocessing. Progress and errors will be shown in the MATLAB command window.  
- **Important:** Remember to save the mean and standard deviation values printed in the command window for deployment use.  

![image](https://github.com/user-attachments/assets/85f62a5f-2f7f-417d-bb34-bf7ff35d56a9)

5. **Model Specification**  
- To specify a new model configuration, select **Specify new model** in the Overview page.  
- Configure the model layers and relevant hyperparameters.  
- Once you’ve decided on the configuration, click **Confirm** (top right).  

![image](https://github.com/user-attachments/assets/a1b5ea92-2efa-449d-9edc-f06857daceea)

6. **Train**  
- Ensure you have selected both a preprocessed dataset and a specified model.  
- Click the **Train** button (top) to start training.  
- Training logs and performance plots will appear to help evaluate progress.  
- If you’re unsatisfied with the model, you can delete it by clicking the **Abandon** button.  

![image](https://github.com/user-attachments/assets/fcb2ef57-5f74-43ee-a39b-929552a02eb8)

7. **Deploy**  
- With a trained model, go to the **Deployment** page.
![image](https://github.com/user-attachments/assets/16dc454a-335c-4296-9810-d6d49dca1d6f)

- **Note**: The export to ONNX functionality is currently incomplete, so you’ll need some manual steps to export the model.  
- After clicking **Convert**, a TensorFlow model is saved in the `tf` folder of your project.  
- Manually run the `model_conversion.py` script to convert the TensorFlow model into ONNX format.  
- Once the ONNX model is in the `onnx` folder, clone the online inference repository for the gesture recognizer (see [WebXR-GestureDeploy](https://github.com/BYGGG/WebXR-GestureDeploy) and follow its steps to set up on-device inference.  

![image](https://github.com/user-attachments/assets/75017498-6f18-4f86-a547-8160c966fac2)

## Known Issues
1. **Export to ONNX**  
- The conversion from the exported TensorFlow model to ONNX must be done manually using `model_conversion.py`.  

2. **Local Deployment**  
- Some HTTPS-related configurations may require manual adjustments when deploying locally.
