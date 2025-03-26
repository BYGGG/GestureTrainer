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

![image](https://github.com/user-attachments/assets/5390bd1f-b948-4826-906e-804823d86d36)

3. **Overview**  
- Once you click **Create/Load**, an Overview page appears.  
- You can preprocess a new dataset or specify model configurations.  
- If there are existing preprocessed datasets or configured models, you can select one and click **View** to see its details.  

![image](https://github.com/user-attachments/assets/ae6a5b44-780a-4742-be57-be16a1b34a3d)

4. **Data Preprocessing**  
- To preprocess a new dataset, select **Preprocess new dataset** in the Overview page.  
- You can configure various design decisions, such as joint selection, degrees of freedom feature selection, data augmentation (currently under development), window size, and train/validation/test split ratios.  
- Click the **Process** button (top right) to start preprocessing. Progress and errors will be shown in the MATLAB command window.  
- **Important:** Remember to save the mean and standard deviation values printed in the command window for deployment use.  

![image](https://github.com/user-attachments/assets/40bf7536-2a88-44e7-b597-a09d9546c983)

5. **Model Specification**  
- To specify a new model configuration, select **Specify new model** in the Overview page.  
- Configure the model layers and relevant hyperparameters.  
- Once you’ve decided on the configuration, click **Confirm** (top right).  

![image](https://github.com/user-attachments/assets/b10a719f-7a8f-49da-89c1-37a231981fbe)

6. **Train**  
- Ensure you have selected both a preprocessed dataset and a specified model.  
- Click the **Train** button (top) to start training.  
- Training logs and performance plots will appear to help evaluate progress.  
- If you’re unsatisfied with the model, you can delete it by clicking the **Abandon** button.  

![image](https://github.com/user-attachments/assets/8d34cb55-72b7-49eb-b023-90061b49c1df)

7. **Deploy**  
- With a trained model, go to the **Deployment** page.
![image](https://github.com/user-attachments/assets/16dc454a-335c-4296-9810-d6d49dca1d6f)

- **Note**: The export to ONNX functionality is currently incomplete, so you’ll need some manual steps to export the model.  
- After clicking **Convert**, a TensorFlow model is saved in the `tf` folder of your project.  
- Manually run the `model_conversion.py` script to convert the TensorFlow model into ONNX format.  
- Once the ONNX model is in the `onnx` folder, clone the online inference repository for the gesture recognizer (see [LINK HERE]) and follow its steps to set up on-device inference.  

![image](https://github.com/user-attachments/assets/568fdf85-61c1-4b58-ab2f-81d2014eed6a)

## Known Issues
1. **Export to ONNX**  
- The conversion from the exported TensorFlow model to ONNX must be done manually using `model_conversion.py`.  

2. **Local Deployment**  
- Some HTTPS-related configurations may require manual adjustments when deploying locally.
