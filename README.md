# GestureTrainer

# Table of Contents
- [Overview](#overview)
- [How this fits into the system](#how-this-fits-into-the-system)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Known Issues](#known-issues)

## Overview
GestureTrainer is a MATLAB desktop app for training hand-gesture classifiers.
It takes the gesture CSVs captured through the GestureLogger web platform,
preprocesses them into a dataset, trains a model, and exports it as `.onnx`
for live inference in the headset.

## How this fits into the system
The system has three parts:
- [GestureLogger](https://github.com/Saatvik-Lochan/GestureLogger) (desktop) —
  authoring and capture control. Define gestures, record demonstrations,
  build trials, register participants, and get each participant's capture
  link.
- [GestureLogger web platform](https://github.com/BYGGG/GestureLogger) — the
  web server. Hosts the WebXR capture pages (`frontend/`), the API
  (`backend/`), the manager portal (`portal/`), and the ONNX inference page
  (`deploy/`). It shares a name with the desktop app but is a different repo.
- GestureTrainer (this repo) — trains the classifier.

The data loop: author trials in GestureLogger → participants perform them in
the headset → captured CSVs land on the server → train a model here → upload
the `.onnx` through the portal → the `deploy/` page runs it live.

## Prerequisites
- MATLAB R2024a or later, with App Designer
- Deep Learning Toolbox
- Python with `tensorflow` and `tf2onnx` (only needed if you run the ONNX
  conversion manually, see [Usage](#usage))

## Usage

1. **Clone the Repository**
   Clone this repository and install the prerequisites listed above.

2. **Set Up the Project**
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
- Point it at a folder of captured data from the web platform. It expects a
  `gestures/` folder of CSVs named `{GestureClass}-id-{N}-participant-{N}.csv`
  and an `annotations/` folder.
- You can configure various design decisions, such as joint selection, degrees of freedom feature selection, data augmentation (currently under development), window size, and train/validation/test split ratios.
- Click the **Process** button (top right) to start preprocessing. Progress and errors will be shown in the MATLAB command window.
- **Important:** Remember to save the mean and standard deviation values printed in the command window for deployment use.

![image](https://github.com/user-attachments/assets/85f62a5f-2f7f-417d-bb34-bf7ff35d56a9)

5. **Model Specification**
- To specify a new model configuration, select **Specify new model** in the Overview page.
- Configure the model layers and relevant hyperparameters.
- Once you've decided on the configuration, click **Confirm** (top right).

![image](https://github.com/user-attachments/assets/a1b5ea92-2efa-449d-9edc-f06857daceea)

6. **Train**
- Ensure you have selected both a preprocessed dataset and a specified model.
- Click the **Train** button (top) to start training.
- Training logs and performance plots will appear to help evaluate progress.
- If you're unsatisfied with the model, you can delete it by clicking the **Abandon** button.

![image](https://github.com/user-attachments/assets/fcb2ef57-5f74-43ee-a39b-929552a02eb8)

7. **Deploy**
- With a trained model, go to the **Deployment** page.
![image](https://github.com/user-attachments/assets/16dc454a-335c-4296-9810-d6d49dca1d6f)

- Click **Convert** to export the trained model as `.onnx`.
- If you need to run the conversion manually (from a TensorFlow export of
  the model), use `model_conversion.py`:

- Upload the `.onnx` model through the
  [GestureLogger web platform](https://github.com/BYGGG/GestureLogger)
  portal. The platform's `deploy/` page loads it and runs inference live in
  the headset.

<img width="800" height="600" alt="Scissors" src="https://github.com/user-attachments/assets/85d3de7d-8dd2-4808-abcc-57343be9d067" />

## Known Issues
1. **Local Deployment**
- WebXR requires HTTPS, so running the web platform locally may need manual
  certificate setup. See the
  [web platform repo](https://github.com/BYGGG/GestureLogger).

2. **Export to ONNX**  
- The conversion from the attention layer to ONNX must be done manually using `model_conversion.py`.
