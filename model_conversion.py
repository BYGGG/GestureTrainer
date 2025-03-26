import os
import tensorflow as tf
import tf2onnx
from t_seq_120 import load_model

# If it's attention layer, then remember to remove the output_shape in model.py
# If it's LSTM layer, then remember to change unroll to True in model.py

# Recreate the exact same model, including its weights and the optimizer
new_model = load_model()

# Show the model architecture (optional)
# new_model.summary()

# Set the window_size and feature_size according to your model's input shape
window_size = 120
feature_size = 350

# Define the input signature
spec = (tf.TensorSpec((None, window_size, feature_size), tf.float32, name="input"),)  # Works for transformer and CNN models

# Get the directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the script's directory
parent_dir = os.path.dirname(current_dir)

# Define the 'onnx' folder path in the parent directory
onnx_folder = os.path.join(parent_dir, 'onnx')

# Create the 'onnx' folder in the parent directory if it does not exist
if not os.path.exists(onnx_folder):
    os.makedirs(onnx_folder)

onnx_file = 't_seq_120.onnx'

# Define the output path for the ONNX model
output_path = os.path.join(onnx_folder, onnx_file)
onnx_model, _ = tf2onnx.convert.from_keras(new_model, input_signature=spec, opset=13, output_path=output_path)
