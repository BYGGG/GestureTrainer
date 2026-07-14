import os
import sys
import argparse
import importlib

import tensorflow as tf
import tf2onnx


def parse_args():
    p = argparse.ArgumentParser(description="Convert a MATLAB-exported TF model to ONNX.")
    p.add_argument("--model-name", required=True,
                   help="Name of the exported TF package/folder (matches trained_name).")
    p.add_argument("--tf-dir", required=True,
                   help="Folder containing the exported TF package.")
    p.add_argument("--onnx-dir", required=True,
                   help="Output folder for the .onnx file.")
    p.add_argument("--window-size", type=int, default=60)
    p.add_argument("--feature-size", type=int, default=350)
    p.add_argument("--opset", type=int, default=13)
    return p.parse_args()


def main():
    args = parse_args()

    # Make the exported package importable, then load it by name
    sys.path.insert(0, os.path.abspath(args.tf_dir))
    model_module = importlib.import_module(args.model_name)
    new_model = model_module.load_model()

    # Input signature (works for transformer and CNN models)
    spec = (tf.TensorSpec((None, args.window_size, args.feature_size),
                          tf.float32, name="input"),)

    os.makedirs(args.onnx_dir, exist_ok=True)
    output_path = os.path.join(args.onnx_dir, args.model_name + ".onnx")

    tf2onnx.convert.from_keras(new_model,
                               input_signature=spec,
                               opset=args.opset,
                               output_path=output_path)

    print("ONNX model saved to:", output_path)


if __name__ == "__main__":
    main()