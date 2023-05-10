#!/usr/bin/env python
"""
Convert Scanoma and Smart Skin Cancer Detection models to onnx format.
"""
import tflite2onnx

# Edit these input paths based on their locations.
SCANOMA_TFLITE_PATH = "/homes/gws/degrave/projects/derm/2021.08.09/scanoma/model.tflite"
SSCD_TFLITE_PATH = "/homes/gws/degrave/projects/derm/sscd.tflite"


# Output paths should not need to be edited.
SCANOMA_ONNX_OUTPUT_PATH = "../pretrained_models/scanoma.onnx"
SSCD_ONNX_OUTPUT_PATH = "../pretrained_models/sscd.onnx"

def main():
    tflite2onnx.convert(SCANOMA_TFLITE_PATH, SCANOMA_ONNX_OUTPUT_PATH)
    tflite2onnx.convert(SSCD_TFLITE_PATH, SSCD_ONNX_OUTPUT_PATH)

if __name__ == "__main__":
    main()
