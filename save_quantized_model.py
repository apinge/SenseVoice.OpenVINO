import openvino as ov

def main():
    """
    Main function to load an ONNX quantized model and save it as an OpenVINO IR model.

    Here we save the ONNX quantized model directly as an OpenVINO IR model. 
    If we use ovc directly, it will report an error, so we read the model first.
    """
    model_path = "model_quant.onnx"
    core = ov.Core()
    model = core.read_model(model_path)
    ov.save_model(model, "model_quant.xml")

if __name__ == "__main__":
    main()
