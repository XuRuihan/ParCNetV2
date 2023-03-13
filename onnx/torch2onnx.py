import sys

sys.path.extend([".", ".."])

import torch
from models.parcnetv2 import parcnetv2_tiny, parcnetv2_small, parcnetv2_base

models = {
    "paracnetv2_tiny": parcnetv2_tiny(),
    "parcnetv2_small": parcnetv2_small(),
    "parcnetv2_base": parcnetv2_base(),
}

for name, model in models.items():
    model.eval()

    input_names = ["input"]
    output_names = ["output"]

    x = torch.randn((1, 3, 224, 224))

    torch.onnx.export(
        model,
        (x,),
        f"onnx/{name}.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"input": [0], "output": [0]},
    )
