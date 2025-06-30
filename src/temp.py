import torch
import torch.nn as nn
import torch.ao.quantization as quant

torch.backends.quantized.engine = 'fbgemm'

model_fp32 = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
    nn.ReLU()
)
model_fp32.eval()
model_fp32.qconfig = quant.get_default_qconfig('fbgemm')

# Prepare and convert
model_prepared = quant.prepare(model_fp32)
with torch.no_grad():
    model_prepared(torch.randn(1, 1, 16, 16))
model_int8 = quant.convert(model_prepared)

# Try running the quantized model
try:
    out = model_int8(torch.randn(1, 1, 16, 16))
    print("Qiuantized model ran successfully.")
except NotImplementedError as e:
    print(" Quantized kernel is missing.")
    print(e)

