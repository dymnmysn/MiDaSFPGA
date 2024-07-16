#!/bin/bash

conda activate vitis-ai-pytorch
python quantize.py
vai_c_xir -x quantization_results/build/quantized/MidasNet_small_int.xmodel -a dpu_arch/kv260_arch4096.json -o quantization_results/xmodel_files -n MidasNet_small_int