# FPGA Implementation of Monocular Depth Estimation Model: MiDaSNet

This repository contains code to FPGA implementation of depth estimation model, MiDaSNet.

The figure below shows the architecture of fully convolutional MiDaSNet_small model.
<p align="center">
  <img src="figures/midas.png" alt="Example Image" width="60%" />
</p>


## Usage

1) To quantize the float model you may use the docker image which is generated from a recipe provided by Xilinx:

    ```shell
    docker pull yasinadiyaman/xilinx-vitis-ai-pytorch-gpu
    ```

2) Run the docker image. Then, inside docker, run the bash script:

    ```shell
    bash quantize.sh
    ```

3) After xmodel file generated, copy xmodel file with meta.json into Kria KV260 board (tested on Ubuntu image, not on Petalinux).

4) Check the required files for run. I could not upload big files into the repo. You should download them from provided links.

5) Evaluate the model on NYUv2. You may use another dataset, just copy images and ground truth depth maps under data/ directory.
   
    ```shell
    python evaluate_xmodel.py
    ```
    
6) To compare inferenced depth maps from CPU and DPU on sample results:  

   ```shell
   python visualize.py
   ```


## Quantitative Comparison 
### Comparison with Baseline on NYUv2 Dataset

| **Architecture**            | **Input Size** | **GOPs** | **δ1** | **δ2** | **δ3** | **REL** | **RMSE** | **fps** | **Power** | **Freq** | **Platform**       |
|-----------------------------|----------------|----------|--------|--------|--------|----------|----------|---------|-----------|----------|--------------------|
| VGG (Eigen et al.)          | 228x304        | 23.4     | 76.9   | -      | -      | -        | -        | -       | -         | -        | CPU                |
| ResNet50 (Xian et al.)      | 384x384        | 61.8     | 78.1   | 95     | 98.7   | 0.155    | 0.66     | -       | -         | -        | CPU                |
| ResNet50 (Swaraja et al.)   | 256x256        | -        | -      | -      | -      | 0.168    | 0.638    | -       | -         | -        | CPU                |
| EfficientNet-B0 (Swaraja et al.) | 256x256    | -        | -      | -      | -      | 0.156    | 0.625    | -       | -         | -        | CPU                |
| ResNet-UpProj (Laina et al.) | 228x304       | 22.9     | 81.1   | 95.3   | 98.8   | 0.127    | 0.573    | 18.18  | -         | -        | GPU                |
| FasterMDE (ZiWen et al.)      | -              | -        | -      | -      | -      | **0.113** | -       | 33.57  | -         | -        | Jetson Xavier NX   |
| DeepVideoMVS (Hashimoto et al.) | -             | -        | -      | -      | -      | -        | -        | 3.6    | -         | 188M    | ZCU104             |
| DepthFCN (Sada et al.)     | 256x256        | **0.66** | 76.2   | -      | -      | -        | -        | **123** | **0.3W** | 200M    | ZU3EG              |
| MiDaSNet (Ranftl et al.)     | 480x640        | 3.47     | **85.8*** | **97.73*** | **99.51*** | 0.117*   | **0.467* | 0.71   | 1.4W    | 1.3G   | ARM Cortex A53    |
| Proposed work               | 480x640        | 7.43     | **82.6*** | **96.81*** | **99.31*** | 0.133*   | **0.506* | 50.74  | 0.62W   | 300M   | Kria-KV260         |

*Zero-shot performance*

## References
- Eigen et al.
- Xian et al.
- Swaraja et al.
- Laina et al.
- Dou et al.
- Nobuho et al.
- Youki et al.
- Midas et al.




### Depth map comparison

Zoom in for better visibility
![](figures/Comparison.png)

### Speed on Camera Feed	

Test configuration	
- Windows 10	
- 11th Gen Intel Core i7-1185G7 3.00GHz	
- 16GB RAM	
- Camera resolution 640x480	
- openvino_midas_v21_small_256	

Speed: 22 FPS


### References

Please cite our paper if you use this code or any of the models:
```
@ARTICLE {Ranftl2022,
    author  = "Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun",
    title   = "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer",
    journal = "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    year    = "2022",
    volume  = "44",
    number  = "3"
}
```


### License 

MIT License 
