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

*Zero-shot performance 

## References

1. Eigen, D., Fergus, R.: Predicting depth, surface normals, and semantic labels with a common multi-scale convolutional architecture. In: 2015 IEEE International Conference on Computer Vision (ICCV), Santiago, Chile, pp. 2650–2658 (2015). [DOI: 10.1109/ICCV.2015.304](https://doi.org/10.1109/ICCV.2015.304)

2. Xian, K., et al.: Monocular relative depth perception with web stereo data supervision. In: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, USA, pp. 311–320 (2018). [DOI: 10.1109/CVPR.2018.00040](https://doi.org/10.1109/CVPR.2018.00040)

3. Swaraja, K., Naga Siva Pavan, K., Suryakanth Reddy, S., Ajay, K., Uday Kiran Reddy, P., Kora, Padmavathi, Meenakshi, K., Chaitanya, Duggineni, Valiveti, Himabindu: CNN based monocular depth estimation. E3S Web Conf. 309, 01070 (2021). [DOI: 10.1051/e3sconf/202130901070](https://doi.org/10.1051/e3sconf/202130901070)

4. Hashimoto, N., Takamaeda-Yamazaki, S.: FADEC: FPGA-based acceleration of video depth estimation by HW/SW co-design. In: 2022 International Conference on Field-Programmable Technology (ICFPT), Hong Kong, pp. 1–9 (2022). [DOI: 10.1109/ICFPT56656.2022.9974565](https://doi.org/10.1109/ICFPT56656.2022.9974565)

5. ZiWen, D., YuQi, L., Dong, Y.: FasterMDE: A real-time monocular depth estimation search method that balances accuracy and speed on the edge. Appl Intell 53, 24566–24586 (2023). [DOI: 10.1007/s10489-023-04872-2](https://doi.org/10.1007/s10489-023-04872-2)

6. Sada, Y., Soga, N., Shimoda, M., Jinguji, A., Sato, S., Nakahara, H.: Fast monocular depth estimation on an FPGA. In: 2020 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), New Orleans, LA, USA, pp. 143–146 (2020). [DOI: 10.1109/IPDPSW50202.2020.00032](https://doi.org/10.1109/IPDPSW50202.2020.00032)

7. Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., Koltun, V.: Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence 44(3) (2022). [DOI: 10.1109/TPAMI.2021.3072546](https://doi.org/10.1109/TPAMI.2021.3072546)



## Depth map comparison

### Results on NYUv2 dataset. From top to bottom: rgb image, ground truth disparity map, float model inference, quantized model inference on FPGA.
<p align="center">
  <img src="figures/nyu.png" alt="Example Image" width="99%" />
</p>



### Results on Coco dataset. Depth maps are generated on DPU (FPGA) using quantized model weights. 
<p align="center">
  <img src="figures/coco.png" alt="Example Image" width="99%" />
</p>



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
