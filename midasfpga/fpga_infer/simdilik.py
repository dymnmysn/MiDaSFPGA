# %%
from PIL import Image
from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse
import matplotlib.pyplot as plt
%matplotlib inline

from PIL import Image
import numpy as np

import torch
import cv2
import numpy as np

from scipy.io import loadmat

import torch.utils.data as data
import torch.nn.functional as F
from torchvision.transforms import Compose

from MiDaS.midas.midas_net import MidasNet
from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet
from glob import glob
from PIL import Image

from utils.metric import BadPixelMetric
from dataset.nyuv2 import NyuDepthV2

from model.model import MidasNet_small
from midasfpga.configs.midas_config import Midas_Config
from midasfpga.utils.func_utils import  get_midas_transform, get_model

import matplotlib.pyplot as plt
from tqdm import tqdm


partition_index = 0
partition_length = 109

    
def validate_pc(model_path, nyu_data_path = 'data'):

    conf = Midas_Config()
    model, device = get_model(conf)
    transform = get_midas_transform()

    ds = NyuDepthV2(nyu_data_path, transform=transform)
    dl = data.DataLoader(
        ds, batch_size=1, num_workers=1, shuffle=False, pin_memory=True
    )
    
    # validate
    metric = BadPixelMetric()

    loss_sum = 0

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dl),total=654):

            for k, v in batch.items():
                batch[k] = v.to(device)

            prediction = model.forward(batch["image"])

            # resize prediction to match target
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=batch["mask"].shape[1:],
                mode="bilinear",
                align_corners=False,
            )
            prediction = prediction.squeeze(1)

            loss,pd,t,e = metric(prediction, batch["depth"], batch["mask"])
            
            loss_sum += loss
            DEBUG = False
            if DEBUG:
                import matplotlib.pyplot as plt
                plt.imshow(pd.cpu().numpy().squeeze())
                plt.show()
                plt.imshow(t.cpu().numpy().squeeze())
                plt.show()
                plt.imshow(e.cpu().numpy().squeeze())
                plt.show()
                plt.imshow(np.swapaxes(np.swapaxes(batch["image"].cpu().numpy().squeeze(),2,0),0,1))
                plt.show()
            

    print(f"bad pixel: {loss_sum / len(ds):.2f}")

# download from https://drive.google.com/file/d/1nqW_Hwj86kslfsXR7EnXpEWdO2csz1cC
MODEL_PATH = "midas_v21_small_256.pt"

# validate
#validate_pc(MODEL_PATH, NYU_DATA_PATH)

##############################

transform = Compose(
    [
        Resize(
            256,
            192,
            resize_target=None,
            keep_aspect_ratio=False,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)


divider='---------------------------'

from utils.dpu_utils import get_child_subgraph_dpu


def runDPU(id,start,dpu,img):

    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    batchSize = input_ndim[0]
    n_of_images = len(img)
    count = 0
    write_index = start
    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count

        '''prepare batch input/output '''
        outputData = []
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
        outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]

        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])

        '''run with batch '''
        job_id = dpu.execute_async(inputData,outputData)
        dpu.wait(job_id)

        '''store output vectors '''
        for j in range(runSize):
            out_q[write_index] = outputData[0][j]
            write_index += 1
        count = count + runSize


def app(image_dir,threads,model):

    listimage=os.listdir(image_dir)
    runTotal = len(listimage)

    global out_q
    out_q = [None] * runTotal

    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    ''' preprocess images '''
    print('Pre-processing',runTotal,'images...')
        

    print('Starting',threads,'threads...')
    threadAll = []
    start=0
    for i in range(threads):
        if (i==threads-1):
            end = len(img)
        else:
            end = start+(len(img)//threads)
        in_q = img[start:end]
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start=end

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print(divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" %(fps, runTotal, timetotal))

    #abslosses = []
    #for i in range(len(out_q)):
        #postprocess_and_save(out_q[i], save_path='output_image.png')
        #ground_truth, _ = listimage[i].split('_',1)
        #l = np.mean(np.abs(out_q[i]-ground_truth))
        #abslosses.append(l)

    #print('Loss mean: %.4f' %(np.array(abslosses).mean()))
    #print(divider)

    return timetotal

lossgirdi1 = []
lossgirdi2 = []
lossgirdi3 = []
outputs = []

sure = 0
for partition_index in range(4,5):

    ds = NyuDepthV2('data', transform=transform, partition_index = partition_index)
    dl = data.DataLoader(
        ds, batch_size=1, num_workers=1, shuffle=False, pin_memory=True
    )

    # validate
    metric = BadPixelMetric()

    loss_sum1 = 0
    loss_sum2 = 0
    loss_sum3 = 0
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    img = []
    for i, batch in tqdm(enumerate(dl),total=partition_length):
        img.append(batch['image'].permute(0,2,3,1))


    image_dir='images'
    threads=1
    model='/home/ubuntu/Downloads/target_kv260/fcn8/model/fcn8.xmodel'

    print(divider)
    ttotal = app(image_dir,threads,model)
    sure += ttotal
    plt.imshow(out_q[0])
    
    for i, batch in tqdm(enumerate(dl),total=9):

        # run model
        #batch['image'][...,[0,2]] = batch['image'][...,[2,0]] 
        prediction = out_q[i]

        # resize prediction to match target
        prediction = F.interpolate(
            torch.tensor(prediction).unsqueeze(0).unsqueeze(1),
            size=batch["mask"].shape[1:],
            mode="bilinear",
            align_corners=False,
        )
        prediction = prediction.squeeze(1)

        loss1,loss2, loss3,pd,t,e = metric(prediction, batch["depth"], batch["mask"])
        loss_sum1 += loss1
        loss_sum2 += loss2
        loss_sum3 += loss3
        
    lossgirdi1.append(loss_sum1 / len(ds))
    lossgirdi2.append(loss_sum2 / len(ds))
    lossgirdi3.append(loss_sum3 / len(ds))
    print(f"bad pixel1: {loss_sum1 / len(ds):.2f}")
    print(f"bad pixel2: {loss_sum2 / len(ds):.2f}")
    print(f"bad pixel:3 {loss_sum3 / len(ds):.2f}")

print("Average bad pixel: ", sum(lossgirdi1)/len(lossgirdi1))
print("Average bad pixel: ", sum(lossgirdi2)/len(lossgirdi2))
print("Average bad pixel: ", sum(lossgirdi3)/len(lossgirdi3))
print("Average fps: ", 654/sure)

ss = metric.scaleshifts



        





