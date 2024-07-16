import time
import xir
import vart
import numpy as np
from typing import List
from tqdm import tqdm

from dpu_utils import app
from dataset.nyuv2 import NyuDepthV2
from utils.metric import BadPixelMetric
from midasfpga.utils.func_utils import get_midas_transform
import torch
import torch.utils.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def runDPU(out_q, start,dpu,img):

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


def app(xmodel_path, dataloader):

    img = []
    for i, batch in tqdm(enumerate(dataloader),total=len(dataloader.dataset)):
        img.append(batch['image'].permute(0,2,3,1).cpu().numpy())

    runTotal = len(img) * img[0].shape[0]

    out_q = [None] * runTotal

    g = xir.Graph.deserialize(xmodel_path)
    subgraphs = get_child_subgraph_dpu(g)

    dpu_runner = vart.Runner.create_runner(subgraphs[0], "run")

    start=0
        
    time1 = time.time()
    out_q = runDPU(out_q, start, dpu_runner, img) #It may result slower inference this way
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print('---------------------------------------------------------------------------------------')
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" %(fps, runTotal, timetotal))

    return timetotal, out_q


def eval_speed(conf):

    xmodel_path = conf.FPGA_XMODEL_PATH 
    data_dir=conf.FPGA_DATA_DIR
    loss_delta_1 = []
    loss_delta_2 = []
    loss_delta_3 = []
    outputs = []
    total_time = 0
    transform = get_midas_transform()

    for partition_index in range(conf.FPGA_INDEX_LENGTH):
        ds = NyuDepthV2(data_dir, transform=transform, partition_index=partition_index)
        dl = data.DataLoader(
            ds, batch_size=1, num_workers=1, shuffle=False, pin_memory=True
        )
        metric = BadPixelMetric()
        loss_bsum1 = 0
        loss_bsum2 = 0
        loss_bsum3 = 0
        img = []

        for i, batch in tqdm(enumerate(dl), total=conf.FPGA_PARTITION_LENGTH):
            img.append(batch['image'].permute(0, 2, 3, 1))

        print('--------------------------------------------------------------')
        duration, out_q = app(xmodel_path, dl)
        outputs.append(out_q)
        total_time += duration

        for i, batch in tqdm(enumerate(dl), total=9):
            # run model
            prediction = out_q[i]

            # resize prediction to match target
            prediction = F.interpolate(
                torch.tensor(prediction).unsqueeze(0).unsqueeze(1),
                size=batch["mask"].shape[1:],
                mode="bilinear",
                align_corners=False,
            )
            prediction = prediction.squeeze(1)

            loss1, loss2, loss3, pd, t, e = metric(prediction, batch["depth"], batch["mask"])
            loss_bsum1 += loss1
            loss_bsum2 += loss2
            loss_bsum3 += loss3

        loss_delta_1.append(loss_bsum1 / len(ds))
        loss_delta_2.append(loss_bsum2 / len(ds))
        loss_delta_3.append(loss_bsum3 / len(ds))
        print(f"bad pixel1: {loss_bsum1 / len(ds):.2f}")
        print(f"bad pixel2: {loss_bsum2 / len(ds):.2f}")
        print(f"bad pixel3: {loss_bsum3 / len(ds):.2f}")

    print("Average bad pixel: ", sum(loss_delta_1) / len(loss_delta_1))
    print("Average bad pixel: ", sum(loss_delta_2) / len(loss_delta_2))
    print("Average bad pixel: ", sum(loss_delta_3) / len(loss_delta_3))
    print("Average fps: ", len(dl.dataset) / total_time)

    return outputs, metric.scaleshifts
