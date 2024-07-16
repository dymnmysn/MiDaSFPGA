import time
import xir
import vart
import numpy as np
from typing import List
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