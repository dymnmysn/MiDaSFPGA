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

# %%
from PIL import Image
import numpy as np

# %%
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

partition_index = 0
index_shift = 109


   
    
def validate(modelpath, nyu_data, nyu_split):
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # select device
    device = torch.device("cpu")
    print("device: %s" % device)

    # load network
    #model = MidasNet(MODEL_PATH, non_negative=True)
    from smallmidas import MidasNet_small
    model = MidasNet_small(align_corners=False)
    parameters = torch.load(modelpath, 
                            map_location=torch.device('cpu'))

    if "optimizer" in parameters:
        parameters = parameters["model"]

    model.load_state_dict(parameters)
    
    
    model.to(device)
    model.eval()

    # get data
    transform = Compose(
        [
            Resize(
                256,
                256,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    #img = transform(Image.open(image_path)).unsqueeze(0).to(model.device)
    ds = NyuDepthV2(NYU_DATA_PATH, NYU_SPLIT_PATH, split="train", transform=transform)
    dl = data.DataLoader(
        ds, batch_size=1, num_workers=1, shuffle=False, pin_memory=True
    )
    

    # validate
    #metric = BadPixelMetric()

    loss_sum = 0
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dl),total=654):
            #print(f"processing: {i + 1} / {len(ds)}")

            # to device
            for k, v in batch.items():
                batch[k] = v.to(device)

            # run model
            #batch['image'][...,[0,2]] = batch['image'][...,[2,0]] 
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
#validate(MODEL_PATH, NYU_DATA_PATH, NYU_SPLIT_PATH)

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
    for i, batch in tqdm(enumerate(dl),total=index_shift):
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

# %%
plt.imshow(out_q[8])

# %%
next(it)['image'].shape

# %%
plt.imsave('/home/ubuntu/Downloads/save/2.png', out_q[1], cmap='viridis')

# %%
import scipy.ndimage

# %%
zoom_factors = (0.4,0.4)


# %%
targets = [np.load(f'/home/ubuntu/Downloads/data/depth_{i:05d}.npy') for i in range(436,545)]

# %%
targets[0].max()

# %%
plt.imshow(targets[0])

# %%
plt.imshow(targets[0])

# %%
t[0].min()

# %%
tt = [t[i][50:-50,50:-50] for i in range(index_shift)]

# %%
oref = [out_q[i][20:-20,20:-20] for i in range(index_shift)]

# %%
plt.imshow(oref[2])

# %%
plt.imshow(tt[2])

# %%
for i in range(index_shift):
    ttt= F.interpolate(
            torch.tensor(tt[i]).unsqueeze(0).unsqueeze(0),
            size=(152, 216),
            mode="bilinear",
            align_corners=False,
        ).numpy().squeeze()
    tt[i] = ttt

# %%
zoom_factors = (0.4, 0.4)
tt = [scipy.ndimage.zoom(tt[i], zoom_factors, order=1) for i in range(index_shift)]

# %%
ss = metric.scaleshifts

# %%
for i in range(index_shift):
    scale, shift = ss[i][0].item(), ss[i][1].item()
    prediction = out_q[i]

    # resize prediction to match target
    prediction = F.interpolate(
        torch.tensor(prediction).unsqueeze(0).unsqueeze(1),
        size=batch["mask"].shape[1:],
        mode="bilinear",
        align_corners=False,
    )
    prediction= prediction.squeeze()
    prediction_aligned = prediction*scale + shift
    prediction_aligned[prediction_aligned < 0.1] = 0.1
    prediciton_depth = 1.0 / prediction_aligned
    mask = targets[i]==0
    targets[i][mask] = prediciton_depth[mask]

# %%
plt.imshow(t[1][50:-50,50:-50])

# %%
plt.imshow(oref[1])

# %%
oref[0].shape

# %%
t = [np.clip((1/targets[i]-ss[i][1].item())/ss[i][0].item() , oref[i].min(), oref[i].max()) for i in range(index_shift)]

# %%
i = 0
a = np.clip((targets[i]-ss[i][1].item())/ss[i][0].item() , out_q[i].min(), out_q[i].max()) 

# %%
plt.imshow(targets[0])

# %%
targets[0].min()

# %%
targets[0][410][580]

# %%
from PIL import Image

def process_image(index = 0, clip_pixels=50, resize_factor=0.4):
    input_path, output_path = f'/home/ubuntu/Downloads/data/rgb_{(index+436):05d}.png', f'/home/ubuntu/Downloads/qualitative/nyu/rgb_{index}.png'
    # Open the image
    image = Image.open(input_path)

    # Clip pixels from each edge
    width, height = image.size
    clipped_image = image.crop((clip_pixels, clip_pixels, width - clip_pixels, height - clip_pixels))

    # Resize the image
    new_width = int(clipped_image.width * resize_factor)
    new_height = int(clipped_image.height * resize_factor)
    resized_image = clipped_image.resize((new_width, new_height), Image.BILINEAR)

    # Save the resized image
    resized_image.save(output_path)


# %%
for i in range(index_shift):
    process_image(i)

# %%
class TargetDisp:
    def __init__(self, threshold=1.25, depth_cap=10):
        self.__threshold = threshold
        self.__depth_cap = depth_cap

    def __call__(self, target):
        # transform predicted disparity to aligned depth
        target_disparity = torch.clip(target,0.01,10)
        target_disparity = 1.0 / target_disparity
        return target_disparity
        

# %%
def minmax_normalize_and_save(array ,type='depth', index=0):
    # Min-max normalization
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    output_filename = f"/home/ubuntu/Downloads/qualitative/nyu/{type}_{index}.png"
    # Save normalized array as PNG using Matplotlib
    plt.imsave(output_filename, normalized_array, cmap='viridis')
    #print(f"Normalized array saved as {output_filename}")

# %%
#tdisp =  TargetDisp()
for i in range(12):
    minmax_normalize_and_save(out_q[i] ,type='rand', index=i)
    #minmax_normalize_and_save(oref[i] ,type='pred', index=i)
    #minmax_normalize_and_save(tt[i] ,type='gt', index=i)
    #loss1,loss2, loss3,pd,t,e = metric(prediction, batch["depth"], batch["mask"])

# %%
#out_q[0].shape
prediction.shape,batch['depth'].shape

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Activate interactive mode for matplotlib
plt.ion()

cap = cv2.VideoCapture(0)  # Use the default camera (change the index if you have multiple cameras)

# Create a figure for displaying the image
fig, ax = plt.subplots()
img_plot = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
plt.title('Input Image')
plt.axis('off')

while True:
    ret, f = cap.read()

    if f is not None:
        #fr = cv2.resize(f, (64, 64))
        frame = f.astype(np.float32) / 256

        # Update the image data
        img_plot.set_data(frame)

        # Refresh the display
        fig.canvas.draw()

        # Pause for a short duration (adjust as needed)
        plt.pause(1)

# %%
def postprocess_and_save(output_tensor, save_path='output_image.png'):
    # Normalize the output tensor between 0 and 255
    normalized_output = ((output_tensor - output_tensor.min()) / (output_tensor.max() - output_tensor.min())) * 255
    normalized_output = normalized_output.astype(np.uint8) # Convert to byte tensor for image saving

    # Convert the byte tensor to a numpy array
    np_output = normalized_output

    # Convert the numpy array to a PIL image
    pil_image = Image.fromarray(np_output)#.transpose((1, 2, 0)))

    # Save the PIL image as a PNG
    pil_image.save(save_path)

# Example usage:
# Assuming `output_tensor` is your output tensor from the neural network
# postprocess_and_save(output_tensor, 'output_image.png')

# %%
import os
from PIL import Image


output_dir = '/home/ubuntu/Downloads/imagess'
input_dir = '/home/ubuntu/Downloads/images'
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the new size
new_size = (256, 192)  # (width, height)

# Process each image in the input directory
for i,filename in enumerate(glob('/home/ubuntu/Downloads/images/*')):
    with Image.open(filename) as img:
        # Resize image
        resized_img = img.resize(new_size, Image.ANTIALIAS)
        
        # Save resized image to the output directory
        resized_img.save(os.path.join(output_dir, f'{i}.png'))
        
        print(f"Resized and saved {filename}")


print("Image resizing completed.")

# %%
def preprocess_fn(image):
    # Resize to (256, 256)
    print(image)
    resized_image = np.array(Image.open(image).resize((256, 256)))
    
    # Convert to tensor
    tensor_image = np.transpose(resized_image, (2, 0, 1))  # Convert to channels-first format
    tensor_image = tensor_image / 255.0  # Normalize to [0, 1]
    tensor_image = (tensor_image - np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))) / np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))  # Normalize
    
    tensor_image = np.transpose(tensor_image, (1, 2, 0))  # Convert to channels-first format
    return tensor_image

# %%
divider='---------------------------'

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

# %%
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
    img = []
    for i in range(runTotal):
        path = os.path.join(image_dir,listimage[i])
        img.append(preprocess_fn(path))

    '''run threads '''
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
    print(divider)

    return

# %%
image_dir='images'
threads=1
model='/home/ubuntu/Downloads/target_kv260/fcn8/model/fcn8.xmodel'

print(divider)

app(image_dir,threads,model)

# %%




