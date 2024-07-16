import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from utils.metric import BadPixelMetric
from dataset.nyuv2 import NyuDepthV2
from midasfpga.configs.midas_config import Midas_Config
from midasfpga.utils.func_utils import  get_midas_transform, get_model
from tqdm import tqdm

def validate_pc(nyu_data_path = 'data', DEBUG = False):

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
        for i, batch in tqdm(enumerate(dl),total=len(ds)):

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

            loss1,loss2,loss3,pd,t,e = metric(prediction, batch["depth"], batch["mask"])
            
            loss_sum += loss1
            
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