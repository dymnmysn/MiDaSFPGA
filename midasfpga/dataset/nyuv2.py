from PIL import Image
import numpy as np
import torch.utils.data as data
from glob import glob

class NyuDepthV2(data.Dataset):
    def __init__(self, datapath = 'data', transform=None, partition_index = 0, shift = 109):
        self.__transform = transform
        img_pathes = sorted(glob(f"{datapath}/rgb/*.png"))
        depth_pathes = sorted(glob(f"{datapath}/depth/*.npy"))
        files = [(img_pathes[i], depth_pathes[i]) for i in range(len(img_pathes))]
        self.files = files[partition_index*shift: (partition_index+1)*shift]
        self.__length = len(self.files)

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        # image
        i,d = self.files[index]
        image = Image.open(i)

        # depth and mask
        depth = np.load(d)
        mask = (depth > 0) & (depth < 10)

        # sample
        sample = {}
        sample["image"] = image
        sample["depth"] = depth
        sample["mask"] = mask

        # transforms
        if self.__transform is not None:
            sample["image"] = self.__transform(sample["image"])

        return sample
    