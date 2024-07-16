from PIL import Image
import numpy as np
import torch.utils.data as data
from glob import glob

class CustomDataset(data.Dataset):
    def __init__(self, datapath = 'data', transform=None):

        self.__image_list = []
        self.__depth_list = []

        self.__transform = transform
        img_pathes = sorted(glob(f"{datapath}/*.png"))
        depth_pathes = sorted(glob(f"{datapath}/*.npy"))
        files = [(img_pathes[i], depth_pathes[i]) for i in range(len(img_pathes))]
        for id in files:
            i,d = id
            #self.__image_list.append(np.array(Image.open(i))) 
            self.__image_list.append(Image.open(i))
            self.__depth_list.append(np.load(d))
        
        self.__length = len(self.__image_list)

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        # image
        image = self.__image_list[index]
        #image = image / 255 #ToTensor deals with it

        # depth and mask
        depth = self.__depth_list[index]
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