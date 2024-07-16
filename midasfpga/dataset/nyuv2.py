from PIL import Image
import numpy as np
import torch.utils.data as data

class NyuDepthV2(data.Dataset):
    def __init__(self, datapath = 'data', transform=None, partition_index = 0, shift = 109):

        self.__image_list = []
        self.__depth_list = []

        self.__transform = transform
        files = [(f'{datapath}/rgb_{i:05d}.png', f'{datapath}/depth_{i:05d}.npy') for i in range(partition_index*shift,partition_index*shift+shift)]
        for id in files:
            i,d = id
            self.__image_list.append(np.array(Image.open(i))) 
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
            sample = self.__transform(sample)

        return sample