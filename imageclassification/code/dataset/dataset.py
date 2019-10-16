import sys
import glob
import pandas as pd
import torch.utils.data as data
import cv2
import torch


defect_label = {
    "EOSINOPHIL":"0",
    "LYMPHOCYTE":"1",
    "MONOCYTE":"2",
    "NEUTROPHIL":"3",
}

def get_image_pd(image_root):
    image_path = glob.glob(image_root + "/*/*.jpeg")
    image_pd = pd.DataFrame(image_path, columns=["image_path"])
    image_pd["label_name"] = image_pd["image_path"].apply(lambda x : x.split("/")[-2])
    image_pd["label"] = image_pd["label_name"].apply(lambda x:defect_label[x])
    print(image_pd["label"].value_counts())

    return image_pd

class dataset(data.Dataset):
    def __init__(self, pd, transforms=None):
        self.path = pd["image_path"].tolist()
        self.label = pd["label"].tolist()
        self.transforms = transforms
    
    def __len__(self):
        return (len(self.path))

    def __getitem__(self, item):
        image = cv2.imread(self.path[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.label[item]

        return torch.from_numpy(image).float(), int(label)

