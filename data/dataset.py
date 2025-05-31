import os
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import pickle
import cv2

hw_ratio = 4

class ImagePaths(Dataset):
    def __init__(self, size=32, labels=None):
        self.max_h = size
        self.max_w = size * hw_ratio
        self.labels = labels
        self._length = len(labels[list(labels.keys())[0]])
        self.resize = albumentations.Resize(height=self.max_h, width=self.max_w)
    
    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        img = Image.open(image_path)
        if not img.mode == "RGB":
            img = img.convert("RGB")
        
        width, height = img.size
        ori_size = (width, height)
        img = np.array(img).astype(np.uint8)
        img = self.resize(image = img)["image"]
        img = (img / 127.5 - 1.0).astype(np.float32)
        height, width, _ = img.shape
        return img, ori_size, (width, height)

    def __getitem__(self, i):
        example = dict()
        try:
            example['img_name'] = os.path.basename(self.labels["image1_paths"][i])
            example['image1'], example['ori_size'], example["image1_size"] = self.preprocess_image(self.labels["image1_paths"][i])
            if "image1_rec" in self.labels and len(self.labels['image1_rec']) != 0:
                example['rec1'] = self.labels["image1_rec"][i]
            if "image2_rec" in self.labels and len(self.labels['image2_rec']) != 0:
                example['rec2'] = self.labels["image2_rec"][i]
            if 'image2_paths' in self.labels and len(self.labels['image2_paths']) != 0:
                example['image2'], example['ori_size'], example["image2_size"] = self.preprocess_image(self.labels["image2_paths"][i])
            return example
        except:
            print("Error in loading image.")
            return self.__getitem__(i+1)

class BaseDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.data_proportion = None
        self._length = None
    
    def __len__(self):
        if self._length is not None:
            return self._length
        
        if isinstance(self.data, list):
            self._length = sum([len(d) for d in self.data])
        else:
            self._length = len(self.data)
        return self._length
    
    def __getitem__(self, i):
        if isinstance(self.data_proportion, list):
            p = np.array(self.data_proportion) / sum(self.data_proportion)
            data_index = np.random.choice(np.arange(len(self.data)), p = p.ravel())
            data = self.data[data_index]
            example = data[i % len(data)]
        else:
            example = self.data[i]
        return example
    
class TrainingDataset(BaseDataset):
    def __init__(self, size, annotation_file, data_proportion=None):
        super().__init__()
        if isinstance(annotation_file, str):
            with open(annotation_file, "rb") as f:
                data = pickle.load(f)
            self.data = ImagePaths(size=size, labels=data)
        else:
            if data_proportion == None:
                data = {}
                for file in annotation_file:
                    with open(file, "rb") as f:
                        data_ = pickle.load(f)
                    for k in data_:
                        if k in data:
                            data[k] += data_[k]
                        else:
                            data[k] = data_[k]
                self.data = ImagePaths(size=size, labels=data)
            else:
                data_proportion = list(data_proportion)
                self.data = []
                for file in annotation_file:
                    with open(file, "rb") as f:
                        data = pickle.load(f)
                    self.data.append(ImagePaths(size=size, labels=data))
                    self.data_proportion = data_proportion

class InferenceDataset(BaseDataset):
    def __init__(self, size, annotation_file):
        super().__init__()
        with open(annotation_file, "rb") as f:
            data = pickle.load(f)
        self.data = ImagePaths(size=size, labels=data)