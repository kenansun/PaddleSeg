
import os

import paddle
import numpy as np
from PIL import Image
import torch 
import rasterio
import torch.nn.functional as F
from paddleseg.utils import seg_env
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
from paddleseg.datasets import Dataset

@manager.DATASETS.add_component
class RSDATASET(Dataset):

    def __init__(self,
                 mode,
                 dataset_root,
                 num_classes,
                 transforms=None,
                 img_channels=3,
                 train_path=None,
                 val_path=None,
                 test_path=None,
                 separator=' ',
                 ignore_index=255,
                 edge=False):
        self.dataset_root = dataset_root
        if transforms is not None : self.transforms = Compose(transforms, img_channels=img_channels,to_rgb=False)
        self.file_list = list()
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.ignore_index = ignore_index
        self.edge = edge

        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    self.mode))
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError('there is not `dataset_root`: {}.'.format(
                self.dataset_root))
        # if self.transforms is None:
        #     raise ValueError("`transforms` is necessary, but it is None.")
        if num_classes < 1:
            raise ValueError(
                "`num_classes` should be greater than 1, but got {}".format(
                    num_classes))
        if img_channels not in [1, 3]:
            raise ValueError("`img_channels` should in [1, 3], but got {}".
                             format(img_channels))

        if self.mode == 'train':
            if train_path is None:
                raise ValueError(
                    'When `mode` is "train", `train_path` is necessary, but it is None.'
                )
            elif not os.path.exists(train_path):
                raise FileNotFoundError('`train_path` is not found: {}'.format(
                    train_path))
            else:
                file_path = train_path
        elif self.mode == 'val':
            if val_path is None:
                raise ValueError(
                    'When `mode` is "val", `val_path` is necessary, but it is None.'
                )
            elif not os.path.exists(val_path):
                raise FileNotFoundError('`val_path` is not found: {}'.format(
                    val_path))
            else:
                file_path = val_path
        else:
            if test_path is None:
                raise ValueError(
                    'When `mode` is "test", `test_path` is necessary, but it is None.'
                )
            elif not os.path.exists(test_path):
                raise FileNotFoundError('`test_path` is not found: {}'.format(
                    test_path))
            else:
                file_path = test_path
        
        LABEL_FILENAME = "y.tif"
        stats = dict(
            rejected_nopath=0,
            rejected_length=0,
            total_samples=0)
        self.data_dirs = [d for d in os.listdir(self.dataset_root) if d.startswith("data")]
        self.classids, self.classes = self.read_classes(os.path.join(self.dataset_root, "classes.txt"))
        dirs = []
        # tileids e.g. "tileids/train_fold0.tileids" path of line separated tileids specifying
        with open(os.path.join(file_path), 'r') as f:
            files = [el.replace("\n", "") for el in f.readlines()]
        for d in self.data_dirs:
            dirs_path = [os.path.join(self.root_dir, d, f) for f in files]
            dirs.extend(dirs_path)
        for path in dirs:
            if not os.path.exists(path):
                stats["rejected_nopath"] += 1
                continue
            if not os.path.exists(os.path.join(path, LABEL_FILENAME)):
                stats["rejected_nopath"] += 1
                continue
            stats["total_samples"] += 1
            self.file_list.append(path)
        print(stats)
    def __getitem__(self, idx):
        data = {}
        data['trans_info'] = []
        path = self.file_list[idx]
        if path.endswith(os.sep):
            path = path[:-1]
        label, profile = read(os.path.join(path, self.LABEL_FILENAME))
        profile["name"] = self.samples[idx]
        dates = get_dates(path)        
        x10 = None
        x20 = None
        x60 = None
        for date in dates:
            if(int(date[4:6]) >= 6):
                x10 = read(os.path.join(path, date + "_10m.tif"))[0]
                x20 = read(os.path.join(path, date + "_20m.tif"))[0]
                x60 = read(os.path.join(path, date + "_60m.tif"))[0]
                if x10 is not None and x20 is not None and x60 is not None:
                    break
        x10 = np.array(x10) * 1e-4
        x20 = np.array(x20) * 1e-4
        x60 = np.array(x60) * 1e-4
        label = label[0]
        self.unique_labels = np.unique(np.concatenate([label.flatten(), self.unique_labels]))
        new = np.zeros(label.shape, np.int)
        for cl, i in zip(self.classids, range(len(self.classids))):
            new[label == cl] = i
        label = new
        label = torch.from_numpy(label)
        x10 = torch.from_numpy(x10)
        x20 = torch.from_numpy(x20)
        x60 = torch.from_numpy(x60)
        x20 = torch.unsqueeze(x20, 0)
        x60 = torch.unsqueeze(x60, 0)
        x20 = F.interpolate(x20, size=x10.shape[1:3])
        x60 = F.interpolate(x60, size=x10.shape[1:3])
        x20 = torch.squeeze(x20, 0)
        x60 = torch.squeeze(x60, 0)
        x = torch.cat((x10, x20, x60), 0)
        # permute channels with time_series (c x h x w) -> ( h x w x c)
        x = x.permute(1, 2, 0)
        x = x.float()
        label = label.long()
        
        data['gt_fields'] = []
        if self.mode == 'val':
            data = self.transforms(data)
            if data['label'].ndim == 2:
                data['label'] = data['label'][np.newaxis, :, :]
        else:
            data['gt_fields'].append('label')
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.file_list)    
    
def read(file):
    with rasterio.open(file) as src:
        return src.read(), src.profile


def get_dates(path, n=None):
    """
    extracts a list of unique dates from dataset sample

    :param path: to dataset sample folder
    :param n: choose n random samples from all available dates
    :return: list of unique dates in YYYYMMDD format
    """

    files = os.listdir(path)
    dates = list()
    for f in files:
        f = f.split("_")[0]
        if len(f) == 8:  # 20160101
            dates.append(f)

    dates = set(dates)

    # if n is not None:
    #     dates = random.sample(dates, n)

    dates = list(dates)
    dates.sort()
    return dates