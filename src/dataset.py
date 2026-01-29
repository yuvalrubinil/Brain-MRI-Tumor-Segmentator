import kagglehub
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import random_split
from torch.utils.data import Subset
import pandas as pd

class MRIDataset(Dataset):
    def __init__(self, root_dir, limit=None, img_size=256):
        self.samples = []
        self.img_size = img_size

        # take the MRI scans, resize then and convert to tensor
        self.transform_img = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

        # use Interpolation Mode NEAREST for masks (ground truth images) to avoid creating "blurry" pixels during resizing
        self.transform_mask = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

        i = 0
        # look into all patient folders
        for patient in os.listdir(root_dir):

            if limit and i >= limit: # allows to look into only a specific amount of patients for debugging
                break

            patient_dir = os.path.join(root_dir, patient)

            if not os.path.isdir(patient_dir):
                continue

            # find all images and pair them with their GT
            for fname in os.listdir(patient_dir):
                if fname.endswith(".tif") and not fname.endswith("_mask.tif"):
                    img_path = os.path.join(patient_dir, fname)
                    mask_path = img_path.replace(".tif", "_mask.tif")

                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))
            i += 1

            # now the samples array is stored as [MRI Image 1, Ground Truth 1, MRI Image 2, Ground Truth 2, ....]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # load as Grayscale to get rid of redundant information
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # apply transforms
        image = self.transform_img(image)
        mask = self.transform_mask(mask)

        # ensure mask is strictly binary (0 or 1)
        # we use a threshold because T.ToTensor() scales pixels to [0, 1]
        mask = (mask > 0.1).float()

        return image, mask

def download_dataset(kagglehub_path):
    dataset_root = kagglehub.dataset_download(f"{kagglehub_path}/mateuszbuda/lgg-mri-segmentation")
    print("Dataset downloaded at: ", dataset_root)
    return dataset_root

def split_dataset(dataset):
    # split: 80% train, 10% validation, 10% test
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - 2 * val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, val_size])

    splits = {
        "test": test_dataset,
        "val": val_dataset,
        "train": train_dataset,
    }

    for name, subset in splits.items():
        indices = subset.indices
        files = [dataset.samples[i][0] for i in indices]

        df = pd.DataFrame(files, columns=["img_path"])
        out_path = f"./dataset/{name}_set_filenames.csv"
        df.to_csv(out_path, index=False)

        print(f"Successfully saved {len(files)} {name} filenames.")

def load_subsets(dataset, train=False, val=False, test=False):
    # load csv files
    train_df = pd.read_csv("dataset/train_set_filenames.csv")
    val_df = pd.read_csv("dataset/val_set_filenames.csv")
    test_df = pd.read_csv("dataset/test_set_filenames.csv")

    # recover indices
    path_to_idx = {path: idx for idx, (path, _) in enumerate(dataset.samples)}
    train_indices = [path_to_idx[p] for p in train_df["img_path"]]
    val_indices = [path_to_idx[p] for p in val_df["img_path"]]
    test_indices = [path_to_idx[p] for p in test_df["img_path"]]

    # create subsets
    train_dataset = Subset(dataset, train_indices) if train else None
    val_dataset = Subset(dataset, val_indices) if val else None
    test_dataset = Subset(dataset, test_indices) if test else None
    return train_dataset, val_dataset, test_dataset