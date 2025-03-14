from pathlib import Path
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

def find_classes(directory):
    directory = Path(directory)
    class_names = sorted(entry.name for entry in os.scandir(directory))

    if not class_names:
        raise FileNotFoundError(f"No valid class names can be found in {directory}. Please check the file structure.")

    class_to_idx = {}
    for idx, name in enumerate(class_names):
        class_to_idx[name] = idx

    return class_names, class_to_idx

class custom_data(Dataset):

    def __init__(self, directory):
        directory = Path(directory)
        self.path_list = list(directory.glob("*/*.png"))
        self.classes, self.class_to_idx = find_classes(directory)

    def load_image(self, index):
        img = Image.open(self.path_list[index])
        return img

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):

        img = self.load_image(index)

        simpleTransform = transforms.ToTensor()
        imgTensor = simpleTransform(img)
        imgNorm = (imgTensor*2) - 1

        class_name = self.path_list[index].parent.stem
        class_label = self.class_to_idx[class_name]

        return imgNorm, class_label
