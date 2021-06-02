import os
from typing import Callable, Optional
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.datasets import VisionDataset


class LFW_People(VisionDataset):
    """
    LFW People dataset customized as per the `peopleDevTrain.txt` and `peopleDevTest.txt` files.
    Args:
        root (string): Root directory path, corresponding to `self.root`.
        people_path (string): Path to text file having people.
            E.g, `people.txt` for train set.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        ext (string): Extension of the images files.
     Attributes:
        people (numpy.array): Array of the identities and respective number of images.
        img_paths (list): List of image paths.
        ids (list): List of identities, used to get names from image labels.
        labels (list): List of integer labels of respective identities.
    """

    def __init__(
            self,
            root: str,
            people_path: str = "peopleDevTrain.txt",
            transform: Optional[Callable] = None,
            ext: str = "jpg"
    ) -> None:
        super(LFW_People, self).__init__(root, transform=transform)
        self.people = self._get_people(people_path)
        self.img_paths = []
        self.ids = []
        self.labels = []

        self.img_paths, self.ids, self.labels = self._get_people_paths(
            self.root, self.people, ext)

    def _get_people(self, people_path):
        people = []
        with open(people_path, 'r') as f:
            flines = f.readlines()
            # num_people = int(flines[0])
            # print("Number of people:", num_people)
            for line in flines[1:]:
                identity = line.strip().split()
                people.append(identity)
        return np.array(people)

    def _get_people_paths(self, images_dir, people, ext="jpg"):
        num_skipped = 0
        img_paths = []
        img_ids = []
        img_id_labels = []
        bar = tqdm(enumerate(people), desc="Getting images", total=len(people))
        print()
        for label, (identity, num_imgs) in bar:
            for idx in range(1, int(num_imgs) + 1):
                img_path = os.path.join(
                    images_dir, identity, "{}_{:04d}.{}".format(identity, idx, ext))
                if os.path.exists(img_path):
                    img_paths.append(img_path)
                    img_ids.append(identity)
                    img_id_labels.append(label)
                else:
                    num_skipped += 1
                    print("Not found:", img_path)
        if num_skipped > 0:
            print(f"Skipped {num_skipped} images")

        return img_paths, img_ids, img_id_labels

    def __len__(self):
        return self.img_paths.__len__()

    def loader(self, path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        label = self.labels[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label


class LFW_Pairs(VisionDataset):
    """
    LFW Pairs dataset customized as per the `pairsDevTrain.txt` and `pairsDevTest.txt` files.
    Args:
        root (string): Root directory path, corresponding to `self.root`.
        pairs_path (string): Path to text file having people.
            E.g, `pairs.txt`.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        ext (string): Extension of the images files.
     Attributes:
        pairs (numpy.array): Array of the pairs from the text file.
        is_same_list (list): List of boolean values corresponding to each pair being
            `same` or `different`.
        ids (list): List of identities in a pair.
    """

    def __init__(
            self,
            root: str,
            pairs_path: str,
            transform: Optional[Callable] = None,
            ext: str = "jpg"
    ) -> None:
        super(LFW_Pairs, self).__init__(root, transform=transform)
        self.pairs = self._get_pairs(pairs_path)
        self.pair_paths = []
        self.is_same_list = []
        self.ids = []

        self.pair_paths, self.is_same_list, self.ids = self._get_pairs_path(
            self.root, self.pairs, ext)

    def _get_pairs(self, pairs_path):
        pairs = []
        with open(pairs_path, 'r') as f:
            flines = f.readlines()
            for line in flines[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs, dtype=object)

    def _get_pairs_path(self, images_dir, pairs, ext="jpg"):
        num_skipped = 0
        pair_paths = []
        is_same = []
        img_ids = []
        bar = tqdm(pairs, desc="Getting pairs", total=len(pairs))
        print()
        for pair in bar:
            if len(pair) == 3:
                img1 = os.path.join(images_dir, pair[0], "{}_{:04d}.{}".format(
                    pair[0], int(pair[1]), ext))
                img2 = os.path.join(images_dir, pair[0], "{}_{:04d}.{}".format(
                    pair[0], int(pair[2]), ext))
                same = True
                img_ids.append((pair[0]))
            elif len(pair) == 4:
                img1 = os.path.join(images_dir, pair[0], "{}_{:04d}.{}".format(
                    pair[0], int(pair[1]), ext))
                img2 = os.path.join(images_dir, pair[2], "{}_{:04d}.{}".format(
                    pair[2], int(pair[3]), ext))
                same = False
                img_ids.append((pair[0], pair[2]))
            if os.path.exists(img1) and os.path.exists(img2):
                pair_paths.append((img1, img2))
                is_same.append(same)
            else:
                num_skipped += 1
                print("Not found:", img1, "or", img2)
            if num_skipped > 0:
                print(f"Skipped {num_skipped} images")
        return pair_paths, is_same, img_ids

    def __len__(self):
        return self.pair_paths.__len__()

    def loader(self, path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, idx):
        path1, path2 = self.pair_paths[idx]
        is_same = self.is_same_list[idx]
        sample1 = self.loader(path1)
        sample2 = self.loader(path2)
        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        return sample1, sample2, is_same
