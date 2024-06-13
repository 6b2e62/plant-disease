import argparse
import glob
import os
import shutil
import zipfile
from pathlib import Path

import cv2
import gdown
from tqdm import tqdm

main_path = Path("data/")
path_to_train_and_valid = main_path / "%s/**/*.*"
original_dataset_name = "original_dataset"

parser = argparse.ArgumentParser()
parser.add_argument("--download", action="store_true",
                    help="Download the data")
parser.add_argument("--unzip", action="store_true",
                    help="Unzip the data")
parser.add_argument("--resize", action="store_true",
                    help="Resize the dataset")
parser.add_argument("--shape", type=int, nargs="+", default=(64, 64),
                    help="Shape of the resized images. Applied only for resize option. Default: (64, 64)")
parser.add_argument("--sobel", action="store_true",
                    help="Apply Sobel filter to the dataset")
parser.add_argument("--source", type=str, default="original_dataset",
                    help="Name of the source dataset. Applied for all arguments except download. Default: original_dataset")
args = parser.parse_args()


class FileManager:

    def download_data(self):
        print("Downloading")
        if not os.path.isfile("archive.zip"):
            gdown.download(id="1IB9T0MTYcF_MIR7_AAjQItWUbvqDMq7c",
                           output="archive.zip")

    def unzip_data(self, file_name, path_to_extract):
        full_path_to_extract = main_path / path_to_extract
        old_path = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
        if not os.path.exists(main_path):
            os.makedirs(main_path)

        with zipfile.ZipFile(file_name) as zf:
            members_to_extract = [member for member in zf.infolist(
            ) if member.filename.startswith(old_path)]

            for member in tqdm(members_to_extract, desc='Extracting'):
                try:
                    zf.extract(member, full_path_to_extract)
                except zipfile.error as e:
                    raise e

        shutil.move(full_path_to_extract / old_path / "train",
                    full_path_to_extract / "train_valid", copy_function=shutil.copytree)
        shutil.copytree(full_path_to_extract / old_path / "valid",
                        full_path_to_extract / "train_valid", dirs_exist_ok=True)
        shutil.rmtree(
            full_path_to_extract / "New Plant Diseases Dataset(Augmented)"
        )
        self.get_test_ds(100)
        self.reduce_ds_size()
        self.train_valid_split()

    def write_image(self, image, path):
        os.makedirs(path.rsplit('/', 1)[0], exist_ok=True)
        cv2.imwrite(path, image)

    def reduce_ds_size(self, files_per_category: int = 1000):
        path_to_extract = main_path / original_dataset_name
        valid_ds = glob.glob(str(path_to_extract / "train_valid/*/*"))

        category_dirs = set([category_dir.split("/")[-2]
                            for category_dir in valid_ds])
        category_lists = {category: [] for category in category_dirs}
        for file_path in valid_ds:
            category = file_path.split("/")[-2]
            category_lists[category].append(file_path)

        for category, files in category_lists.items():
            files.sort()
            for file in files[files_per_category-1:]:
                os.remove(file)

    def get_test_ds(self, files_per_category: int = 2):
        path_to_extract = main_path / original_dataset_name
        valid_ds = glob.glob(str(path_to_extract / "train_valid/*/*"))

        category_dirs = set([category_dir.split("/")[-2]
                            for category_dir in valid_ds])
        category_lists = {category: [] for category in category_dirs}
        for file_path in valid_ds:
            category = file_path.split("/")[-2]
            category_lists[category].append(file_path)

        test_dir = path_to_extract / "test"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir, exist_ok=True)

        for category, files in category_lists.items():
            os.makedirs(test_dir / category, exist_ok=True)
            files.sort()
            for file in files[:files_per_category]:
                shutil.move(file, test_dir / category)

    def train_valid_split(self, split_ratio=0.8):
        path_to_extract = main_path / original_dataset_name
        ds = glob.glob(str(path_to_extract / "train_valid/*/*"))

        category_dirs = set([category_dir.split("/")[-2]
                            for category_dir in ds])
        category_lists = {category: [] for category in category_dirs}
        for file_path in ds:
            category = file_path.split("/")[-2]
            category_lists[category].append(file_path)

        train_dir = path_to_extract / "train"
        valid_dir = path_to_extract / "valid"
        if not os.path.exists(train_dir):
            os.makedirs(train_dir, exist_ok=True)
        if not os.path.exists(valid_dir):
            os.makedirs(valid_dir, exist_ok=True)

        for category, files in category_lists.items():
            os.makedirs(train_dir / category, exist_ok=True)
            os.makedirs(valid_dir / category, exist_ok=True)
            files.sort()
            split_index = int(len(files) * split_ratio)
            for file in files[:split_index]:
                shutil.move(file, train_dir / category)
            for file in files[split_index:]:
                shutil.move(file, valid_dir / category)

    def resize_dataset(self, source_dataset_name, shape):
        dataset_name = "resized_dataset_%s_%s"
        if not os.path.exists(main_path / dataset_name):
            counter = 0
            for file in glob.glob(str(path_to_train_and_valid) % source_dataset_name, recursive=True):
                if file.find('train_valid') >= 0:
                    continue
                counter += 1
                path_to_file = file.replace("\\", "/")
                image = cv2.imread(path_to_file)
                image = cv2.resize(image, shape)
                new_path = path_to_file.replace(
                    source_dataset_name, dataset_name % (shape[0], shape[1]))
                self.write_image(image, new_path)
                print("Resized %s files" % (counter), end='\r')

    def sobelx(self, source_dataset_name):
        dataset_name = "sobel_dataset"
        if not os.path.exists(main_path / dataset_name):
            counter = 0
            for file in glob.glob(str(path_to_train_and_valid) % source_dataset_name, recursive=True):
                if file.find('train_valid') >= 0:
                    continue
                counter += 1
                path_to_file = file.replace("\\", "/")
                image = cv2.imread(path_to_file)
                sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
                new_path = path_to_file.replace(
                    source_dataset_name, dataset_name)
                self.write_image(sobel, new_path)
                print("Sobel processed %s files" % (counter), end='\r')


if __name__ == "__main__":
    file_manager = FileManager()
    if args.download:
        file_manager.download_data()
    if args.unzip:
        file_manager.unzip_data("archive.zip", original_dataset_name)
    if args.resize:
        file_manager.resize_dataset(args.source, tuple(args.shape))
    if args.sobel:
        file_manager.sobelx(args.source)
