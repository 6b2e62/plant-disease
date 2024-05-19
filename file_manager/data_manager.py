import argparse
import glob
import os
import shutil
from pathlib import Path
import zipfile
from tqdm import tqdm

import cv2
import wget

main_path = Path("data/")
path_to_train_and_valid = main_path / "%s/**/*.*"
original_dataset_name = "original_dataset"

parser = argparse.ArgumentParser()
parser.add_argument("--download", action="store_true",
                    help="Download the data")
parser.add_argument("--resize", action="store_true",
                    help="Resize the dataset")
parser.add_argument("--shape", type=int, nargs="+", default=(64, 64),
                    help="Shape of the resized images. Applied only for resize option. Default: (64, 64)")
parser.add_argument("--sobel", action="store_true",
                    help="Apply Sobel filter to the dataset")
parser.add_argument("--source", type=str, default="original_dataset",
                    help="Name of the source dataset. Applied for all arguments except download. Default: original_dataset")
args = parser.parse_args()


class DataManager:

    def download_data(self):
        print("Downloading")
        if not os.path.isfile("archive.zip"):
            wget.download("https://storage.googleapis.com/kaggle-data-sets/78313/182633/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240512%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240512T222712Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=48a59c070b4f57e2746696d7ce6c77a7efd7e2d421a7d1f66411ab3fb21f688c0a0c81907ef4b788d99767cfd2e72200e6ee42e41fc4548d7874c128beabc2ff12d38aa47dfae1c10a10659e81b8d34c23515c1d6682bcf1f3eefc4e75a1ddba65dec5a03b03eec674272e772279b723f3a2739ca9099b185cf110cc5fb98e96b92622070c8cdd521f6ea0d676e6ba5dc37b23faf919e5dbc8b631e5be8e25f8da5bc74fbb94ff72393702c7348b3adf8140e80269d571ff00dd6aa065c43492d66131f62b9e59c503e1490851748c683680dbf3f929602239c6de297d329c9f3c4b12e389007783c88526f38064afbad73dba9897e408d8e1856b013fadc480")

    def unzip_data(self, file_name, path_to_extract):
        full_path_to_extract = main_path / path_to_extract
        old_path = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
        if not os.path.exists(main_path):
            os.makedirs(main_path)

        print("Extracting")
        with zipfile.ZipFile(file_name) as zf:
            for member in tqdm(zf.infolist(), desc='Extracting'):
                try:
                    zf.extract(member, full_path_to_extract)
                except zipfile.error as e:
                    pass
        # shutil.move("data/test/test",
        #             full_path_to_extract, copy_function=shutil.copytree)
        shutil.move(full_path_to_extract / old_path / "train",
                    full_path_to_extract / "train", copy_function=shutil.copytree)
        shutil.move(full_path_to_extract / old_path / "valid",
                    full_path_to_extract / "valid", copy_function=shutil.copytree)
        shutil.rmtree(
            full_path_to_extract / "New Plant Diseases Dataset(Augmented)"
        )
        shutil.rmtree(
            full_path_to_extract / "new plant diseases dataset(augmented)"
        )
        shutil.rmtree(full_path_to_extract / "test")
        self.get_test_ds_from_validation()

    def write_image(self, image, path):
        os.makedirs(path.rsplit('/', 1)[0], exist_ok=True)
        cv2.imwrite(path, image)

    def get_test_ds_from_validation(self, files_per_category: int = 2):
        path_to_extract = main_path / original_dataset_name
        valid_ds = glob.glob(str(path_to_extract / "valid/*/*"))

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

    def resize_dataset(self, source_dataset_name, shape):
        dataset_name = "resized_dataset_%s_%s"
        if not os.path.exists(main_path / dataset_name):
            counter = 0
            for file in glob.glob(str(path_to_train_and_valid) % source_dataset_name, recursive=True):
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
                counter += 1
                path_to_file = file.replace("\\", "/")
                image = cv2.imread(path_to_file)
                sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
                new_path = path_to_file.replace(
                    source_dataset_name, dataset_name)
                self.write_image(sobel, new_path)
                print("Sobel processed %s files" % (counter), end='\r')


if __name__ == "__main__":
    data_manager = DataManager()
    if args.download:
        data_manager.download_data()
        data_manager.unzip_data("archive.zip", original_dataset_name)
    if args.resize:
        data_manager.resize_dataset(args.source, tuple(args.shape))
    if args.sobel:
        data_manager.sobelx(args.source)
