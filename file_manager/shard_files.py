from pathlib import Path

# TODO: split the files into smaller dirs and make list of them
class FileSharder:
    def __init__(self,
                 train_dir: Path = Path('./data/resized_dataset/train'),
                 valid_dir: Path = Path('./data/resized_dataset/valid'),
                 test_dir: Path = Path('./data/resized_dataset/test'),
                 shard_size = 5_000) -> None:
        self.shard_size = shard_size

        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir

        self.shard()

    def shard(self):
        pass
