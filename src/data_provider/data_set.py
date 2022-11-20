from torch.utils.data import Dataset
import os


class ImageDataset(Dataset):

    DIRECTORY = (
        "/Users/sebastian/Documents/Projects/sordi_ai/src/data/SORDI_2022_Single_Assets"
    )
    IMG_EXTENSIONS = (
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    )

    def __init__(
        self,
        root_path: str,
        data_path: str,
        flag="train",
    ) -> None:
        super().__init__()
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag

    def _set_directory(self) -> None:
        self.DIRECTORY = os.path.join(self.root_path, self.data_path)

    def get_raw_image(self) -> None:
        pass

    def _make_dataset(self) -> None:
        raise NotImplementedError

    def __len__(self) -> None:
        raise NotImplementedError

    def __getitem__(self, index) -> None:
        raise NotImplementedError
