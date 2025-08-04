import kagglehub

def download_kaggle_dataset(dataset_name: str):
    """Download a kaggle dataset
    dataset_name: str
    """

    path = kagglehub.dataset_download(dataset_name)
    print("Path to dataset files:", path)
    return path


if __name__ == "__main__":
    download_kaggle_dataset("abdulhasibuddin/plant-doc-dataset")
    download_kaggle_dataset("emmarex/plantdisease")
    download_kaggle_dataset("rtlmhjbn/ip02-dataset")