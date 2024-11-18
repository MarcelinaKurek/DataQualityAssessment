import os
import kaggle
import zipfile


def view_kaggle_datasets(keyword, max_datasets=10):
    api = kaggle.api
    api.get_config_value("username")
    datasets = kaggle.api.dataset_list(search=keyword)
    print(f"{len(datasets)} datasets found with keyword: {keyword}. Showing top {min(max_datasets, len(datasets))} results.")
    for i in range(min(max_datasets,len(datasets))):
        print(str(i) + " : "+ str(datasets[i]))
        if datasets[i].description != "":
            print(datasets[i].description)
    return datasets


def download_kaggle_dataset(filename, directory="dataset_download"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    kaggle.api.dataset_download_files(filename.ref, path=directory)
    print(f"{filename.ref} successfully downloaded!")
    name_only = str(filename.ref).split('/')[-1]
    fullpath = os.path.join(directory, name_only)
    with zipfile.ZipFile(f"{fullpath}.zip", 'r') as zip_ref:
        target_path = os.path.join(directory, name_only)
        zip_ref.extractall(target_path)
    os.remove(f"{fullpath}.zip")


datasets_list = view_kaggle_datasets("classification")
download_kaggle_dataset(datasets_list[3], directory="test")