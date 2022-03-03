import os

def get_kaggle_dataset_from_url(dataset_url: str):
    if not os.path.isfile('~/.kaggle'):
        assert os.path.isfile('kaggle.json'), f'\n\n`kaggle.json` file not found, upload it in your current path. You can get it from your Kaggle Account settings.\n'

        os.system('mkdir ~/.kaggle -p')
        os.system('cp kaggle.json ~/.kaggle/')
        os.system('chmod 600 ~/.kaggle/kaggle.json')
        
    os.system(f'kaggle datasets download {"/".join(dataset_url.split("/")[-2:])}')


def get_file_from_url(file_url: str):
    pass
    # if not os.path.isfile('~/.kaggle'):
    #     assert os.path.isfile('kaggle.json'), f'\n\n`kaggle.json` file not found, upload it in your current path. You can get it from your Kaggle Account settings.\n'

    #     os.system('mkdir ~/.kaggle -p')
    #     os.system('cp kaggle.json ~/.kaggle/')
    #     os.system('chmod 600 ~/.kaggle/kaggle.json')
        
    #     os.system(f'kaggle datasets download {"/".join(dataset_url.split("/")[-2])}')