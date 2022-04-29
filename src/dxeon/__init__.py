from . import interpretability
from . import io
from . import losses
from . import modules
from . import stats
from . import utils
from . import viz
from . import warmups

from glob import glob
import os
from os import mkdir as os_mkdir, remove, listdir
from os.path import isdir, isfile, join, basename
from sys import exit
import matplotlib.pyplot as plt
from shutil import rmtree as rmdir, move, copy
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split, WeightedRandomSampler
import torch.nn.functional as F
from torchvision import models, datasets, transforms as T
import torchvision.transforms.functional as TF
from torchvision.datasets import DatasetFolder
import zipfile
import logging

# plt.style.use('ggplot')
plt.style.use('seaborn')

def mkdir_if_not_exists(path):
    if not isdir(path):
        os_mkdir(path)

def mkdir(path):
    if isdir(path):
        rmdir(path)
    os_mkdir(path)

def rename(path, new_path):
    if isdir(path):
        os.rename(path, new_path)

def extract_zip(path, output_dir = './'):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)


def setup_logging(filename = None, level = 'DEBUG'):
    handlers = [logging.StreamHandler()]
    
    if filename:
        handlers.append(logging.FileHandler(filename))

    logging.basicConfig(
        level = getattr(logging, level.upper()), 
        format = '[%(asctime)s]:[%(levelname)s] - %(message)s',
        datefmt = '%H:%M:%S',
        handlers = handlers,
    )

def debug(message):
    logging.debug(message)

def info(message):
    logging.info(message)

def warning(message):
    logging.warning(message)

def error(message):
    logging.error(message)

def critical(message):
    logging.critical(message)

def exception(message):
    logging.error(message, exc_info = True)