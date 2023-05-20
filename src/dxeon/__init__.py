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
from os.path import isdir, isfile, join, basename, sep
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
from rich.console import Console
from rich.text import Text
from rich.logging import RichHandler
import tqdm

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


def setup_logging(filename = None, level = 'DEBUG', mode = 'w'):
    handlers = [RichHandler()]
    
    if filename:
        handlers.append(logging.FileHandler(filename, mode = mode))

    logging.basicConfig(
        level = getattr(logging, level.upper()), 
        format = '[%(asctime)s]:[%(levelname)s] - %(message)s',
        datefmt = '%H:%M:%S',
        handlers = handlers,
    )


def debug(*messages):
    logging.debug(' '.join([str(item) for item in messages]))


def info(*messages):
    logging.info(' '.join([str(item) for item in messages]))


def warning(*messages):
    logging.warning(' '.join([str(item) for item in messages]))


def error(*messages):
    logging.error(' '.join([str(item) for item in messages]))


def critical(*messages):
    logging.critical(' '.join([str(item) for item in messages]))


def exception(*messages):
    logging.error(' '.join([str(item) for item in messages]), exc_info = True)


def log_rich_table(rich_table):
  """Generate an ascii formatted presentation of a Rich table
  Eliminates any column styling
  """
  console = Console(width = 150)
  with console.capture() as capture:
      console.print(rich_table)
  return Text.from_ansi(capture.get())