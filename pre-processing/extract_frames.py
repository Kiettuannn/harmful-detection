import cv2
import os
import numpy as np
from tqdm import tqdm
import logging
from decord import VideoReader, cpu

# Setup logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

