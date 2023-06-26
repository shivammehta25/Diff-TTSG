import datetime as dt
import warnings
from pathlib import Path

import ffmpeg
import IPython.display as ipd
import joblib as jl
import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

from diff_ttsg.hifigan.config import v1
from diff_ttsg.hifigan.denoiser import Denoiser
from diff_ttsg.hifigan.env import AttrDict
from diff_ttsg.hifigan.models import Generator as HiFiGAN
from diff_ttsg.models.diff_ttsg import Diff_TTSG
from diff_ttsg.text import cmudict, sequence_to_text, text_to_sequence
from diff_ttsg.text.symbols import symbols
from diff_ttsg.utils.model import denormalize
from diff_ttsg.utils.utils import intersperse
from pymo.preprocessing import MocapParameterizer
from pymo.viz_tools import render_mp4
from pymo.writers import BVHWriter
