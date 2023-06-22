import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio as ta
from einops import pack
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from diff_ttsg.text import cmudict, text_to_sequence
from diff_ttsg.text.symbols import symbols
from diff_ttsg.utils.audio import mel_spectrogram
from diff_ttsg.utils.model import fix_len_compatibility, normalize
from diff_ttsg.utils.utils import intersperse, parse_filelist


class CormacDataModule(LightningDataModule):

    def __init__(
        self,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        pin_memory,
        cmudict_path,
        motion_folder,
        add_blank,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        motion_pipeline_filename,
        seed
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        
        self.trainset = TextMelDataset(
            self.hparams.train_filelist_path,
            self.hparams.cmudict_path,
            self.hparams.motion_folder,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length, 
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed
        )
        self.validset = TextMelDataset(
            self.hparams.valid_filelist_path,
            self.hparams.cmudict_path,
            self.hparams.motion_folder,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed
        )
        

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelBatchCollate()
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate()
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path, motion_folder, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000, data_parameters=None, seed=None):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.motion_fileloc = Path(motion_folder)        
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        if data_parameters is not None:
            self.data_parameters = data_parameters
        else:
            self.data_parameters = { 'mel_mean': 0, 'mel_std': 1, 'motion_mean': 0, 'motion_std': 1 }
        random.seed(seed)
        random.shuffle(self.filepaths_and_text)

    def get_pair(self, filepath_and_text):
        filepath, text = filepath_and_text[0], filepath_and_text[1]
        text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)
        motion = self.get_motion(filepath, mel.shape[1])
        return (text, mel, motion)
    
    def get_motion(self, filename, mel_shape, ext=".expmap_86.1328125fps.pkl"):
        file_loc = self.motion_fileloc / Path(Path(filename).name).with_suffix(ext)
        motion = torch.from_numpy(pd.read_pickle(file_loc).to_numpy())
        motion = F.interpolate(motion.T.unsqueeze(0), mel_shape).squeeze(0)
        motion = normalize(motion, self.data_parameters['motion_mean'], self.data_parameters['motion_std'])
        return motion 

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, 80, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        mel = normalize(mel, self.data_parameters['mel_mean'], self.data_parameters['mel_std'])
        return mel

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        text, mel, motion = self.get_pair(self.filepaths_and_text[index])
        item = {'y': mel, 'x': text, 'y_motion': motion}
        return item

    def __len__(self):
        return len(self.filepaths_and_text)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]
        n_motion = batch[0]['y_motion'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_motion = torch.zeros((B, n_motion, y_max_length), dtype=torch.float32)
        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, x_, y_motion_ = item['y'], item['x'], item['y_motion']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            y_motion[i, :, :y_motion_.shape[-1]] = y_motion_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths, 'y_motion': y_motion}