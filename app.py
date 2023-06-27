import argparse
import datetime as dt
import warnings
from pathlib import Path

import ffmpeg
import gradio as gr
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
from diff_ttsg.utils.utils import intersperse, plot_tensor
from pymo.preprocessing import MocapParameterizer
from pymo.viz_tools import render_mp4
from pymo.writers import BVHWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Modify the checkpoint locations incase of someother locations
DIFF_TTSG_CHECKPOINT = "diff_ttsg_checkpoint.ckpt"
HIFIGAN_CHECKPOINT = "g_02500000"
MOTION_PIPELINE = "diff_ttsg/resources/data_pipe.expmap_86.1328125fps.sav"
CMU_DICT_PATH = "diff_ttsg/resources/cmu_dictionary"

OUTPUT_FOLDER = "synth_output"

# Model loading tools
def load_model(checkpoint_path):
    model = Diff_TTSG.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    return model

# Vocoder loading tools
def load_vocoder(checkpoint_path):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)['generator'])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan

# Setup text preprocessing
cmu = cmudict.CMUDict(CMU_DICT_PATH)
def process_text(text: str):
    x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).to(device)[None]
    x_lengths = torch.LongTensor([x.shape[-1]]).to(device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    return {
        'x_orig': text,
        'x': x,
        'x_lengths': x_lengths,
        'x_phones': x_phones
    }

# Setup motion visualisation
motion_pipeline = jl.load(MOTION_PIPELINE)
bvh_writer = BVHWriter()
mocap_params = MocapParameterizer("position")



## Load models

model = load_model(DIFF_TTSG_CHECKPOINT)
vocoder = load_vocoder(HIFIGAN_CHECKPOINT)
denoiser = Denoiser(vocoder, mode='zeros')


# Synthesis functions

@torch.inference_mode()
def synthesise(text, mel_timestep, motion_timestep, length_scale, mel_temp, motion_temp):
    
    ## Number of timesteps to run the reverse denoising process
    n_timesteps = {
        'mel': mel_timestep,
        'motion': motion_timestep,
    }

    ## Sampling temperature
    temperature = {
        'mel': mel_temp,
        'motion': motion_temp
    }
    text_processed = process_text(text)
    t = dt.datetime.now()
    output = model.synthesise(
        text_processed['x'], 
        text_processed['x_lengths'],
        n_timesteps=n_timesteps,
        temperature=temperature,
        stoc=False,
        spk=None,
        length_scale=length_scale
    )

    t = (dt.datetime.now() - t).total_seconds()
    print(f'RTF: {t * 22050 / (output["mel"].shape[-1] * 256)}')

    output.update(text_processed) # merge everything to one dict    
    return output

@torch.inference_mode()
def to_waveform(mel, vocoder):
    audio = vocoder(mel).clamp(-1, 1)
    audio = denoiser(audio.squeeze(0)).cpu().squeeze()
    return audio


def to_bvh(motion):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return motion_pipeline.inverse_transform([motion.cpu().squeeze(0).T])
    
    
def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    np.save(folder / f'{filename}', output['mel'].cpu().numpy())
    sf.write(folder / f'{filename}.wav', output['waveform'], 22050, 'PCM_24')
    with open(folder / f'{filename}.bvh', 'w') as f:
        bvh_writer.write(output['bvh'], f)
        
        
def to_stick_video(filename, bvh, folder):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_pos = mocap_params.fit_transform([bvh])
    print(f"rendering {filename} ...")
    render_mp4(X_pos[0], folder / f'{filename}.mp4', axis_scale=200)
    
    
def combine_audio_video(filename: str, folder: str):
    print("Combining audio and video")
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)

    input_video = ffmpeg.input(str(folder / f'{filename}.mp4'))
    input_audio = ffmpeg.input(str(folder / f'{filename}.wav'))
    output_filename = folder / f'{filename}_audio.mp4'
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(str(output_filename)).run(overwrite_output=True)
    print(f"Final output with audio: {output_filename}")


def run(text, output, mel_timestep, motion_timestep, length_scale, mel_temp, motion_temp):
    print("Running synthesis")
    output = synthesise(text, mel_timestep, motion_timestep, length_scale, mel_temp, motion_temp)
    output['waveform'] = to_waveform(output['mel'], vocoder)
    output['bvh'] = to_bvh(output['motion'])[0]
    save_to_folder('temp', output, OUTPUT_FOLDER)
    return (
        output,
        output['x_phones'],
        plot_tensor(output['mel'].squeeze().cpu().numpy()),
        plot_tensor(output['motion'].squeeze().cpu().numpy()),
        str(Path(OUTPUT_FOLDER) / f'temp.wav'),
        gr.update(interactive=True)
    )

def visualize_it(output):
    to_stick_video('temp', output['bvh'], OUTPUT_FOLDER)
    combine_audio_video('temp', OUTPUT_FOLDER)
    return str(Path(OUTPUT_FOLDER) / 'temp_audio.mp4')

def main():
    with gr.Blocks() as demo:

        output = gr.State(value=None)

        with gr.Row():
            gr.Markdown("# Text Input")
        with gr.Row():
            text = gr.Textbox(label="Text Input")

        with gr.Box():    
            with gr.Row():
                gr.Markdown("### Hyper parameters")
            with gr.Row():
                mel_timestep = gr.Slider(label="Number of timesteps (mel)", minimum=0, maximum=1000, step=1, value=50, interactive=True)
                motion_timestep = gr.Slider(label="Number of timesteps (motion)", minimum=0, maximum=1000, step=1, value=500, interactive=True)
                length_scale = gr.Slider(label="Length scale (Speaking rate)", minimum=0.01, maximum=3.0, step=0.05, value=1.15, interactive=True)
                mel_temp = gr.Slider(label="Sampling temperature (mel)", minimum=0.01, maximum=5.0, step=0.05, value=1.3, interactive=True)
                motion_temp = gr.Slider(label="Sampling temperature (motion)", minimum=0.01, maximum=5.0, step=0.05, value=1.5, interactive=True)
        
        synth_btn = gr.Button("Synthesise")

        with gr.Box():    
            with gr.Row():
                gr.Markdown("### Phonetised text")
            with gr.Row():
                phonetised_text = gr.Textbox(label="Phonetised text", interactive=False)
        
        with gr.Box():
            with gr.Row():    
                mel_spectrogram = gr.Image(interactive=False, label="mel spectrogram")
                motion_representation = gr.Image(interactive=False, label="Motion representation")
            
            with gr.Row(): 
                audio = gr.Audio(interactive=False, label="Audio")
        
        with gr.Box():
            with gr.Row():
                gr.Markdown("### Generate stick figure visualisation")
            with gr.Row():
                gr.Markdown("(This will take a while)")
            with gr.Row(): 
                visualize = gr.Button("Visualize", interactive=False)
            
            with gr.Row():
                video = gr.Video(label="Video", interactive=False)
            
        synth_btn.click(
            fn=run,
            inputs=[
                text,
                output,
                mel_timestep,
                motion_timestep,
                length_scale,
                mel_temp,
                motion_temp
            ], 
            outputs=[
                output,
                phonetised_text,
                mel_spectrogram,
                motion_representation, 
                audio,
                # video,
                visualize 
            ], api_name="diff_ttsg")
        
        visualize.click(
            fn=visualize_it,
            inputs=[output],
            outputs=[video],
        )

    demo.queue(1)
    demo.launch()
    
if __name__ == "__main__":
    main()