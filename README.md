# Diff-TTSG: Denoising probabilistic integrated speech and gesture synthesis

#### [Shivam Mehta][shivam_profile], [Siyang Wang][siyang_profile], [Simon Alexanderson][simon_profile], [Jonas Beskow][jonas_profile], [Éva Székely][eva_profile], and [Gustav Eje Henter][gustav_profile]

This is the official code repository of [Diff-TTSG: Denoising probabilistic integrated speech and gesture synthesis][arxiv_link].

Demo Page: [https://shivammehta25.github.io/Diff-TTSG/][this_page]

Huggingface Space: [https://huggingface.co/spaces/shivammehta25/Diff-TTSG][huggingface_space]

We present Diff-TTSG, the first diffusion model that jointly learns to synthesise speech and gestures together. Our method is probabilistic and non-autoregressive, and can be trained on small datasets from scratch. In addition, to showcase the efficacy of these systems and pave the way for their evaluation, we describe a set of careful uni- and multi-modal subjective tests for evaluating integrated speech and gesture synthesis systems.

[shivam_profile]: https://www.kth.se/profile/smehta
[siyang_profile]: https://www.kth.se/profile/siyangw
[simon_profile]: https://www.kth.se/profile/simonal
[jonas_profile]: https://www.kth.se/profile/beskow
[eva_profile]: https://www.kth.se/profile/szekely
[gustav_profile]: https://people.kth.se/~ghe/
[this_page]: https://shivammehta25.github.io/Diff-TTSG/
[arxiv_link]: https://arxiv.org/abs/2306.09417
[huggingface_space]: https://huggingface.co/spaces/shivammehta25/Diff-TTSG

## Teaser

[![Watch the video](https://img.youtube.com/vi/xYxcqyMJjsE/maxresdefault.jpg)](https://www.youtube.com/watch?v=xYxcqyMJjsE)

# Installation

1. Clone this repository

   ```bash
   git clone https://github.com/shivammehta25/Diff-TTSG.git
   cd Diff-TTSG
   ```

2. Create a new environment (optional)

   ```bash
   conda create -n diff-ttsg python=3.10 -y
   conda activate diff-ttsg
   ```

3. Setup diff ttsg (This will install all the dependencies and download the pretrained models)

   - Is you are using Linux or Mac OS, run the following command

   ```bash
   make install
   ```

   - else install all dependencies and alignment build simply by

   ```bash
   pip install -e .
   ```

4. Run gradio UI

   ```bash
   gradio app.py
   ```

or use `synthesis.ipynb`

[Pretrained checkpoint](https://github.com/shivammehta25/Diff-TTSG/releases/download/checkpoint/diff_ttsg_checkpoint.ckpt) (Should be autodownloaded by running either `make install` or `gradio app.py`)

## Citation information

If you use or build on our method or code for your research, please cite our paper:

```
@article{mehta2023diffttsg,
      title={Diff-TTSG: Denoising probabilistic integrated speech and gesture synthesis},
      author={Shivam Mehta and Siyang Wang and Simon Alexanderson and Jonas Beskow and Éva Székely and Gustav Eje Henter},
      journal={arXiv preprint arXiv:2306.09417},
      year={2023},
}
```

### Acknowledgement

The code in the repository is heavily inspired by the source code of

- [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)
- [Diffusers](https://github.com/huggingface/diffusers)
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
