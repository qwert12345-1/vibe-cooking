---
license: openrail++
tags:
- text-to-image
- stable-diffusion
library_name: diffusers
inference: false
---

# SDXS-512-DreamShaper

SDXS is a model that can generate high-resolution images in real-time based on prompt texts, trained using score distillation and feature matching. 
For more information, please refer to our research paper: [SDXS: Real-Time One-Step Latent Diffusion Models with Image Conditions](https://arxiv.org/abs/2403.16627). 
We open-source the model as part of the research.

SDXS-512-DreamShaper is the version we trained specifically for community. 
The model is trained without focusing on FID, and sacrifices diversity for better image generation quality. 
In order to avoid some possible risks, the SDXS-512-1.0 and SDXS-1024-1.0 will not be available shortly. 
Watch [our repo](https://github.com/IDKiro/sdxs) for any updates.

Model Information:
- Teacher DM: [dreamshaper-8-lcm](https://huggingface.co/Lykon/dreamshaper-8-lcm)
- Offline DM: [dreamshaper-8](https://huggingface.co/Lykon/dreamshaper-8)
- VAE: [TAESD](https://huggingface.co/madebyollin/taesd)

Similar to SDXS-512-0.9, since our image decoder is not compatible with diffusers, we use TAESD. 
Currently, our pull request has been merged in to reduce the gap between TAESD and our image decoder. 
In the next diffusers release update, we may replace the image decoder.

## Diffusers Usage

![](output.png)

```python
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL

repo = "IDKiro/sdxs-512-dreamshaper"
seed = 42
weight_type = torch.float16     # or float32

# Load model.
pipe = StableDiffusionPipeline.from_pretrained(repo, torch_dtype=weight_type)
pipe.to("cuda")

prompt = "a close-up picture of an old man standing in the rain"

# Ensure using 1 inference step and CFG set to 0.
image = pipe(
    prompt, 
    num_inference_steps=1, 
    guidance_scale=0,
    generator=torch.Generator(device="cuda").manual_seed(seed)
).images[0]

image.save("output.png")
```

## Cite Our Work

```
@article{song2024sdxs,
  author    = {Yuda Song, Zehao Sun, Xuanwu Yin},
  title     = {SDXS: Real-Time One-Step Latent Diffusion Models with Image Conditions},
  journal   = {arxiv},
  year      = {2024},
}
```
