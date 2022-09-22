import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

prompt = "A corgi is dancing on a ball"
with autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5)["sample"][0]  
    
image.save("images/21.png")