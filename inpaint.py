import numpy as np
from PIL import Image
import argparse 
import torch
from inpaint_pipeline import StableDiffusionInpaintPipeline
import os
from tqdm import tqdm

def infer(
    pipe, 
    prompt, 
    img, 
    mask,
    samples_num, 
    steps_num, 
    scale, 
    strength, 
    generator=None,
    option='Replace selection'
):

    invert_mask = False
    if option == "Replace selection":
      invert_mask = True

    with torch.autocast("cuda"):
        images = pipe(
            [prompt] * samples_num, 
            init_image=img, 
            mask_image=mask, 
            invert_mask=invert_mask, 
            num_inference_steps=steps_num, 
            guidance_scale=scale, 
            strength=strength,
            generator=generator
        )["sample"]

    return images

def read_image_paths(dir_path):
    paths = sorted([os.path.join(dir_path, img) for img in os.listdir(dir_path)])
    return paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir_src')
    parser.add_argument('-dir_out')
    parser.add_argument('-mask_path')
    parser.add_argument('-prompt')
    parser.add_argument('-samples_num', type=int, default=4, help='Number of output images ~[1, 8]')
    parser.add_argument('-steps_num', type=int, default=64, help='~[2, 512]')
    parser.add_argument('-scale', type=float, default=8.0, help='~[0, 30]')
    parser.add_argument('-strength', type=float, default=0.8, help='~[0.0, 1.0]')
    parser.add_argument('-seed', type=int, default=0)
    args = parser.parse_args()

    img_paths  = read_image_paths(args.dir_src)
    masks = np.load(args.mask_path).astype(np.uint8)*255
    img_num = len(img_paths)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16", 
        torch_dtype=torch.float16,
        use_auth_token=True
    ).to("cuda")

    def dummy(images, **kwargs): return images, False
    pipe.safety_checker = dummy
    pipe.enable_attention_slicing()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    
    os.makedirs(args.dir_out, exist_ok=True)
    os.system('cp scripts/inpaint.sh {}'.format(args.dir_out))
    os.system('cp scripts/inpaint.py {}'.format(args.dir_out))
    
    for i in tqdm(range(img_num)):
        img  = Image.open(img_paths[i])
        mask = Image.fromarray(masks[i])
        h_old, w_old = img.size
        h_new, w_new = 512, 512
        img  = img.resize((h_new, w_new), Image.Resampling.LANCZOS)
        mask = mask.resize((h_new, w_new), Image.Resampling.LANCZOS)

        imgs_out = infer(
            pipe=pipe, 
            prompt=args.prompt, 
            img=img, 
            mask=mask,
            samples_num=args.samples_num, 
            steps_num=args.steps_num, 
            scale=args.scale, 
            strength=args.strength, 
            generator=generator
        )[0]

        # imgs_out = [np.array(img) for img in imgs_out]
        # imgs_out = np.concatenate(imgs_out, axis=1)
        # imgs_out = Image.fromarray(imgs_out)
        imgs_out = imgs_out.resize((h_old, w_old), Image.Resampling.LANCZOS)
        path_out = os.path.join(args.dir_out, '{:0>5d}.png'.format(i))
        imgs_out.save(path_out)
        


if __name__ == '__main__':
    main()