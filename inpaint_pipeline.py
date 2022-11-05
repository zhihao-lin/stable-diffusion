import gradio as gr
import numpy as np
import torch
# import torchcsprng as csprng
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import requests
import PIL
from PIL import Image
from io import BytesIO

from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, PNDMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from tqdm.auto import tqdm

import inspect
from typing import List, Optional, Union

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

# generator = csprng.create_random_device_generator('/dev/urandom')


def preprocess_image(image): #here image is a numpy array
    #image = Image.fromarray(image) #remove if input is PIL
    # w, h = image.size
    # if w > 512:
    #   h = int(h * (512/w))
    #   w = 512
    # if h > 512:
    #   w = int(w*(512/h))
    #   h = 512
    # w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64, 32 can sometimes result in tensor mismatch errors

    # image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    # print('Image size:', image.size)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def preprocess_mask(mask): #here mask is a numpy array
    #mask = Image.fromarray(mask) #remove if input is PIL
    # mask = mask.convert("L")
    w, h = mask.size
    # if w > 512:
    #   h = int(h * (512/w))
    #   w = 512
    # if h > 512:
    #   w = int(w*(512/h))
    #   h = 512
    # w, h = map(lambda x: x - x % 64, (w, h)) 
    w //= 8
    h //= 8
    mask = mask.resize((w, h), resample=PIL.Image.LANCZOS)
    # print('Mask size:', mask.size)
    #mask = mask.resize((64,64), resample=PIL.Image.LANCZOS)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask,(4,1,1))
    mask = mask[None].transpose(0, 1, 2, 3)
    mask[mask < 0.5] = 0.0
    mask[mask >= 0.5] = 1.0
    mask = torch.from_numpy(mask)
    return mask #may need to 1-mask depending on goal of mask selection

class StableDiffusionInpaintPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-guided image inpainting using Stable Diffusion. *This is an experimental feature*.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offsensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `set_attention_slice`
        self.enable_attention_slice(None)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        init_image: Union[torch.FloatTensor, PIL.Image.Image],
        mask_image: Union[torch.FloatTensor, PIL.Image.Image],
        invert_mask: bool = False,
        strength: float = 1.0,
        num_inference_steps: Optional[int] = 64,
        guidance_scale: Optional[float] = 8,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # preprocess image
        init_image = preprocess_image(init_image).to(self.device)

        # encode the init image into latents and scale the latents
        init_latent_dist = self.vae.encode(init_image.to(self.device)).latent_dist
        init_latents = init_latent_dist.sample(generator=generator)

        init_latents = 0.18215 * init_latents

        # Expand init_latents for batch_size
        init_latents = torch.cat([init_latents] * batch_size)
        init_latents_orig = init_latents

        # preprocess mask
        mask = preprocess_mask(mask_image).to(self.device)

        if invert_mask == True:
          mask = 1 - mask

        mask = torch.cat([mask] * batch_size)

        # check sizes
        if not mask.shape == init_latents.shape:
            print(mask.shape)
            print(init_latents.shape)
            raise ValueError("The mask and init_image should be the same size!")

        # get the original timestep using init_timestep
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=self.device)

        # add noise to latents using the timesteps
        noise = torch.randn(init_latents.shape, generator=generator, device=self.device)
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        for i, t in tqdm(enumerate(self.scheduler.timesteps[t_start:])):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # masking
            if t > 1:
              t_noise = torch.randn(latents.shape, generator=generator, device=self.device)
              init_latents_proper = self.scheduler.add_noise(init_latents_orig, t_noise, t-1)
              latents = init_latents_proper * mask + latents * (1-mask)
            else:
              latents = init_latents_orig * mask + latents * (1-mask)

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        has_nsfw_concept = 0

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


def infer(prompt, img, samples_num, steps_num, scale, strength, option):
    invert_mask = False
    if option == "Replace selection":
      invert_mask = True

    mask = img["mask"]
    img = img["image"]


    with autocast("cuda"):
        images = pipe([prompt] * samples_num, init_image=img, mask_image=mask, invert_mask=invert_mask, num_inference_steps=steps_num, guidance_scale=scale, strength=strength)["sample"]
    
    return images


if __name__ == '__main__':
    #inpaint
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16", 
        torch_dtype=torch.float16,
        use_auth_token=True
    ).to("cuda")

    def dummy(images, **kwargs): return images, False
    pipe.safety_checker = dummy

    pipe.enable_attention_slicing()

    block = gr.Blocks(css=".container { max-width: 1200px; margin: auto; }")

    with block as demo:
        gr.Markdown("<h1><center>Stable Diffusion INPAINTING</center></h1>arbitrary resolutions should be 'working'")
        with gr.Group():
            with gr.Box():
                with gr.Row().style(mobile_collapse=False, equal_height=True):

                    text = gr.Textbox(
                        label="Enter your prompt", show_label=False, max_lines=1
                    ).style(
                        border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,
                    )
                    btn = gr.Button("Run").style(
                        margin=False,
                        rounded=(False, True, True, False),
                    )
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                    samples_num = gr.Slider(label="Images", minimum=1, maximum=8, value=4, step=1)
                    steps_num = gr.Slider(label="Generation Steps", minimum=2, maximum=512, value=64, step=1)
                    scale = gr.Slider(label="CFG Scale", minimum=0, maximum=30, value=8.0, step=0.1)
                    strength = gr.Slider(label="Strength", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
            with gr.Row().style(mobile_collapse=False, equal_height=True):
              option = gr.Radio(choices=["Replace selection", "Replace everything else"])

            image = gr.Image(
                tool="sketch",
                label="Input Image",
                type="pil"
            )


            gallery = gr.Gallery(label="Generated images", show_label=False).style(
                grid=[2], height="auto"
            )
            text.submit(infer, inputs=[text,image,samples_num,steps_num,scale,strength,option], outputs=gallery)
            btn.click(infer, inputs=[text,image,samples_num,steps_num,scale,strength,option], outputs=gallery)


    demo.launch(debug=True)

