python scripts/img2img.py --prompt "a flooding scene" --init-img images/src/0_512.png --strength 0.8 --scale 2 --n_samples 3 \
 --outdir images/img2img-samples/03 \
 --config configs/stable-diffusion/v1-inference.yaml \
 --ckpt   models/ldm/stable-diffusion-v1/model.ckpt
#  --config models/ldm/semantic_synthesis512/config.yaml \
#  --ckpt   models/ldm/semantic_synthesis512/model.ckpt