python inpaint.py -prompt "flooding" -seed 1 -samples_num 1 \
-dir_src   ../weather_nvs/results/tnt/family-flood-mask/frames \
-mask_path ../weather_nvs/results/tnt/family-flood-mask/mask.npy \
-dir_out   ../weather_nvs/output/paper/tnt/Family/diffusion/frames

python inpaint.py -prompt "flooding" -seed 1 -samples_num 1 \
-dir_src   ../weather_nvs/results/tnt/horse-flood-mask/frames \
-mask_path ../weather_nvs/results/tnt/horse-flood-mask/mask.npy \
-dir_out   ../weather_nvs/output/paper/tnt/Horse/diffusion/frames

python inpaint.py -prompt "flooding" -seed 1 -samples_num 1 \
-dir_src   ../weather_nvs/results/tnt/playground-flood-mask/frames \
-mask_path ../weather_nvs/results/tnt/playground-flood-mask/mask.npy \
-dir_out   ../weather_nvs/output/paper/tnt/Playground/diffusion/frames

python inpaint.py -prompt "flooding" -seed 1 -samples_num 1 \
-dir_src   ../weather_nvs/results/tnt/train-flood-mask/frames \
-mask_path ../weather_nvs/results/tnt/train-flood-mask/mask.npy \
-dir_out   ../weather_nvs/output/paper/tnt/Train/diffusion/frames

python inpaint.py -prompt "flooding" -seed 1 -samples_num 1 \
-dir_src   ../weather_nvs/results/tnt/truck-flood-mask/frames \
-mask_path ../weather_nvs/results/tnt/truck-flood-mask/mask.npy \
-dir_out   ../weather_nvs/output/paper/tnt/Truck/diffusion/frames

python inpaint.py -prompt "flooding" -seed 1 -samples_num 1 \
-dir_src   ../weather_nvs/results/colmap/garden-flood-mask/frames \
-mask_path ../weather_nvs/results/colmap/garden-flood-mask/mask.npy \
-dir_out   ../weather_nvs/output/paper/colmap/garden/diffusion/frames