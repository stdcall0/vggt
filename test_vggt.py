import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import numpy as np
import os
from PIL import Image
import torchvision.transforms as T
import time

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# save_dir = "output_images"
# os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
model.eval()
model = model.to(device)

start_time = time.time()
# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
# model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (replace with your own image paths)
# image_names = ["./examples/kitchen/images/00.png", "./examples/kitchen/images/01.png", 
#         "./examples/kitchen/images/02.png", "./examples/kitchen/images/03.png",
#         "./examples/kitchen/images/04.png", "./examples/kitchen/images/05.png",
#         "./examples/kitchen/images/06.png", "./examples/kitchen/images/07.png",
#         "./examples/kitchen/images/08.png", "./examples/kitchen/images/09.png",
#         "./examples/kitchen/images/10.png"]

# image_names = ["./KITTI-360-sample/0000000000.png","./KITTI-360-sample/0000000001.png",
#               "./KITTI-360-sample/0000000002.png","./KITTI-360-sample/0000000003.png",
#               "./KITTI-360-sample/0000000004.png","./KITTI-360-sample/0000000005.png",
#               "./KITTI-360-sample/0000000006.png","./KITTI-360-sample/0000000007.png",
#               "./KITTI-360-sample/0000000008.png","./KITTI-360-sample/0000000009.png",]

# image_names = ["./KITTI-360-sample/0000000000.png","./KITTI-360-sample/0000000001.png",
#               "./KITTI-360-sample/0000000002.png","./KITTI-360-sample/0000000003.png",
#               "./KITTI-360-sample/0000000004.png","./KITTI-360-sample/0000000005.png",
#               "./KITTI-360-sample/0000000006.png","./KITTI-360-sample/0000000007.png",
#               "./KITTI-360-sample/0000000008.png","./KITTI-360-sample/0000000009.png",
#               "./KITTI-360-sample/0000000010.png","./KITTI-360-sample/0000000011.png",
#               "./KITTI-360-sample/0000000012.png","./KITTI-360-sample/0000000013.png",
#               "./KITTI-360-sample/0000000014.png","./KITTI-360-sample/0000000015.png",
#               "./KITTI-360-sample/0000000016.png","./KITTI-360-sample/0000000017.png",
#               "./KITTI-360-sample/0000000018.png","./KITTI-360-sample/0000000019.png",]

image_names = ["./KITTI-360-sample/0000000000.png","./KITTI-360-sample/0000000001.png",
              "./KITTI-360-sample/0000000002.png","./KITTI-360-sample/0000000003.png",
              "./KITTI-360-sample/0000000004.png","./KITTI-360-sample/0000000005.png",
              "./KITTI-360-sample/0000000006.png","./KITTI-360-sample/0000000007.png",
              "./KITTI-360-sample/0000000008.png","./KITTI-360-sample/0000000009.png",
              "./KITTI-360-sample/0000000010.png","./KITTI-360-sample/0000000011.png",
              "./KITTI-360-sample/0000000012.png","./KITTI-360-sample/0000000013.png",
              "./KITTI-360-sample/0000000014.png","./KITTI-360-sample/0000000015.png",
              "./KITTI-360-sample/0000000016.png","./KITTI-360-sample/0000000017.png",
              "./KITTI-360-sample/0000000018.png","./KITTI-360-sample/0000000019.png",
              "./KITTI-360-sample/0000000020.png","./KITTI-360-sample/0000000021.png",
              "./KITTI-360-sample/0000000022.png","./KITTI-360-sample/0000000023.png",
              "./KITTI-360-sample/0000000024.png","./KITTI-360-sample/0000000025.png",
              "./KITTI-360-sample/0000000026.png","./KITTI-360-sample/0000000027.png",
              "./KITTI-360-sample/0000000028.png","./KITTI-360-sample/0000000029.png",]


images = load_and_preprocess_images(image_names).to(device)
torch.cuda.reset_peak_memory_stats()
mem_before = torch.cuda.memory_allocated() / 1024**2

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)
        
end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")

mem_after = torch.cuda.memory_allocated() / 1024**2
mem_peak = torch.cuda.max_memory_allocated() / 1024**2
print(f"Memory (MB): before {mem_before:.1f}, after {mem_after:.1f}, peak {mem_peak:.1f}")


print("Converting pose encoding to extrinsic and intrinsic matrices...")
