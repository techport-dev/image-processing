from basicsr.archs.rrdbnet_arch import RRDBNet
import cv2

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from gfpgan import GFPGANer
import tempfile
import base64
import numpy as np
import io
from PIL import Image
import os


model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
netscale = 4
model_path = os.path.join("weights", "RealESRGAN_x4plus.pth")
upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True)


image = "./images/input/dog.jpg"
scale = 4

img = Image.open(image)
img = np.array(img)
output, _ = upsampler.enhance(img, outscale=scale)

output_image = Image.fromarray(output)
# Save the image
output_image.save('./images/output/scaled_dog.jpg');