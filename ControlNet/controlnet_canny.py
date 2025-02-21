from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2


def create_pipe():
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
    )
    return pipe

def generate( pipe, text_image, prompt):
    canny_image = cv2.Canny(text_image, 100, 200)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)
    negative_prompt = 'low quality, bad quality, sketches'
    controlnet_conditioning_scale = 0.7
    images = pipe(prompt, negative_prompt=negative_prompt, image=canny, controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images
    output = images[0]
    return output