from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2


def create_pipe(cpu_offload=False):
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
    pipe.enable_model_cpu_offload()
    return pipe

def generate(pipe, image, prompt, num_denoising_steps=30):
    w,h,_ = image.shape
    image = cv2.resize(image, (1024,1024))
    canny_image = cv2.Canny(image, 100, 200)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)
    negative_prompt = 'low quality, bad quality, sketches'
    controlnet_conditioning_scale = 0.7
    images = pipe(prompt, negative_prompt=negative_prompt, image=canny_image, controlnet_conditioning_scale=controlnet_conditioning_scale, num_inference_steps=num_denoising_steps
    ).images
    output = images[0]
    generated_img = cv2.resize(np.asarray(output), (h,w))
    canny_image = cv2.resize(np.asarray(canny_image), (h,w))
    return canny_image, generated_img