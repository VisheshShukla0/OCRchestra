import os
import random
import string
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from numpy.random import default_rng

from GlyphEngine.glyph_renderer import get_OCR_data

word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
response = requests.get(word_site)
WORDS = response.content.splitlines()
word = random.choice(WORDS)
if random.randint(0,1)==0: word = word.upper()
text = word.decode()
text = "HELLO!"

font_files_root = 'all_fonts/English/'
save_root = 'OCR_Data/'

TEXTS = ["MEDICAL","Total","PHONE","please","YOLO!","Where?","'QUOTE'"]
# TEXTS = ["भूगोल","सड़क","लगभग"]
#TEXTS = ["చమక","బహ","అంధ"]

rng = default_rng()
colormap = [rng.choice(255, size=128, replace=True), rng.choice(255, size=128, replace=True), rng.choice(255, size=128, replace=True)]
    
for idx in range(len(TEXTS)):
    text = TEXTS[idx]
    subdir = random.choice(os.listdir(font_files_root))
    font_files = os.listdir(os.path.join(font_files_root,subdir))
    font_file_path = f"{font_files_root}/{subdir}/{random.choice(font_files)}"

    img, img_seg, img_rect, img_annot = get_OCR_data(content=text, canvas_resolution=(2048,2048), text_area=(1024,1024), font_file_path = font_file_path, colormap=colormap)
    os.makedirs(f"{save_root}/{idx}/", exist_ok=True)
    cv2.imwrite(f"{save_root}/{idx}/text.jpg", img)
    cv2.imwrite(f"{save_root}/{idx}/seg.jpg", img_seg)
    cv2.imwrite(f"{save_root}/{idx}/boxes.jpg", img_rect)
    cv2.imwrite(f"{save_root}/{idx}/annot.jpg", img_annot)
