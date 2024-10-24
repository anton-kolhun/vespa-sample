import json
import os
import random
import string

import requests
import torch
from PIL import Image
from numpyencoder import NumpyEncoder
from torchvision.transforms import Resize, Compose, CenterCrop, Normalize, ToTensor, InterpolationMode

device = "cuda" if torch.cuda.is_available() else "cpu"

def image_to_rgb(image):
    return image.convert("RGB")


transformer = Compose([
    Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
    CenterCrop(size=(224, 224)),
    image_to_rgb,
    ToTensor(),
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
])


class ClipVisualConverter:
    @staticmethod
    def image_to_tensor(image_url):
        img_data = requests.get(image_url).content
        temp_file_name = str(''.join(random.choices(string.ascii_letters, k=5)))
        with open(f"{temp_file_name}", 'wb') as handler:
            handler.write(img_data)
            tensor = transformer(Image.open(f"{temp_file_name}")).unsqueeze(0).to(device)
            os.remove(handler.name)

        return json.dumps(tensor.numpy(), cls=NumpyEncoder)
