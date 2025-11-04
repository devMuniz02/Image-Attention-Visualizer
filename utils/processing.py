import torchvision.transforms as T
import fsspec
import io
from PIL import Image

def image_transform(img_size=512):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def open_binary(path: str):
    """
    Open any (local or gs://) file for binary reading.
    Returns a file-like object (context manager).
    """
    return fsspec.open(path, mode="rb").open()

def pil_from_path(path: str) -> Image.Image:
    """
    Load an image from local or GCS; returns a PIL image in RGB.
    """
    with open_binary(path) as f:
        img_bytes = f.read()
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return im