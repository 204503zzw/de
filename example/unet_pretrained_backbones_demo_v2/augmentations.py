import random

import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from common import pad_image


class TrainAugmentation:
    def __init__(
        self,
        image_size: tuple[int, int],
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate90_prob: float = 0.5,
        shift_prob: float = 0.5,
        max_shift_ratio: float = 0.1,
        scale_prob: float = 0.5,
        min_scale: float = 0.9,
        max_scale: float = 1.1,
        noise_prob: float = 0.3,
        noise_std: float = 8.0,
        brightness_prob: float = 0.3,
        brightness_range: tuple[float, float] = (0.85, 1.15),
        contrast_prob: float = 0.3,
        contrast_range: tuple[float, float] = (0.85, 1.15),
        pad: bool = False,
        pad_align: str = "center",
    ):
        self.height, self.width = image_size
        self.pad = bool(pad)
        self.pad_align = str(pad_align or "center").strip().lower()
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate90_prob = rotate90_prob
        self.shift_prob = shift_prob
        self.max_shift_ratio = max_shift_ratio
        self.scale_prob = scale_prob
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.brightness_prob = brightness_prob
        self.brightness_range = brightness_range
        self.contrast_prob = contrast_prob
        self.contrast_range = contrast_range

    def _resize(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.pad:
            image, _ = pad_image(image, (self.height, self.width), fill=0, align=self.pad_align)
            mask, _ = pad_image(mask, (self.height, self.width), fill=0, align=self.pad_align)
            return image, mask
        image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        mask = mask.resize((self.width, self.height), Image.Resampling.NEAREST)
        return image, mask

    def _apply_affine(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        translate = [0, 0]
        scale = 1.0

        if random.random() < self.shift_prob:
            translate[0] = int(random.uniform(-self.max_shift_ratio, self.max_shift_ratio) * self.width)
            translate[1] = int(random.uniform(-self.max_shift_ratio, self.max_shift_ratio) * self.height)

        if random.random() < self.scale_prob:
            scale = random.uniform(self.min_scale, self.max_scale)

        image = TF.affine(
            image,
            angle=0.0,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        mask = TF.affine(
            mask,
            angle=0.0,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        return image, mask

    def _apply_noise(self, image: Image.Image) -> Image.Image:
        array = np.asarray(image, dtype=np.float32)
        noise = np.random.normal(0.0, self.noise_std, size=array.shape).astype(np.float32)
        array = np.clip(array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(array)

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        image, mask = self._resize(image, mask)

        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < self.vflip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() < self.rotate90_prob:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)

        image, mask = self._apply_affine(image, mask)

        if random.random() < self.brightness_prob:
            factor = random.uniform(*self.brightness_range)
            image = TF.adjust_brightness(image, factor)

        if random.random() < self.contrast_prob:
            factor = random.uniform(*self.contrast_range)
            image = TF.adjust_contrast(image, factor)

        if random.random() < self.noise_prob:
            image = self._apply_noise(image)

        return image, mask


class EvalTransform:
    def __init__(self, image_size: tuple[int, int], pad: bool = False, pad_align: str = "center"):
        self.height, self.width = image_size
        self.pad = bool(pad)
        self.pad_align = str(pad_align or "center").strip().lower()

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.pad:
            image, _ = pad_image(image, (self.height, self.width), fill=0, align=self.pad_align)
            mask, _ = pad_image(mask, (self.height, self.width), fill=0, align=self.pad_align)
            return image, mask
        image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        mask = mask.resize((self.width, self.height), Image.Resampling.NEAREST)
        return image, mask
