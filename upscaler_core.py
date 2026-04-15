from __future__ import annotations

import datetime as dt
from pathlib import Path
import ssl
import sys
from typing import Any

import certifi

# PyInstaller-packaged macOS apps cannot locate the system CA store, so urllib
# SSL handshakes fail.  Patch the default HTTPS context factory to use certifi's
# bundled CA certificates before any network call is made.
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import cv2
import numpy as np
from torchvision.transforms import _functional_tensor

sys.modules.setdefault("torchvision.transforms.functional_tensor", _functional_tensor)

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
OUTPUT_DIR = ROOT / "outputs"


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "RealESRGAN x4 Plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "netscale": 4,
        "model_factory": lambda: RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        ),
        "default_denoise": 0.5,
        "description": "El modelo más equilibrado para fotos generales.",
    },
    "RealESRNet x4 Plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        "netscale": 4,
        "model_factory": lambda: RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        ),
        "default_denoise": 0.5,
        "description": "Más conservador; suele retener mejor texturas limpias.",
    },
    "RealESRGAN x4 Plus Anime": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x4plus_anime_6B.pth",
        "netscale": 4,
        "model_factory": lambda: RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6,
            num_grow_ch=32,
            scale=4,
        ),
        "default_denoise": 0.5,
        "description": "Optimizado para ilustración, anime y arte digital.",
    },
    "realesr-general-x4v3": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "netscale": 4,
        "model_factory": lambda: SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        ),
        "default_denoise": 0.35,
        "description": "Rápido y muy bueno en fotos pequeñas o comprimidas.",
    },
}


def ensure_dirs() -> None:
    WEIGHTS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def get_model_description(model_name: str) -> str:
    return MODEL_CONFIGS[model_name]["description"]


def get_default_denoise(model_name: str) -> float:
    return float(MODEL_CONFIGS[model_name]["default_denoise"])


def model_supports_denoise_mix(model_name: str) -> bool:
    return model_name == "realesr-general-x4v3"


def get_model(model_name: str, denoise_strength: float) -> RealESRGANer:
    config = MODEL_CONFIGS[model_name]
    weight_path = Path(
        load_file_from_url(
            url=config["url"],
            model_dir=str(WEIGHTS_DIR),
            progress=True,
            file_name=Path(config["url"]).name,
        )
    )

    dni_weight = None
    if model_supports_denoise_mix(model_name) and denoise_strength < 1:
        wdn_path = Path(
            load_file_from_url(
                url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                model_dir=str(WEIGHTS_DIR),
                progress=True,
                file_name="realesr-general-wdn-x4v3.pth",
            )
        )
        dni_weight = [denoise_strength, 1 - denoise_strength]
        model_path: str | list[str] = [str(weight_path), str(wdn_path)]
    else:
        model_path = str(weight_path)

    return RealESRGANer(
        scale=config["netscale"],
        model_path=model_path,
        dni_weight=dni_weight,
        model=config["model_factory"](),
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=None,
    )


def save_output_image(image: np.ndarray, model_name: str) -> Path:
    ensure_dirs()
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = model_name.lower().replace(" ", "-")
    output_path = OUTPUT_DIR / f"upscaled-{slug}-{timestamp}.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return output_path


def upscale_array(
    image: np.ndarray,
    model_name: str,
    output_scale: float,
    denoise_strength: float,
) -> tuple[np.ndarray, str, Path]:
    if image is None:
        raise ValueError("No hay ninguna imagen cargada.")

    ensure_dirs()
    upsampler = get_model(model_name, denoise_strength)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    restored_bgr, _ = upsampler.enhance(bgr_image, outscale=output_scale)
    restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
    output_path = save_output_image(restored_rgb, model_name)

    details = [
        f"Modelo: {model_name}",
        f"Escala de salida: x{output_scale:.1f}",
        f"Resolución final: {restored_rgb.shape[1]} x {restored_rgb.shape[0]} px",
        f"Archivo: {output_path.name}",
    ]
    if model_supports_denoise_mix(model_name):
        details.append(f"Noise reduction mix: {denoise_strength:.2f}")

    return restored_rgb, "\n".join(details), output_path
