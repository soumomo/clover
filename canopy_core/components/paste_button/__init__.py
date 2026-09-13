from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from PIL import Image
import io
import base64
import streamlit.components.v1 as components

frontend_dir = (Path(__file__).parent / "frontend").absolute()
_component_func = components.declare_component(
    "clover_paste_button", path=str(frontend_dir)
)

@dataclass
class PasteResult:
    image_data: Optional[Image.Image] = None

def _data_url_to_image(data_url: str) -> Image.Image:
    _, encoded = data_url.split(";base64,")
    return Image.open(io.BytesIO(base64.b64decode(encoded)))

def paste_image_button(
    label: str = "Paste Screenshot",
    text_color: str = "#FFFFFF",
    background_color: str = "#111827",
    hover_background_color: str = "#1F2937",
    key: Optional[str] = None,
) -> PasteResult:
    """Native bundled paste button component with zero ghost text."""
    component_value = _component_func(
        label=label,
        text_color=text_color,
        background_color=background_color,
        hover_background_color=hover_background_color,
        key=key,
        default=None,
    )

    if component_value is None:
        return PasteResult(image_data=None)
    if str(component_value).startswith("error"):
        return PasteResult(image_data=None)
    try:
        img = _data_url_to_image(component_value)
        return PasteResult(image_data=img)
    except Exception:
        return PasteResult(image_data=None)
