"""
Clover · CanopyMRV Engine
Autonomous High-Resolution Optical Tree Crown Delineation & Carbon Accounting System.
Clean minimalist design language:
Pure White Canvas (#FFFFFF), Deep Ink Charcoal (#111827), Emerald Clover Green (#16A34A),
Forest Accents (#15803D), Hairline Stone Borders (#E5E7EB), and Satellite Basemap.
"""

from __future__ import annotations

import base64
import io
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import folium
from folium import plugins
from branca.element import MacroElement
from jinja2 import Template
import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
import rasterio
from rasterio.warp import transform_bounds
import shapely
from shapely.geometry import Polygon, mapping
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import folium_static
from canopy_core.components.paste_button import paste_image_button

import torch

from canopy_core.ingest import inspect_raster, parse_aoi, clip_raster_to_aoi, RasterMetadata
from canopy_core.models.detector import TreeDetector
from canopy_core.models.area_engine import CanopyAreaEngine, CanopyMetrics
from canopy_core.filters.weed_filter import WeedWaterFilter
from canopy_core.tiler import SlidingWindowTiler

def get_hardware_label() -> str:
    """Return dynamic human-readable hardware compute accelerator name."""
    try:
        if torch.cuda.is_available():
            return f"NVIDIA {torch.cuda.get_device_name(0)} GPU"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple Silicon GPU (MPS)"
        else:
            return "Multi-Core CPU"
    except Exception:
        return "Accelerated Compute"

# -----------------------------------------------------------------------------
# 1. Page Configuration & Clover Design System
# -----------------------------------------------------------------------------

CLOVER_LOGO_PATH = Path(__file__).parent / "assets" / "clover_logo_trans.png"
if CLOVER_LOGO_PATH.exists():
    with open(CLOVER_LOGO_PATH, "rb") as f:
        CLOVER_LOGO_B64 = base64.b64encode(f.read()).decode("utf-8")
else:
    CLOVER_LOGO_B64 = ""

CLOVER_FAVICON_PATH = Path(__file__).parent / "assets" / "clover_favicon.png"

st.set_page_config(
    page_title="Clover",
    page_icon=Image.open(CLOVER_FAVICON_PATH) if CLOVER_FAVICON_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clover Custom CSS: Pure White (#FFFFFF), Deep Charcoal (#111827), Clover Green (#16A34A)
FLORA_WHITE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Force pure white background on main view and all parent containers */
html, body, [class*="css"], .main, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #111827 !important;
    background-color: #FFFFFF !important;
}

/* Transparent Streamlit header keeping sidebar expand controls fully visible */
header[data-testid="stHeader"] {
    background-color: transparent !important;
    height: 3.2rem !important;
    z-index: 99 !important;
}

/* Ensure stToolbar is visible for the expand sidebar chevron */
[data-testid="stToolbar"] {
    display: flex !important;
    visibility: visible !important;
    background: transparent !important;
}

/* Hide only unnecessary Streamlit menu and deploy items, NOT the toolbar or sidebar buttons */
#MainMenu, 
.stDeployButton, 
footer, 
[data-testid="stToolbar"] > div > div:last-child,
[data-testid="stHeaderActionElements"],
[data-testid="manage-app-button"] {
    display: none !important;
    visibility: hidden !important;
}

/* High-contrast accessible sidebar expand button when collapsed (Streamlit 1.51+ and legacy) */
button[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] button,
[data-testid="stExpandSidebarButton"],
button[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: #111827 !important;
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08) !important;
    margin-left: 0.5rem !important;
    margin-top: 0.4rem !important;
    padding: 6px !important;
    z-index: 1000 !important;
    cursor: pointer !important;
}
button[data-testid="stExpandSidebarButton"]:hover,
button[data-testid="stSidebarCollapsedControl"]:hover {
    background-color: #F9FAFB !important;
    border-color: #16A34A !important;
}
button[data-testid="stExpandSidebarButton"] svg,
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stSidebarCollapsedControl"] svg {
    fill: #111827 !important;
    stroke: #111827 !important;
    color: #111827 !important;
}

/* Sidebar collapse chevron inside sidebar */
[data-testid="stSidebarCollapseButton"] button,
button[data-testid="baseButton-header"] {
    color: #111827 !important;
}
[data-testid="stSidebarCollapseButton"] svg,
button[data-testid="baseButton-header"] svg {
    fill: #111827 !important;
}

/* Container Spacing */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* Sidebar styling: Clean, light off-white with fine border */
section[data-testid="stSidebar"] {
    background-color: #FAFAFA !important;
    border-right: 1px solid #E5E7EB !important;
}

/* Enforce explicit width ONLY when expanded! */
section[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 320px !important;
}

/* When collapsed, allow it to fully collapse to 0 width with no protruding sliver */
section[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 0px !important;
    width: 0px !important;
    display: none !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
}

/* Enforce dark high-contrast text on all typography */
h1, h2, h3, h4, h5, h6 {
    color: #111827 !important;
}
p, span, li, label, b, strong {
    color: #1F2937 !important;
}

/* Streamlit Tabs: Clear, high-contrast dark charcoal text */
button[data-baseweb="tab"] {
    background-color: transparent !important;
}
button[data-baseweb="tab"] p, button[data-baseweb="tab"] div, button[data-baseweb="tab"] span {
    color: #374151 !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
}
button[data-baseweb="tab"][aria-selected="true"] p, button[data-baseweb="tab"][aria-selected="true"] div, button[data-baseweb="tab"][aria-selected="true"] span {
    color: #165A4C !important;
    font-weight: 700 !important;
}

/* Sidebar Radio Buttons */
div[data-testid="stRadio"] label p, div[data-testid="stRadio"] label span, div[data-testid="stRadio"] label div {
    color: #111827 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

/* Sidebar Sliders */
div[data-testid="stSlider"] label p, div[data-testid="stSlider"] span, div[data-testid="stSlider"] div {
    color: #111827 !important;
    font-weight: 500 !important;
}

/* Sidebar Checkboxes */
div[data-testid="stCheckbox"] label p, div[data-testid="stCheckbox"] label span {
    color: #111827 !important;
    font-weight: 500 !important;
}

/* Selectbox */
div[data-testid="stSelectbox"] label p {
    color: #111827 !important;
    font-weight: 600 !important;
}
div[data-baseweb="select"] * {
    color: #111827 !important;
    background-color: #FFFFFF !important;
}

/* Module Badges */
.flora-module-badge, .clover-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #166534 !important;
    background-color: #F0FDF4;
    border: 1px solid #BBF7D0;
    padding: 3px 8px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 8px;
}

iframe[title*="paste_button"], iframe[title*="clover_paste_button"] {
    width: 100% !important;
    border: none !important;
    overflow: hidden !important;
}

iframe[title="streamlit.components.v1.html"] {
    display: none !important;
    height: 0px !important;
    width: 0px !important;
    border: none !important;
}

/* Metric Cards: Clean uniform boxes with corner info hover tooltip */
.flora-metric-card {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    height: 145px !important;
    min-height: 145px !important;
    max-height: 145px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    box-sizing: border-box !important;
    position: relative !important;
    overflow: visible !important;
}
.flora-metric-top {
    display: flex !important;
    align-items: flex-start !important;
    justify-content: space-between !important;
    width: 100% !important;
    height: 32px !important;
    min-height: 32px !important;
    max-height: 32px !important;
    position: relative !important;
}
.flora-metric-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #4B5563 !important;
    font-weight: 600;
    line-height: 1.3;
    display: flex;
    align-items: flex-start;
    gap: 5px;
    padding-right: 8px;
}
.flora-tooltip-wrap {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.flora-info-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background-color: #F3F4F6;
    color: #6B7280;
    border: 1px solid #E5E7EB;
    cursor: pointer;
    transition: all 0.15s ease;
}
.flora-info-btn:hover {
    background-color: #ECFDF5;
    border-color: #A7F3D0;
    color: #16A34A;
}
.flora-info-btn:hover svg {
    stroke: #16A34A !important;
}
.flora-tooltip-card {
    visibility: hidden;
    opacity: 0;
    width: 250px;
    background-color: #FFFFFF;
    color: #1F2937;
    text-align: left;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 12px 14px;
    position: absolute;
    z-index: 999999 !important;
    top: 26px;
    right: 0;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.12), 0 8px 10px -6px rgba(0, 0, 0, 0.06);
    font-size: 0.76rem;
    line-height: 1.45;
    font-family: 'Inter', sans-serif;
    pointer-events: none;
    transition: opacity 0.2s ease, transform 0.2s ease;
    transform: translateY(4px);
}
.flora-tooltip-wrap:hover .flora-tooltip-card {
    visibility: visible;
    opacity: 1;
    pointer-events: auto;
    transform: translateY(0);
}
.flora-metric-val {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 2.2rem;
    font-weight: 600;
    color: #111827 !important;
    margin: 0;
    line-height: 1.05;
}
.flora-metric-sub {
    font-size: 0.81rem;
    color: #374151 !important;
    line-height: 1.35;
}

/* Scientific Disclosure Box */
.flora-disclosure-box {
    background-color: #FAFAFA;
    border-left: 3px solid #165A4C;
    border-top: 1px solid #E5E7EB;
    border-right: 1px solid #E5E7EB;
    border-bottom: 1px solid #E5E7EB;
    padding: 16px 20px;
    border-radius: 0 6px 6px 0;
    margin: 14px 0;
}

/* Sidebar & General Action Buttons: Clean high-contrast white card buttons */
.stButton>button {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 6px !important;
    padding: 10px 14px !important;
    font-weight: 500 !important;
    font-size: 0.86rem !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    transition: all 0.15s ease !important;
}
.stButton>button:hover {
    background-color: #F9FAFB !important;
    border-color: #16A34A !important;
    box-shadow: 0 2px 4px rgba(22, 163, 74, 0.08) !important;
}
.stButton>button p, .stButton>button span, .stButton>button div {
    color: #111827 !important;
    font-weight: 500 !important;
}
.stButton>button:hover p, .stButton>button:hover span, .stButton>button:hover div {
    color: #16A34A !important;
}

/* Download Buttons in Header Bar */
.stDownloadButton>button {
    background-color: #111827 !important;
    color: #FFFFFF !important;
    border: 1px solid #111827 !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    font-weight: 500 !important;
    font-size: 0.86rem !important;
    transition: background-color 0.15s ease !important;
}
.stDownloadButton>button:hover {
    background-color: #1F2937 !important;
}
.stDownloadButton>button p, .stDownloadButton>button span, .stDownloadButton>button div {
    color: #FFFFFF !important;
    font-weight: 500 !important;
}

/* Synchronized Header Row & Dividers */
[data-testid="stSidebarHeader"] {
    margin-bottom: 0px !important;
    padding-bottom: 0px !important;
}
[data-testid="stSidebarUserContent"] {
    padding-top: 0px !important;
}

.clover-sidebar-brand-wrap {
    height: 74.00px !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}
.clover-sidebar-title {
    font-family: 'Newsreader', Georgia, serif !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: #111827 !important;
    letter-spacing: -0.01em !important;
    line-height: 1 !important;
}

.clover-main-brand-wrap {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    height: 70px !important;
}
.clover-main-title {
    font-family: 'Newsreader', Georgia, serif !important;
    font-size: 2.2rem !important;
    font-weight: 600 !important;
    color: #111827 !important;
    line-height: 1 !important;
    letter-spacing: -0.01em !important;
}

.clover-gpu-status-row {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    margin-bottom: 6px !important;
}
.clover-gpu-badge {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.74rem !important;
    color: #166534 !important;
    background: #F0FDF4 !important;
    border: 1px solid #BBF7D0 !important;
    padding: 4px 10px !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    display: inline-flex !important;
    align-items: center !important;
    white-space: nowrap !important;
}

hr.clover-header-divider {
    border: 0 !important;
    border-top: 1px solid #E5E7EB !important;
    margin: 12px 0 20px 0 !important;
    padding: 0 !important;
    height: 1px !important;
    box-sizing: border-box !important;
}

/* Audit-Ready Ledger & Breakdown Card */
.clover-breakdown-card {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    padding: 16px 18px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    box-sizing: border-box !important;
    height: auto !important;
    min-height: 290px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    position: relative !important;
    overflow: visible !important;
}
.clover-breakdown-header {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding-bottom: 8px !important;
    margin-bottom: 6px !important;
    border-bottom: 1px solid #F3F4F6 !important;
    position: relative !important;
    overflow: visible !important;
}
.clover-breakdown-title {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #4B5563 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
.clover-breakdown-badge {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    color: #166534 !important;
    background: #F0FDF4 !important;
    border: 1px solid #BBF7D0 !important;
    padding: 2px 7px !important;
    border-radius: 4px !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
}
.clover-breakdown-item {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: 8px 0 !important;
    border-bottom: 1px solid #F3F4F6 !important;
}
.clover-item-label {
    font-size: 0.83rem !important;
    font-weight: 600 !important;
    color: #111827 !important;
    line-height: 1.25 !important;
}
.clover-item-sub {
    font-size: 0.71rem !important;
    color: #6B7280 !important;
    line-height: 1.2 !important;
    margin-top: 1px !important;
}
.clover-item-val {
    text-align: right !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-end !important;
}
.clover-val-main {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.86rem !important;
    font-weight: 600 !important;
    color: #111827 !important;
    line-height: 1.2 !important;
}
.clover-val-sub {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.69rem !important;
    color: #6B7280 !important;
    line-height: 1.2 !important;
    margin-top: 1px !important;
}
</style>
"""
st.markdown(FLORA_WHITE_CSS, unsafe_allow_html=True)

# Auto-expand sidebar if it was collapsed by cached browser state
components.html(
    """
    <script>
    function ensureSidebarOpen() {
        try {
            const p = window.parent;
            // Clear any stale collapsed flag in parent localStorage
            for (let i = 0; i < p.localStorage.length; i++) {
                const k = p.localStorage.key(i);
                if (k && k.indexOf("stSidebarCollapsed") !== -1) {
                    p.localStorage.removeItem(k);
                }
            }
            // If expand button is rendered, click it to open sidebar immediately
            const expandBtn = p.document.querySelector('button[data-testid="stExpandSidebarButton"]');
            if (expandBtn) {
                expandBtn.click();
            }
        } catch (e) {}
    }
    ensureSidebarOpen();
    setTimeout(ensureSidebarOpen, 150);
    setTimeout(ensureSidebarOpen, 450);
    </script>
    """,
    height=0,
    width=0,
)

# Minimal Lucide/Feather Style SVG Icons (High contrast, clean lines, zero emojis)
SVG_ICONS = {
    "clover": (
        '<svg width="24" height="24" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle;">'
        '<path d="M16 16C12 11 8 8 5 11C2 14 5 18 10 17C12 16.5 14 16 16 16Z" fill="#16A34A" stroke="#15803D" stroke-width="1.2"/>'
        '<path d="M16 16C11 12 8 8 11 5C14 2 18 5 17 10C16.5 12 16 14 16 16Z" fill="#22C55E" stroke="#15803D" stroke-width="1.2"/>'
        '<path d="M16 16C20 11 24 8 27 11C30 14 27 18 22 17C20 16.5 18 16 16 16Z" fill="#16A34A" stroke="#15803D" stroke-width="1.2"/>'
        '<path d="M16 16C11 20 8 24 11 27C14 30 18 27 17 22C16.5 20 16 18 16 16Z" fill="#22C55E" stroke="#15803D" stroke-width="1.2"/>'
        '<path d="M16 16C15 22 14 27 12 30" stroke="#15803D" stroke-width="1.8" stroke-linecap="round"/>'
        '</svg>'
    ),
    "leaf_white": (
        '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 3.5 1 9.2-1.3 7.5-6.5 8.8-9 8.8z"/>'
        '<path d="M2 21c0-3 1.85-5.36 5.08-6"/>'
        '</svg>'
    ),
    "leaf_green": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#165A4C" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: text-bottom; margin-right: 6px;">'
        '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 3.5 1 9.2-1.3 7.5-6.5 8.8-9 8.8z"/>'
        '<path d="M2 21c0-3 1.85-5.36 5.08-6"/>'
        '</svg>'
    ),
    "map": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#165A4C" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: text-bottom; margin-right: 6px;">'
        '<polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/>'
        '<line x1="8" y1="2" x2="8" y2="18"/>'
        '<line x1="16" y1="6" x2="16" y2="22"/>'
        '</svg>'
    ),
    "layers": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#165A4C" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: text-bottom; margin-right: 6px;">'
        '<polygon points="12 2 2 7 12 12 22 7 12 2"/>'
        '<polyline points="2 17 12 22 22 17"/>'
        '<polyline points="2 12 12 17 22 12"/>'
        '</svg>'
    ),
    "bar_chart": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#165A4C" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: text-bottom; margin-right: 6px;">'
        '<line x1="18" y1="20" x2="18" y2="10"/>'
        '<line x1="12" y1="20" x2="12" y2="4"/>'
        '<line x1="6" y1="20" x2="6" y2="14"/>'
        '</svg>'
    ),
    "shield_check": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#165A4C" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: text-bottom; margin-right: 6px;">'
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
        '<polyline points="9 12 11 14 15 10"/>'
        '</svg>'
    ),
    "download": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#165A4C" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: text-bottom; margin-right: 6px;">'
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="7 10 12 15 17 10"/>'
        '<line x1="12" y1="15" x2="12" y2="3"/>'
        '</svg>'
    ),
    "dot_green": (
        '<svg width="8" height="8" viewBox="0 0 8 8" style="display: inline-block; vertical-align: middle; margin-right: 6px;">'
        '<circle cx="4" cy="4" r="3.5" fill="#16A34A"/>'
        '</svg>'
    ),
    "tree_kpi": (
        '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#166534" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -1px; margin-right: 4px;">'
        '<path d="M12 20V10"/>'
        '<path d="M18 20H6"/>'
        '<path d="M12 4a6 6 0 0 0-6 6c0 3 2 5 6 5s6-2 6-5a6 6 0 0 0-6-6z"/>'
        '</svg>'
    ),
    "target_kpi": (
        '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#166534" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -1px; margin-right: 4px;">'
        '<circle cx="12" cy="12" r="10"/>'
        '<circle cx="12" cy="12" r="6"/>'
        '<circle cx="12" cy="12" r="2"/>'
        '</svg>'
    ),
    "diameter_kpi": (
        '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#166534" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -1px; margin-right: 4px;">'
        '<polyline points="15 3 21 3 21 9"/>'
        '<polyline points="9 21 3 21 3 15"/>'
        '<line x1="21" y1="3" x2="14" y2="10"/>'
        '<line x1="3" y1="21" x2="10" y2="14"/>'
        '</svg>'
    ),
    "carbon_kpi": (
        '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#166534" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -1px; margin-right: 4px;">'
        '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 3.5 1 9.2-1.3 7.5-6.5 8.8-9 8.8z"/>'
        '<path d="M2 21c0-3 1.85-5.36 5.08-6"/>'
        '</svg>'
    ),
    "info": (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#165A4C" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px; margin-right: 6px;">'
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="12" y1="16" x2="12" y2="12"/>'
        '<line x1="12" y1="8" x2="12.01" y2="8"/>'
        '</svg>'
    ),
    "info_icon": (
        '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="12" y1="16" x2="12" y2="12"/>'
        '<line x1="12" y1="8" x2="12.01" y2="8"/>'
        '</svg>'
    ),
}

# -----------------------------------------------------------------------------
# 2. Model & Pipeline Resource Loaders (Cached)
# -----------------------------------------------------------------------------

@st.cache_resource
def load_detector() -> TreeDetector:
    return TreeDetector(score_threshold=0.25)

@st.cache_resource
def load_weed_filter() -> WeedWaterFilter:
    return WeedWaterFilter()

@st.cache_resource
def load_area_engine() -> CanopyAreaEngine:
    return CanopyAreaEngine()

def generate_scalloped_crown_polygon(lat_center: float, lon_center: float, radius_m: float, seed_idx: int = 0, num_points: int = 36) -> list:
    """
    Generate natural organic scalloped crown polygon coordinates matching Flora Carbon AI's signature visual style.
    Uses deterministic multi-lobed harmonic perturbation per tree stem.
    """
    rng = np.random.RandomState((int(seed_idx) * 31 + 42) & 0xFFFFFFFF)
    num_lobes = int(rng.choice([6, 7, 8, 9]))
    phase1 = float(rng.uniform(0, 2 * np.pi))
    phase2 = float(rng.uniform(0, 2 * np.pi))
    amp1 = float(rng.uniform(0.08, 0.13))
    amp2 = float(rng.uniform(0.02, 0.05))
    
    thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    r_factors = 1.0 + amp1 * np.cos(num_lobes * thetas + phase1) + amp2 * np.sin((num_lobes * 2) * thetas + phase2)
    
    m_per_lat = 111139.0
    m_per_lon = 111139.0 * np.cos(np.radians(lat_center))
    
    coords = []
    for theta, rf in zip(thetas, r_factors):
        r_curr = radius_m * rf
        d_lat = (r_curr * np.sin(theta)) / m_per_lat
        d_lon = (r_curr * np.cos(theta)) / m_per_lon
        coords.append([lat_center + d_lat, lon_center + d_lon])
    coords.append(coords[0])
    return coords

class FloatingHud(MacroElement):
    def __init__(self, tree_count: int, canopy_area_ha: float, canopy_area_m2: float):
        super().__init__()
        self._template = Template(f"""
        {{% macro html(this, kwargs) %}}
        <div style="
            position: fixed;
            top: 14px;
            right: 60px;
            z-index: 9999;
            background: rgba(10, 26, 18, 0.85);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.22);
            border-radius: 12px;
            padding: 12px 18px;
            color: #FFFFFF;
            font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
            pointer-events: none;
            min-width: 140px;
        ">
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: #86EFAC; font-weight: 600; margin-bottom: 2px;">Tree count</div>
            <div style="font-size: 24px; font-weight: 700; color: #FFFFFF; line-height: 1.1; margin-bottom: 8px;">{tree_count}</div>
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: #86EFAC; font-weight: 600; margin-bottom: 2px;">Canopy area</div>
            <div style="font-size: 15px; font-weight: 600; color: #FFFFFF; line-height: 1.2;">
                {canopy_area_ha:.2f} ha <span style="font-size: 11px; font-weight: 400; opacity: 0.85;">({canopy_area_m2:,.0f} m²)</span>
            </div>
        </div>
        {{% endmacro %}}
        """)

@st.cache_data(show_spinner=False)
def generate_boxed_geotiff_png(tif_path: str, df_boxes: pd.DataFrame) -> bytes:
    """Generate high-resolution PNG with delineated bounding boxes and apical stem points."""
    with rasterio.open(tif_path) as src:
        rgb = src.read([1, 2, 3])
        rgb = np.transpose(rgb, (1, 2, 0))
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    annotated = rgb.copy()
    for _, row in df_boxes.iterrows():
        x1, y1 = int(row["xmin"]), int(row["ymin"])
        x2, y2 = int(row["xmax"]), int(row["ymax"])
        # High-contrast Emerald Green box in RGB
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (34, 197, 94), 2)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        # Glowing apex center dot
        cv2.circle(annotated, (cx, cy), 3, (74, 222, 128), -1)
    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")
    return buf.getvalue()

@st.cache_data(show_spinner=False)
def generate_boxed_plain_image_png(pil_image: Image.Image, df_boxes: pd.DataFrame) -> bytes:
    """Generate annotated PNG with bounding boxes for non-georeferenced images."""
    annotated = np.array(pil_image.convert("RGB")).copy()
    for _, row in df_boxes.iterrows():
        x1, y1 = int(row["xmin"]), int(row["ymin"])
        x2, y2 = int(row["xmax"]), int(row["ymax"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (34, 197, 94), 2)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(annotated, (cx, cy), 3, (74, 222, 128), -1)
    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")
    return buf.getvalue()

# -----------------------------------------------------------------------------
# 3. Presets Registry (Removed Tollygunge from Presets, Retained Clean 5)
# -----------------------------------------------------------------------------

PRESET_CONFIGS = {
    "kolkata_central_park": {
        "name": "Kolkata Central Park (Banabitan, Salt Lake)",
        "badge": "01 · URBAN ECOLOGICAL PARK",
        "description": "8.1 ha urban park with Banabitan lake. Features active hydrological filter purging floating water hyacinth (Eichhornia crassipes).",
        "tif_path": "data/samples/kolkata_central_park.tif",
        "kml_path": "data/samples/kolkata_central_park_aoi.kml",
        "default_gsd": 0.27,
        "location": "Bidhannagar (Salt Lake Sector II), Kolkata",
        "crs_str": "EPSG:32645 (UTM Zone 45N)",
    },
    "jadavpur_university": {
        "name": "Jadavpur University Main Campus",
        "badge": "02 · INSTITUTIONAL CANOPY",
        "description": "7.94 ha dense academic campus with multi-story buildings, walkways, and mature tropical banyan, rain, and mahogany trees.",
        "tif_path": "data/samples/jadavpur_university.tif",
        "default_gsd": 0.27,
        "location": "Jadavpur, Kolkata",
        "crs_str": "EPSG:32645 (UTM Zone 45N)",
    },
    "victoria_memorial": {
        "name": "Victoria Memorial Gardens & Kolkata Maidan",
        "badge": "03 · HERITAGE LANDSCAPE",
        "description": "8.16 ha heritage grounds surrounding the white marble monument, manicured lawns, formal ornamental ponds, and perimeter heritage trees.",
        "tif_path": "data/samples/kolkata_maidan_victoria.tif",
        "default_gsd": 0.27,
        "location": "Maidan / Queen's Way, Kolkata",
        "crs_str": "EPSG:32645 (UTM Zone 45N)",
    },
    "sundarbans_mangrove": {
        "name": "Sundarbans Biosphere Mangrove Reserve",
        "badge": "04 · BLUE CARBON / MANGROVE",
        "description": "8.20 ha halophytic intertidal mangrove forest (Heritiera fomes & Rhizophora mangle) along tidal creeks.",
        "tif_path": "data/samples/sundarbans_mangrove.tif",
        "default_gsd": 0.27,
        "location": "Sundarbans Biosphere Reserve, West Bengal",
        "crs_str": "EPSG:32645 (UTM Zone 45N)",
    },
    "neon_benchmark": {
        "name": "Official NEON Ground-Truth Benchmark (OSBS_029)",
        "badge": "05 · SCIENTIFIC BENCHMARK",
        "description": "VHR 10cm airborne imagery with 61 ground-truthed surveyed trees. Proves 80.0% precision and Li et al. 2023 bias calibration.",
        "tif_path": "data/samples/OSBS_029.tif",
        "csv_path": "data/samples/OSBS_029.csv",
        "default_gsd": 0.10,
        "location": "Ordway-Swisher Biological Station, FL",
        "crs_str": "EPSG:32617 (UTM Zone 17N)",
    },
    "delhi_lodi_gardens": {
        "name": "Lodi Gardens Heritage Park, New Delhi",
        "badge": "06 · HERITAGE CANOPY",
        "description": "Historic Mughal park with ancient neem, banyan, and jamun canopies. Features 71 delineated crowns within official 2.7 ha boundary.",
        "tif_path": "data/samples/delhi_lodi_gardens.tif",
        "kml_path": "data/samples/delhi_lodi_gardens_aoi.kml",
        "default_gsd": 0.26,
        "location": "Lodhi Road, New Delhi",
        "crs_str": "EPSG:32643 (UTM Zone 43N)",
    },
    "bengaluru_cubbon_park": {
        "name": "Cubbon Park Arboretum, Bengaluru",
        "badge": "07 · TROPICAL ARBORETUM",
        "description": "Dense urban garden city forest with silver oak, mahogany, and bamboo groves (~150 crowns across 2.9 ha).",
        "tif_path": "data/samples/bengaluru_cubbon_park.tif",
        "kml_path": "data/samples/bengaluru_cubbon_park_aoi.kml",
        "default_gsd": 0.29,
        "location": "Kasturba Road, Bengaluru",
        "crs_str": "EPSG:32643 (UTM Zone 43N)",
    },
    "bengaluru_lalbagh_garden": {
        "name": "Lalbagh Botanical Garden, Bengaluru",
        "badge": "08 · BOTANICAL CANOPY",
        "description": "Centuries-old royal botanical garden with century-old tropical deciduous trees surrounding the lotus lake.",
        "tif_path": "data/samples/bengaluru_lalbagh_garden.tif",
        "kml_path": "data/samples/bengaluru_lalbagh_garden_aoi.kml",
        "default_gsd": 0.29,
        "location": "Mavalli, Bengaluru",
        "crs_str": "EPSG:32643 (UTM Zone 43N)",
    },
    "mumbai_aarey_forest": {
        "name": "Aarey Forest / SGNP, Mumbai",
        "badge": "09 · URBAN CONSERVATION",
        "description": "Tropical moist deciduous forest canopy contiguous with Sanjay Gandhi National Park (~87 mature stems).",
        "tif_path": "data/samples/mumbai_aarey_forest.tif",
        "kml_path": "data/samples/mumbai_aarey_forest_aoi.kml",
        "default_gsd": 0.28,
        "location": "Goregaon East, Mumbai",
        "crs_str": "EPSG:32643 (UTM Zone 43N)",
    },
    "california_muir_woods": {
        "name": "Muir Woods Coastal Redwoods, California",
        "badge": "10 · TEMPERATE RAINFOREST",
        "description": "Ancient old-growth coastal redwood overstory (Sequoia sempervirens) with 200 mature crowns.",
        "tif_path": "data/samples/california_muir_woods.tif",
        "kml_path": "data/samples/california_muir_woods_aoi.kml",
        "default_gsd": 0.23,
        "location": "Mill Valley, Marin County, CA",
        "crs_str": "EPSG:32610 (UTM Zone 10N)",
    },
}

# -----------------------------------------------------------------------------
# 4. Processing Pipelines (Cached for Sub-Second Performance)
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def process_geotiff(
    tif_path: str,
    score_threshold: float = 0.25,
    apply_weed_filter: bool = True,
    kml_clip_path: Optional[str] = None,
) -> dict:
    """Analyze a georeferenced GeoTIFF."""
    t0 = time.time()
    detector = load_detector()
    area_engine = load_area_engine()
    weed_filter = load_weed_filter()

    active_tif = tif_path
    kml_gdf = None
    if kml_clip_path and os.path.exists(kml_clip_path):
        meta_raw = inspect_raster(tif_path)
        kml_gdf = parse_aoi(kml_clip_path, target_crs=meta_raw.crs)
        clipped_dst = tempfile.mktemp(suffix="_kml_clipped.tif")
        active_tif, _ = clip_raster_to_aoi(tif_path, kml_gdf, output_path=clipped_dst)

    meta = inspect_raster(active_tif)
    tiler = SlidingWindowTiler(tile_size=400, overlap=0.20)
    pred_df, _ = detector.predict_tiled_raster(active_tif, tiler=tiler, score_threshold=score_threshold)

    purged_df = pd.DataFrame()
    if apply_weed_filter and len(pred_df) > 0:
        filter_res = weed_filter.filter_raster_detections(active_tif, pred_df)
        clean_df = filter_res.clean_detections
        purged_df = filter_res.purged_detections
    else:
        clean_df = pred_df

    metrics, enriched_df, dissolved_poly = area_engine.compute_metrics(
        clean_df, raster_metadata=meta, aoi_polygon=kml_gdf
    )

    n_trees = max(1, metrics.tree_count)
    # Dual-component uncertainty: random tree-level residual (Jucker et al. 2016, 56.5% / √N) 
    # combined with systematic stand-level allometric/wood-density error floor (~12.0%, Chave et al. 2014)
    rand_err_pct = 56.5 / np.sqrt(n_trees)
    sys_err_pct = 12.0  # Systematic allometric model and species variance floor
    stand_uncertainty_pct = float(np.sqrt(rand_err_pct**2 + sys_err_pct**2))
    stand_uncertainty_mg = metrics.agb_total_mg * (stand_uncertainty_pct / 100.0)
    carbon_uncertainty_mg = metrics.carbon_stock_mg_c * (stand_uncertainty_pct / 100.0)

    with rasterio.open(active_tif) as src:
        bounds = src.bounds
        crs = src.crs

    elapsed = time.time() - t0

    if crs and crs.is_projected:
        wgs_bounds = transform_bounds(crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top)
    else:
        wgs_bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)

    return {
        "metrics": metrics,
        "enriched_df": enriched_df,
        "purged_df": purged_df,
        "meta": meta,
        "wgs_bounds": wgs_bounds,
        "stand_uncertainty_pct": stand_uncertainty_pct,
        "stand_uncertainty_mg": stand_uncertainty_mg,
        "carbon_uncertainty_mg": carbon_uncertainty_mg,
        "elapsed_sec": elapsed,
        "is_geotiff": True,
        "tif_path": active_tif,
    }


def process_plain_image_upload(
    pil_image: Image.Image,
    assumed_gsd_m: float = 0.35,
    score_threshold: float = 0.18,
    use_multi_scale: bool = True,
) -> dict:
    """Analyze a non-georeferenced uploaded image or pasted screenshot with adaptive multi-scale pyramid."""
    t0 = time.time()
    img_np = np.array(pil_image.convert("RGB"))
    h, w, _ = img_np.shape

    total_area_m2 = (w * assumed_gsd_m) * (h * assumed_gsd_m)
    total_area_ha = total_area_m2 / 10000.0

    detector = load_detector()
    tiler = SlidingWindowTiler(tile_size=400, overlap=0.20)

    tile_preds = []

    # 1. Native Resolution (1.0x) - Detects mature overstory & emergent heritage crowns
    for tile in tiler.iterate_array(img_np):
        tile_df = detector.predict_tile(tile.image, score_threshold=score_threshold)
        if len(tile_df) > 0:
            tile_preds.append(tiler.map_local_to_global_pixels(tile, tile_df))

    # 2. Adaptive Feature Pyramid (2.0x+) - Detects smaller crowns & zoomed-out captures
    if use_multi_scale and max(h, w) <= 4000:
        # Scale factor tuned to match model canonical receptive field (~0.25m)
        pyramid_scale = 2.0
        if assumed_gsd_m >= 0.45:
            pyramid_scale = min(3.0, max(1.75, assumed_gsd_m / 0.25))

        rescaled_w = int(w * pyramid_scale)
        rescaled_h = int(h * pyramid_scale)
        img_scaled = cv2.resize(img_np, (rescaled_w, rescaled_h), interpolation=cv2.INTER_LANCZOS4)

        for tile in tiler.iterate_array(img_scaled):
            tile_df = detector.predict_tile(tile.image, score_threshold=score_threshold)
            if len(tile_df) > 0:
                mapped = tiler.map_local_to_global_pixels(tile, tile_df)
                mapped["xmin"] /= pyramid_scale
                mapped["ymin"] /= pyramid_scale
                mapped["xmax"] /= pyramid_scale
                mapped["ymax"] /= pyramid_scale
                tile_preds.append(mapped)

    if tile_preds:
        raw_df = pd.concat(tile_preds, ignore_index=True)
        dedup_df = detector.apply_nms(raw_df, iou_threshold=0.22)
    else:
        dedup_df = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])

    # Scale pixel boxes to metric space
    if len(dedup_df) > 0:
        dedup_df["center_x"] = ((dedup_df["xmin"] + dedup_df["xmax"]) / 2.0) * assumed_gsd_m
        dedup_df["center_y"] = ((dedup_df["ymin"] + dedup_df["ymax"]) / 2.0) * assumed_gsd_m
        dedup_df["width_m"] = (dedup_df["xmax"] - dedup_df["xmin"]) * assumed_gsd_m
        dedup_df["height_m"] = (dedup_df["ymax"] - dedup_df["ymin"]) * assumed_gsd_m
        dedup_df["crown_diameter_m"] = np.sqrt(dedup_df["width_m"] * dedup_df["height_m"])

    area_engine = load_area_engine()
    metrics, enriched_df, _ = area_engine.compute_metrics(dedup_df, aoi_area_m2=total_area_m2)

    n_trees = max(1, metrics.tree_count)
    rand_err_pct = 56.5 / np.sqrt(n_trees)
    sys_err_pct = 12.0
    stand_uncertainty_pct = float(np.sqrt(rand_err_pct**2 + sys_err_pct**2))
    stand_uncertainty_mg = metrics.agb_total_mg * (stand_uncertainty_pct / 100.0)
    carbon_uncertainty_mg = metrics.carbon_stock_mg_c * (stand_uncertainty_pct / 100.0)

    elapsed = time.time() - t0

    return {
        "metrics": metrics,
        "enriched_df": enriched_df,
        "purged_df": pd.DataFrame(),
        "meta": None,
        "wgs_bounds": None,
        "stand_uncertainty_pct": stand_uncertainty_pct,
        "stand_uncertainty_mg": stand_uncertainty_mg,
        "carbon_uncertainty_mg": carbon_uncertainty_mg,
        "elapsed_sec": elapsed,
        "is_geotiff": False,
        "pil_image": pil_image,
        "used_multi_scale": use_multi_scale,
    }

# -----------------------------------------------------------------------------
# 5. Interactive Popout Modals (Technical Deep Dive & Disclosures)
# -----------------------------------------------------------------------------

@st.dialog("Technical Deep Dive: Closed-Canopy vs Discrete Crown Dilemma", width="large")
def show_closed_canopy_modal():
    st.markdown(
        r"""
        ### Why is this Methodology Required in Clover?

        When machine learning models quantify urban and natural forests from optical imagery, they face a fundamental ecological split:
        - **In open savannahs or street tree rows**: Individual crowns have clear open ground gaps, distinct shadows, and can be isolated cleanly with traditional bounding boxes.
        - **In contiguous forests (natural tropical broadleaf, mangroves, dense urban parks)**: Touching branches intertwine into a seamless green roof ("closed canopy").

        #### The Risk of Naive Bounding Boxes in Carbon Accounting
        1. **Fraudulent Double-Counting**: In closed canopies, overlapping bounding boxes overlap one another by 25% to 50%. Summing raw box areas double-counts the same ground patch, artificially inflating verified carbon credits.
        2. **Nadir Optical Limits**: 2D nadir cameras cannot physically see tree trunk separations beneath a contiguous canopy without 3D waveform LiDAR.

        #### How Clover Solves This (The 3-Step Scientific Solution):
        1. **Continuous Dissolved Footprint ($m^2$)**: Rather than summing bounding boxes, Clover dissolves overlapping crown geometries into a continuous topological polygon using geometric unary union (`shapely.unary_union`). This guarantees that **no square meter of forest is double-credited**.
        2. **Empirical Area Calibration (+20% Correction)**: Bounding-box detectors overestimate crown area by $+20\\%$ in open trees and underestimate in closed clusters. Clover applies empirical bias calibration before running allometric equations.
        3. **Stand Uncertainty Modeling**: Individual crown allometry from 2D optical diameter carries single-tree residual error ($\sigma \approx \pm 56.5\%$ relative standard error in *Jucker et al. 2016*). While random tree-level errors attenuate across $N$ stems ($\sigma_{\text{random}} \approx 56.5\% / \sqrt{N}$), systematic model and wood density errors impose an allometric stand-level floor ($\sim 10\text{--}12\%$, *Chave et al. 2014*). Verra VCS precision guidelines require accounting for both components:
        $$\sigma_{\text{stand}} = \sqrt{\frac{(56.5\%)^2}{N} + \sigma_{\text{systematic}}^2}$$
        Across a representative stand of $N = 100\text{--}300$ stems, random noise cancels out and aggregate uncertainty settles near $\pm 12\text{--}13\%$, providing realistic, auditor-defensible confidence intervals rather than falsely claiming near-zero error.
        """
    )
    if os.path.exists("reports/why_boxes_missed_comparison.png"):
        st.image(
            "reports/why_boxes_missed_comparison.png",
            caption="Comparison: Traditional Bounding Boxes vs. Continuous Dissolved Canopy Coverage",
            use_container_width=True,
        )
    st.markdown(
        """
        ---
        #### Embedded Academic Literature & Peer-Reviewed Sources
        - **Li et al. (PNAS Nexus 2023)**: *Biases in individual tree crown delineation from high-resolution remote sensing and allometric biomass implications.* [DOI: 10.1093/pnasnexus/pgad098](https://doi.org/10.1093/pnasnexus/pgad098)
        - **Jucker et al. (Global Change Biology 2016)**: *Allometric equations for integrating aerial LiDAR with optical remote sensing to map tropical forest carbon stocks.* [DOI: 10.1111/gcb.13388](https://doi.org/10.1111/gcb.13388)
        - **Weinstein et al. (Remote Sensing of Environment 2020)**: *Cross-site evaluation of deep learning for individual tree crown detection in airborne RGB imagery.* [DOI: 10.1016/j.rse.2020.111815](https://doi.org/10.1016/j.rse.2020.111815)
        - **Brandt et al. (Nature 2020)**: *An unexpectedly large count of trees in the West African Sahara and Sahel.* [DOI: 10.1038/s41586-020-2824-5](https://doi.org/10.1038/s41586-020-2824-5)
        """
    )

@st.dialog("Scientific Disclosures & MRV Integrity (Verra VM0047 & IPCC Standards)", width="large")
def show_regulatory_modal():
    st.markdown(
        r"""
        ### Why are Scientific Disclosures Required in Clover?

        Under high-integrity voluntary and compliance carbon registries (**Verra VCS**, Gold Standard, Plan Vivo, Article 6), automated MRV platforms **cannot issue credits without explicit scientific disclosures**. Third-party validation and verification bodies (VVBs) mandate full transparency on how optical remote sensing physical constraints are treated.

        ---
        ### The 4 Core MRV Physical Disclosures

        #### 1. Understory Blind Spot Disclosure (Sensor Penetration Constraints)
        - **Physical Limit**: Optical RGB satellite and drone sensors only capture top-of-canopy reflections; sub-canopy saplings and shaded stems are hidden from nadir view.
        - **Biomass Impact**: In stratified mature stands, dominant and co-dominant overstory trees represent **75% to 85% of above-ground biomass (AGB)** (*IPCC AFOLU 2019 Refinement, Vol 4*).
        - **Audit Treatment**: Clover quantifies overstory canopy biomass. For formal carbon credit issuance under Verra VM0047, project developers pair optical baselines with terrestrial sample plots or conservative regional Biomass Expansion Factors (BEF).

        #### 2. Wood Density ($\\rho$) Allometric Variance
        - **Physical Limit**: 3-band optical RGB sensors cannot discern internal cellular wood density.
        - **Peer-Reviewed Values**: Native dense hardwoods like Sal (*Shorea robusta*, $\\rho \\approx 0.72\\text{ g/cm}^3$) or Teak (*Tectona grandis*, $\\rho \\approx 0.65\\text{ g/cm}^3$) sequester double the carbon of softwoods like Silk Cotton (*Bombax ceiba*, $\\rho \\approx 0.35\\text{ g/cm}^3$) for identical crown diameters (*Source: Global Wood Density Database, Dryad; Zanne et al. 2009 / Chave et al. 2009*).
        - **Audit Treatment**: Clover couples optical crown diameters with *Jucker et al. (2016)* pantropical allometric equations as an illustrative baseline. Project developers must calibrate with localized species stratification.

        #### 3. Closed-Canopy Intertwining & Area Accounting (Verra VM0047 Principles)
        - **Physical Limit**: Contiguous tree canopies touch and overlap.
        - **Audit Treatment**: To prevent double-counting in afforestation/reforestation (ARR) accounting, Clover quantifies forest coverage by **dissolved non-overlapping surface footprint ($m^2$)** using unary topological union, guaranteeing zero double-crediting.

        #### 4. Dual-Component Uncertainty Decomposition (Verra VCS Standard v4.5)
        - **Audit Requirement**: Verra requires projects to report confidence bounds (typically 90% or 95% CI). If overall uncertainty exceeds precision thresholds, conservative deductions are applied to the credit pool (*VCS Standard Section 4.5.18*).
        - **Audit Treatment**: Rather than naively assuming all error cancels to zero via $1/\sqrt{N}$, Clover models both random tree variance ($\sigma_{\text{rand}} \approx 56.5\%$, *Jucker et al. 2016*) and a representative systematic allometric/sensor error floor ($\sigma_{\text{sys}} \approx 12.0\%$, selected within the empirical 10–44% stand-scale range documented by *Chave et al. 2014* and *Réjou-Méchain et al. 2017*):
        $$\sigma_{\text{stand}} = \sqrt{\frac{(56.5\%)^2}{N} + (12.0\%)^2}$$
        For stands of $N > 100$ trees, stand uncertainty converges realistically to **$\pm 12\text{--}14\%$**, reflecting true remote sensing accuracy without fabricating precision.

        ---
        #### Regulatory Standards & Methodology References
        - **Verra VM0047 (2023)**: *Methodology for Afforestation, Reforestation, and Revegetation Projects (ARR), Version 1.0.* [Official Verra VM0047](https://verra.org/methodologies/vm0047-afforestation-reforestation-and-revegetation-v1-0/)
        - **Verra VCS Standard (v4.5, 2023)**: *Section 4.5: Precision, Uncertainty, and Confidence Deductions.* [Official VCS Standard](https://verra.org/programs/vcs/)
        - **Chave et al. (Global Change Biology 2014)**: *Improved pantropical allometric models to estimate the aboveground biomass of tropical trees.* [DOI: 10.1111/gcb.12629](https://doi.org/10.1111/gcb.12629)
        - **Réjou-Méchain et al. (Methods in Ecology and Evolution 2017)**: *BIOMASS: An R package for estimating above-ground biomass and its uncertainty in tropical forests.* [DOI: 10.1111/2041-210X.12753](https://doi.org/10.1111/2041-210X.12753)
        - **Jucker et al. (Global Change Biology 2016)**: *Allometric equations for integrating aerial LiDAR with optical remote sensing to map tropical forest carbon stocks.* [DOI: 10.1111/gcb.13388](https://doi.org/10.1111/gcb.13388)
        - **Zanne et al. / Chave et al. (Global Wood Density Database, Dryad 2009)**: *Towards a worldwide wood economics spectrum.* [Dryad Repository](https://datadryad.org/stash/dataset/doi:10.5061/dryad.234)
        - **IPCC Guidelines (2019 Refinement)**: *IPCC Guidelines for National Greenhouse Gas Inventories: Volume 4 (AFOLU).* [Official IPCC Guidance](https://www.ipcc-nggip.iges.or.jp/public/2019rf/vol4.html)
        """
    )

# -----------------------------------------------------------------------------
# 6. Sidebar Navigation & Control Panel
# -----------------------------------------------------------------------------

with st.sidebar:
    logo_html = (
        f'<img src="data:image/png;base64,{CLOVER_LOGO_B64}" width="32" height="32" style="object-fit: contain; vertical-align: middle;" />'
        if CLOVER_LOGO_B64
        else SVG_ICONS["clover"]
    )
    st.markdown(
        f"""
        <div class="clover-sidebar-brand-wrap">
            {logo_html}
            <span class="clover-sidebar-title">Clover</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='clover-header-divider' />", unsafe_allow_html=True)

    input_mode = st.radio(
        "Select Workflow Input",
        options=["Curated Site & Benchmark Presets", "Upload Custom Imagery / Paste Screenshot"],
        index=0,
    )

    selected_preset_key = "kolkata_central_park"
    active_plain_image = None
    custom_gsd = 0.27

    # Dynamic OS keyboard shortcut detection
    is_mac = platform.system() == "Darwin"
    shortcut_hint = "Cmd+V" if is_mac else "Ctrl+V"

    if input_mode == "Curated Site & Benchmark Presets":
        selected_preset_key = st.selectbox(
            "Select Site Preset",
            options=list(PRESET_CONFIGS.keys()),
            format_func=lambda k: PRESET_CONFIGS[k]["name"],
            index=0,
        )
        current_cfg = PRESET_CONFIGS[selected_preset_key]
        st.markdown(
            f"<div style='font-size: 0.8rem; color: #4B5563; background: #FFFFFF; padding: 12px; border-radius: 6px; border: 1px solid #E5E7EB; margin-top: 8px;'>"
            f"<b>Location:</b> {current_cfg['location']}<br/>"
            f"<b>CRS:</b> <code>{current_cfg['crs_str']}</code><br/>"
            f"<b>Nominal GSD:</b> {current_cfg['default_gsd']*100:.1f} cm/px"
            f"</div>",
            unsafe_allow_html=True,
        )

        use_kml = False
        if "kml_path" in current_cfg:
            use_kml = st.checkbox("Clip to Official Cadastral Boundary (KML)", value=False)

    else:
        st.markdown("#### **Custom Upload & Clipboard Paste**")
        st.markdown(
            f"<small style='color: #6B7280;'>Paste an image directly from clipboard ({shortcut_hint}) or upload a file.</small>",
            unsafe_allow_html=True,
        )

        # 1. Paste Screenshot Button (Dynamic OS Cmd+V / Ctrl+V)
        paste_result = paste_image_button(
            label=f"Paste Screenshot ({shortcut_hint})",
            background_color="#111827",
            hover_background_color="#1F2937",
            text_color="#FFFFFF",
            key="clipboard_paste",
        )
        if paste_result.image_data is not None:
            active_plain_image = paste_result.image_data
            st.success("Screenshot pasted successfully from clipboard.")

        # 2. File Uploader for GeoTIFF, JPG, PNG
        uploaded_file = st.file_uploader(
            "Or Upload Satellite Image / Screenshot",
            type=["tif", "tiff", "jpg", "jpeg", "png"],
            help="Upload a GeoTIFF or standard drone / Google Earth screenshot.",
        )
        if uploaded_file is not None and active_plain_image is None:
            if uploaded_file.name.lower().endswith((".tif", ".tiff")):
                # Save temp geotiff
                temp_tif = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
                temp_tif.write(uploaded_file.read())
                temp_tif.close()
                custom_tif_path = temp_tif.name
            else:
                active_plain_image = Image.open(uploaded_file).convert("RGB")

        # Zoom / Scale Preset Selection
        zoom_preset = st.selectbox(
            "Capture Zoom Scale",
            options=[
                "Auto-Adaptive Multi-Scale (Recommended for Google Maps / Satellite)",
                "Standard Google Earth Zoom 19 (~0.27 m/px)",
                "High-Res Drone / Aerial (< 0.15 m/px)",
                "Manual Custom GSD",
            ],
            index=0,
            help="Auto-Adaptive automatically scales the receptive field so any zoom level detects trees reliably without manual calibration.",
        )

        if zoom_preset.startswith("Auto-Adaptive"):
            custom_gsd = 0.40
            use_multi_scale = True
            st.markdown(
                "<div style='font-size: 0.78rem; color: #166534; background: #F0FDF4; border: 1px solid #BBF7D0; padding: 6px 10px; border-radius: 4px; margin-top: 4px;'>"
                "<b>Active:</b> Dual-resolution pyramid (1.0x + 2.0x) enabled. Resolves both emergent crowns and small clusters automatically."
                "</div>",
                unsafe_allow_html=True,
            )
        elif zoom_preset.startswith("Standard Google Earth"):
            custom_gsd = 0.27
            use_multi_scale = True
        elif zoom_preset.startswith("High-Res Drone"):
            custom_gsd = 0.12
            use_multi_scale = False
        else:
            custom_gsd = st.slider(
                "Estimated Ground Sampling Distance (GSD)",
                min_value=0.05,
                max_value=1.50,
                value=0.35,
                step=0.01,
                help="Spatial resolution in meters per pixel. Google Earth Zoom 19 is ~0.27m. High-res drone is ~0.10m.",
            )
            use_multi_scale = st.checkbox("Enable Multi-Scale Feature Pyramid", value=True)

    st.markdown("---")
    st.markdown("#### **Detection Tuning**")
    default_score = 0.18 if input_mode != "Curated Site & Benchmark Presets" else 0.25
    score_thresh = st.slider(
        "Confidence Score Threshold",
        min_value=0.10,
        max_value=0.60,
        value=default_score,
        step=0.02,
        help="Higher values favor precision; lower values favor recall in shaded tree canopies.",
    )
    enable_filter = st.checkbox(
        "Enable Hydrological Lake Weed Filter",
        value=True,
        help="Purges floating water hyacinth (Eichhornia crassipes) from lakes and ponds.",
    )

    st.markdown("---")
    st.markdown("##### **Auditor Disclosures & Science**")
    if st.button("Why Dissolved Footprints? (Closed Canopy)", icon=":material/open_in_new:", use_container_width=True):
        show_closed_canopy_modal()

    if st.button("Verra VM0047 Regulatory Standards", icon=":material/open_in_new:", use_container_width=True):
        show_regulatory_modal()

# -----------------------------------------------------------------------------
# 6. Execute Analysis Pipeline
# -----------------------------------------------------------------------------

if input_mode == "Curated Site & Benchmark Presets":
    active_cfg = PRESET_CONFIGS.get(selected_preset_key, PRESET_CONFIGS["kolkata_central_park"])
    kml_to_pass = active_cfg.get("kml_path") if use_kml else None
    with st.spinner("Processing satellite imagery with deep neural canopy delineation..."):
        results = process_geotiff(
            tif_path=active_cfg["tif_path"],
            score_threshold=score_thresh,
            apply_weed_filter=enable_filter,
            kml_clip_path=kml_to_pass,
        )
else:
    # Custom Upload or Pasted Screenshot
    if "use_multi_scale" not in locals():
        use_multi_scale = True
    if active_plain_image is not None:
        with st.spinner("Processing pasted image with adaptive multi-scale pyramid..."):
            results = process_plain_image_upload(
                pil_image=active_plain_image,
                assumed_gsd_m=custom_gsd,
                score_threshold=score_thresh,
                use_multi_scale=use_multi_scale,
            )
    elif "custom_tif_path" in locals() and os.path.exists(custom_tif_path):
        with st.spinner("Processing custom GeoTIFF..."):
            results = process_geotiff(
                tif_path=custom_tif_path,
                score_threshold=score_thresh,
                apply_weed_filter=enable_filter,
            )
    else:
        # Fallback to Kolkata Central Park default
        active_cfg = PRESET_CONFIGS["kolkata_central_park"]
        with st.spinner("Loading Kolkata Central Park..."):
            results = process_geotiff(
                tif_path=active_cfg["tif_path"],
                score_threshold=score_thresh,
                apply_weed_filter=enable_filter,
            )

metrics: CanopyMetrics = results["metrics"]
df_trees: pd.DataFrame = results["enriched_df"]
df_purged: pd.DataFrame = results["purged_df"]

# Prepare GeoJSON & CSV Export buffers for top download buttons
features = []
for idx, row in df_trees.iterrows():
    cd = float(row.get("crown_diameter_m", 6.0))
    if "lat" in row and "lon" in row:
        point = shapely.geometry.Point(row["lon"], row["lat"])
        circle = point.buffer(cd / (2.0 * 111320.0))
        features.append({
            "type": "Feature",
            "geometry": mapping(circle),
            "properties": {
                "stem_id": int(idx + 1),
                "crown_diameter_m": round(cd, 2),
                "crown_area_m2": round(np.pi / 4.0 * cd**2, 2),
                "agb_kg": round(float(np.exp(-0.328 + 2.404 * np.log(max(0.1, cd)))), 2),
                "confidence_score": round(float(row.get("score", 0.5)), 3),
            },
        })

geojson_data = {
    "type": "FeatureCollection",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
    "features": features,
}
geojson_str = json.dumps(geojson_data, indent=2)

csv_df = df_trees.copy()
drop_cols = [c for c in ["tile_idx", "geometry"] if c in csv_df.columns]
csv_df = csv_df.drop(columns=drop_cols, errors="ignore")
csv_buffer = io.StringIO()
csv_df.to_csv(csv_buffer, index=False)
csv_str = csv_buffer.getvalue()

# Prepare Bounding Box Annotated Image PNG Buffer
if results.get("is_geotiff", False):
    tif_p = results.get("tif_path") or active_cfg.get("tif_path", "")
    boxed_png_bytes = generate_boxed_geotiff_png(tif_p, df_trees) if tif_p and os.path.exists(tif_p) else b""
elif "pil_image" in results:
    boxed_png_bytes = generate_boxed_plain_image_png(results["pil_image"], df_trees)
else:
    boxed_png_bytes = b""

# -----------------------------------------------------------------------------
# 7. Main Dashboard Header & Quick Actions (Top Download Buttons)
# -----------------------------------------------------------------------------

col_logo, col_actions = st.columns([1.5, 1.5], vertical_alignment="center")

with col_logo:
    logo_html = (
        f'<img src="data:image/png;base64,{CLOVER_LOGO_B64}" width="38" height="38" style="object-fit: contain; vertical-align: middle;" />'
        if CLOVER_LOGO_B64
        else SVG_ICONS["clover"]
    )
    st.markdown(
        f"""
        <div class="clover-main-brand-wrap">
            {logo_html}
            <span class="clover-main-title">
                Clover
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_actions:
    sub_col1, sub_col2, sub_col3 = st.columns(3)
    with sub_col1:
        st.download_button(
            label="GeoJSON",
            data=geojson_str,
            file_name=f"clover_crowns_{selected_preset_key}.geojson",
            mime="application/geo+json",
            icon=":material/download:",
            use_container_width=True,
        )
    with sub_col2:
        st.download_button(
            label="CSV",
            data=csv_str,
            file_name=f"clover_stem_inventory_{selected_preset_key}.csv",
            mime="text/csv",
            icon=":material/download:",
            use_container_width=True,
        )
    with sub_col3:
        st.download_button(
            label="Boxed PNG",
            data=boxed_png_bytes,
            file_name=f"clover_bounding_boxes_{selected_preset_key}.png",
            mime="image/png",
            icon=":material/image:",
            use_container_width=True,
            disabled=len(boxed_png_bytes) == 0,
        )

st.markdown("<hr class='clover-header-divider' />", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 8. Executive KPI Cards
# -----------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="flora-metric-card">
            <div class="flora-metric-top">
                <div class="flora-metric-lbl">{SVG_ICONS['tree_kpi']} 01 · STEM COUNT</div>
                <div class="flora-tooltip-wrap">
                    <span class="flora-info-btn">{SVG_ICONS['info_icon']}</span>
                    <div class="flora-tooltip-card">
                        <b style="color: #111827;">Tree Detection Methodology</b><br/>
                        • <b>Model:</b> DeepForest 2.1 RetinaNet with sliding-window tiling (400×400 px, 15% overlap).<br/>
                        • <b>Deduplication:</b> Non-Maximum Suppression (NMS, IoU &lt; 0.22) across boundary seams.<br/>
                        • <b>Hydrological Filter:</b> Purges floating weeds (water hyacinth) inside water bodies.
                    </div>
                </div>
            </div>
            <div class="flora-metric-val">{metrics.tree_count} <span style="font-size: 1rem; font-family: 'Inter'; color: #6B7280; font-weight: normal;">stems</span></div>
            <div class="flora-metric-sub">Density: <b>{metrics.tree_density_per_ha:.1f} stems/ha</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="flora-metric-card">
            <div class="flora-metric-top">
                <div class="flora-metric-lbl">{SVG_ICONS['target_kpi']} 02 · CANOPY COVER</div>
                <div class="flora-tooltip-wrap">
                    <span class="flora-info-btn">{SVG_ICONS['info_icon']}</span>
                    <div class="flora-tooltip-card">
                        <b style="color: #111827;">Canopy Area Calculation</b><br/>
                        • <b>Geometric Dissolve:</b> Overlapping crowns merged with Shapely <code>unary_union</code> to prevent double-counting (Verra VM0047).<br/>
                        • <b>Bias Calibration:</b> +20% adjustment factor per <i>Li et al. (PNAS Nexus 2023)</i>.<br/>
                        • <b>Formula:</b> Canopy Cover % = Dissolved Area / Parcel Area × 100.
                    </div>
                </div>
            </div>
            <div class="flora-metric-val">{metrics.canopy_cover_pct_calibrated:.1f}%</div>
            <div class="flora-metric-sub">Area: <b>{metrics.total_calibrated_canopy_area_m2:,.0f} m²</b> ({metrics.total_calibrated_canopy_area_m2/10000:.2f} ha)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="flora-metric-card">
            <div class="flora-metric-top">
                <div class="flora-metric-lbl">{SVG_ICONS['diameter_kpi']} 03 · MEAN CROWN DIAMETER</div>
                <div class="flora-tooltip-wrap">
                    <span class="flora-info-btn">{SVG_ICONS['info_icon']}</span>
                    <div class="flora-tooltip-card">
                        <b style="color: #111827;">Crown Metric Derivation</b><br/>
                        • <b>Diameter:</b> Geometric mean of bounding box dimensions: D = √(width × height) × GSD.<br/>
                        • <b>Shape Factor:</b> Elliptical crown footprint factor k = π/4 ≈ 0.7854.<br/>
                        • <b>Benchmark:</b> Validated against NEON OSBS airborne LiDAR ground truth.
                    </div>
                </div>
            </div>
            <div class="flora-metric-val">{metrics.mean_crown_diameter_m:.2f} <span style="font-size: 1rem; font-family: 'Inter'; color: #6B7280; font-weight: normal;">m</span></div>
            <div class="flora-metric-sub">Median: <b>{metrics.median_crown_diameter_m:.2f} m</b> (±{metrics.std_crown_diameter_m:.2f} m)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="flora-metric-card">
            <div class="flora-metric-top">
                <div class="flora-metric-lbl">{SVG_ICONS['carbon_kpi']} 04 · STAND CARBON STOCK</div>
                <div class="flora-tooltip-wrap">
                    <span class="flora-info-btn">{SVG_ICONS['info_icon']}</span>
                    <div class="flora-tooltip-card">
                        <b style="color: #111827;">Biomass & Carbon Model</b><br/>
                        • <b>Allometry:</b> <i>Jucker et al. (2016)</i> pantropical model: ln(AGB) = -0.328 + 2.404 × ln(D).<br/>
                        • <b>Carbon Fraction:</b> 47% carbon fraction (<i>IPCC AFOLU 2019 Refinement</i>).<br/>
                        • <b>Uncertainty:</b> Random error 56.5%/√N + 12% systematic floor (<i>VCS Standard v4.5</i>).
                    </div>
                </div>
            </div>
            <div class="flora-metric-val">{metrics.carbon_stock_mg_c:.2f} <span style="font-size: 1rem; font-family: 'Inter'; color: #6B7280; font-weight: normal;">t C</span></div>
            <div class="flora-metric-sub">Confidence: <b>±{results['carbon_uncertainty_mg']:.2f} t C</b> (±{results['stand_uncertainty_pct']:.1f}% · 90% CI)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr style='border: 0; border-top: 1px solid #E5E7EB; margin: 24px 0 20px 0;'/>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 9. Interactive Spatial Delineation (Satellite View)
# -----------------------------------------------------------------------------

st.markdown(
    f"""
    <div style="display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 4px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            {SVG_ICONS['map']}
            <span style="font-size: 1.15rem; font-weight: 600; color: #111827; letter-spacing: -0.01em;">Canopy Delineation</span>
        </div>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.05em;">Sub-meter Optical Survey</span>
    </div>
    <p style='font-size: 0.84rem; color: #6B7280; margin-bottom: 14px; line-height: 1.4;'>
    Optical satellite survey with individual crown contours. Click any crown to inspect stem attributes.
    </p>
    """,
    unsafe_allow_html=True,
)

if selected_preset_key == "sundarbans_mangrove":
    st.markdown(
        f"""
        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 10px 14px; margin-bottom: 14px; font-size: 0.83rem; color: #166534; display: flex; align-items: center; gap: 8px;">
            {SVG_ICONS['info']}
            <span><b>Dense Contiguous Mangrove Reserve:</b> Intertwined crowns form a continuous canopy roof. Total dissolved canopy area is the primary regulatory metric. See sidebar for technical details.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

if results.get("is_geotiff") and results.get("wgs_bounds") is not None:
    min_lon, min_lat, max_lon, max_lat = results["wgs_bounds"]
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0

    # Build Folium map defaulting directly to Esri World Imagery Satellite
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=18,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        control_scale=True,
    )

    # AOI Perimeter (Fine emerald boundary)
    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        color="#22C55E",
        weight=1.5,
        fill=False,
        popup=f"Survey Parcel AOI: {metrics.aoi_area_ha:.2f} ha",
    ).add_to(m)

    # Delineated Trees (Flora Carbon AI Signature Style: Organic Scalloped Polygons + Apical Stem Dots)
    trees_group = folium.FeatureGroup(name=f"Delineated Tree Crowns ({len(df_trees)} stems)", show=True)
    if "lat" in df_trees.columns and "lon" in df_trees.columns:
        for idx, row in df_trees.iterrows():
            cd = float(row.get("crown_diameter_m", 6.0))
            score = float(row.get("score", 0.5))
            popup_html = f"""
            <div style="font-family: 'Inter', sans-serif; font-size: 12px; width: 190px;">
                <b style="color: #165A4C;">Tree Stem #{idx+1}</b><br/>
                <b>Crown Diameter:</b> {cd:.2f} m<br/>
                <b>Crown Area:</b> {np.pi/4 * cd**2:.1f} m²<br/>
                <b>Confidence:</b> {score*100:.1f}%<br/>
                <b>Biomass:</b> {np.exp(-0.328 + 2.404*np.log(max(0.1, cd))):.1f} kg AGB
            </div>
            """
            # 1. Organic Scalloped Crown Contours
            poly_coords = generate_scalloped_crown_polygon(row["lat"], row["lon"], cd / 2.0, seed_idx=idx)
            folium.Polygon(
                locations=poly_coords,
                color="#00FF66",
                weight=1.8,
                fill=True,
                fill_color="#22C55E",
                fill_opacity=0.22,
                popup=folium.Popup(popup_html, max_width=220),
            ).add_to(trees_group)

            # 2. Apical Dominance Stem Centroid (Glowing center dot)
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=2.5,
                color="#FFFFFF",
                weight=0.8,
                fill=True,
                fill_color="#4ADE80",
                fill_opacity=1.0,
                tooltip=f"Stem #{idx+1} · {cd:.1f}m crown",
            ).add_to(trees_group)
    trees_group.add_to(m)

    # Purged Weeds
    if len(df_purged) > 0 and "lat" in df_purged.columns:
        weed_group = folium.FeatureGroup(name=f"Purged Lake Weeds ({len(df_purged)} purged)", show=True)
        for idx, row in df_purged.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6,
                color="#EF4444",
                fill=True,
                fill_color="#EF4444",
                fill_opacity=0.85,
                popup="<b>Purged False Positive</b><br/>Water Hyacinth (Eichhornia crassipes)<br/>Centroid inside water body",
            ).add_to(weed_group)
        weed_group.add_to(m)

    # 3. Flora Carbon AI Signature Glassmorphic Floating HUD Card
    hud = FloatingHud(
        tree_count=len(df_trees),
        canopy_area_ha=metrics.total_calibrated_canopy_area_m2 / 10000.0,
        canopy_area_m2=metrics.total_calibrated_canopy_area_m2,
    )
    m.add_child(hud)

    folium.LayerControl(position="bottomright").add_to(m)
    folium_static(m, width=1100, height=560)

    meta = results["meta"]
    st.markdown(
        f"<div class='flora-disclosure-box'>"
        f"<div style='display: flex; align-items: center; gap: 6px; margin-bottom: 4px;'>{SVG_ICONS['info']}<b style='color: #111827;'>Audit Telemetry</b></div>"
        f"Processed {metrics.tree_count} mature stems across {metrics.aoi_area_ha:.2f} ha in {results['elapsed_sec']:.2f}s on {get_hardware_label()}. "
        f"CRS: <code>{meta.crs}</code> | Resolution: <b>{meta.gsd_meters:.3f} m/pixel</b>."
        f"</div>",
        unsafe_allow_html=True,
    )
else:
    # Non-georeferenced uploaded image or pasted screenshot
    st.markdown("##### **Pasted / Uploaded Optical Image View**")
    if "pil_image" in results:
        pil_img = results["pil_image"]
        annotated_np = np.array(pil_img.copy())
        for _, row in df_trees.iterrows():
            x1, y1 = int(row["xmin"]), int(row["ymin"])
            x2, y2 = int(row["xmax"]), int(row["ymax"])
            # High-contrast Emerald Green box in RGB
            cv2.rectangle(annotated_np, (x1, y1), (x2, y2), (34, 197, 94), 2)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            # Scarlet blossom center dot
            cv2.circle(annotated_np, (cx, cy), 3, (220, 38, 38), -1)

        st.image(
            annotated_np,
            caption=f"Delineated Detections: {metrics.tree_count} Individual Crowns across {metrics.aoi_area_ha:.2f} ha (Assumed GSD: {custom_gsd*100:.1f} cm/px)",
            use_container_width=True,
        )

        pyramid_status = "Active (Dual-resolution 1.0x + 2.0x pyramid)" if results.get("used_multi_scale", False) else "Single-Scale Native"
        st.markdown(
            f"<div class='flora-disclosure-box'>"
            f"<div style='display: flex; align-items: center; gap: 6px; margin-bottom: 4px;'>{SVG_ICONS['info']}<b style='color: #111827;'>Adaptive Multi-Scale Telemetry</b></div>"
            f"Delineated <b>{metrics.tree_count} individual crowns</b> across <b>{metrics.aoi_area_ha:.2f} ha</b> in {results['elapsed_sec']:.2f}s on {get_hardware_label()}.<br/>"
            f"Feature Pyramid: <b>{pyramid_status}</b> · Effective Receptive Field: <b>~0.25 m/pixel</b>."
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("<hr style='border: 0; border-top: 1px solid #E5E7EB; margin: 28px 0 20px 0;'/>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 10. Canopy Architecture & Structural Analytics
# -----------------------------------------------------------------------------

st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
        {SVG_ICONS['bar_chart']}
        <span style="font-size: 1.15rem; font-weight: 600; color: #111827;">Canopy Architecture & Structural Analytics</span>
    </div>
    """,
    unsafe_allow_html=True,
)

an_c1, an_c2 = st.columns([1.5, 1])

with an_c1:
    if "crown_diameter_m" in df_trees.columns and len(df_trees) > 0:
        diameters = df_trees["crown_diameter_m"].values
        bins = [0, 3, 6, 9, 12, 15, 20, 30]
        counts, _ = np.histogram(diameters, bins=bins)
        bin_labels = ["<3m (Sapling)", "3-6m (Pole)", "6-9m (Mature)", "9-12m (Large)", "12-15m (Overstory)", "15-20m (Emergent)", ">20m (Heritage)"]
        
        hist_df = pd.DataFrame({"Size Class": bin_labels[:len(counts)], "Stem Count": counts})
        st.markdown("##### **Stem Count by Crown Size Class**")
        st.bar_chart(hist_df.set_index("Size Class"), color="#165A4C")

with an_c2:
    st.markdown("##### **Canopy & Biomass Ledger**")
    st.markdown(
        f"""
        <div class="clover-breakdown-card">
            <div class="clover-breakdown-header">
                <span class="clover-breakdown-title">PARCEL BREAKDOWN & AUDIT TRAIL</span>
                <div class="flora-tooltip-wrap">
                    <span class="flora-info-btn">{SVG_ICONS['info_icon']}</span>
                    <div class="flora-tooltip-card" style="width: 295px;">
                        <b style="color: #111827; font-size: 0.8rem;">Verra VM0047 · Tier 2 MRV Standard</b><br/>
                        • <b>What is VM0047?</b> The leading global carbon crediting standard for afforestation, reforestation, and revegetation (ARR) projects.<br/>
                        • <b>Zero Double-Counting:</b> Overlapping crowns are dissolved using computational geometry so overlapping canopies are never counted twice.<br/>
                        • <b>Edge Calibration:</b> +20% crown bias calibration applied per <i>Li et al. (PNAS Nexus 2023)</i>.<br/>
                        • <b>Biomass Allometry:</b> Living tree mass derived from crown diameter using pantropical allometry (<i>Jucker et al. 2016</i>).<br/>
                        • <b>Carbon Fraction:</b> 47% carbon content conversion with ±35% confidence bounds (<i>IPCC AFOLU 2019 Refinement</i>).
                    </div>
                </div>
            </div>
            <div class="clover-breakdown-item">
                <div>
                    <div class="clover-item-label">Surveyed Land Area</div>
                    <div class="clover-item-sub">Total parcel boundary analyzed</div>
                </div>
                <div class="clover-item-val">
                    <span class="clover-val-main">{metrics.aoi_area_ha:.2f} ha</span>
                    <span class="clover-val-sub">{metrics.aoi_area_m2:,.0f} m²</span>
                </div>
            </div>
            <div class="clover-breakdown-item">
                <div>
                    <div class="clover-item-label">True Ground Canopy Cover</div>
                    <div class="clover-item-sub">Ground shaded by tree leaves (calibrated)</div>
                </div>
                <div class="clover-item-val">
                    <span class="clover-val-main" style="color: #166534;">{metrics.canopy_cover_pct_calibrated:.2f}%</span>
                    <span class="clover-val-sub">{metrics.total_calibrated_canopy_area_m2:,.1f} m²</span>
                </div>
            </div>
            <div class="clover-breakdown-item">
                <div>
                    <div class="clover-item-label">Overlapping Foliage Deducted</div>
                    <div class="clover-item-sub">Prevents double-counting crowns</div>
                </div>
                <div class="clover-item-val">
                    <span class="clover-val-main">{metrics.crown_overlap_pct:.1f}%</span>
                    <span class="clover-val-sub">-{metrics.crown_overlap_area_m2:,.1f} m²</span>
                </div>
            </div>
            <div class="clover-breakdown-item">
                <div>
                    <div class="clover-item-label">Living Wood Biomass</div>
                    <div class="clover-item-sub">Dry tree mass (Jucker et al. allometry)</div>
                </div>
                <div class="clover-item-val">
                    <span class="clover-val-main">{metrics.agb_total_mg:.2f} tonnes</span>
                    <span class="clover-val-sub">{metrics.agb_total_mg:.2f} Mg AGB</span>
                </div>
            </div>
            <div class="clover-breakdown-item" style="border-bottom: none;">
                <div>
                    <div class="clover-item-label">Stored Carbon (In-Wood)</div>
                    <div class="clover-item-sub">IPCC Tier 2 carbon fraction (47% biomass)</div>
                </div>
                <div class="clover-item-val">
                    <span class="clover-val-main" style="color: #166534;">{metrics.carbon_stock_mg_c:.2f} tonnes C</span>
                    <span class="clover-val-sub">±{results['carbon_uncertainty_mg']:.2f} t (95% CI)</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr style='border: 0; border-top: 1px solid #E5E7EB; margin: 32px 0 16px 0;'/>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; font-size: 0.85rem; color: #6B7280; padding: 12px 0 24px 0;">
        made by momo :)
    </div>
    """,
    unsafe_allow_html=True,
)
