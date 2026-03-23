#!/usr/bin/env python3
"""
astro_reduce.py

A practical FITS reduction pipeline for:
- bias
- dark
- flat
- flat-dark

Features
--------
1. Scans a directory tree of FITS files.
2. Classifies files using FITS headers and/or filename parsing.
3. Groups light frames by subject/date/filter/binning/temperature.
4. Writes a per-project descriptor JSON file.
5. Builds master bias, dark, flat-dark, and flat frames.
6. Calibrates light frames.
7. Saves calibrated frames and a stacked frame per project.

Notes
-----
- Matching is based on:
    * binning
    * filter
    * sensor temperature (within tolerance)
    * exposure time for dark/flat-dark matching
- Bias frames generally do not depend on filter, but the script can prefer same filter if present.
- Dark frames do not normally depend on filter, but some capture software includes filter in filenames;
  this script does not require filter matching for darks by default.
- Flat frames are matched strongly by filter and binning, and optionally by temperature.
- If exact exposure matches are not found for darks/flat-darks, the nearest exposure match is used.
- Calibration order:
    light_cal = (light - master_bias - scaled/master_dark)
    flat_cal  = (flat  - master_bias - master_flatdark_or_dark)
    normalized_flat = flat_cal / median(flat_cal)
    final = light_cal / normalized_flat
- Stacking uses sigma-clipped mean.

Example
-------
python astro_reduce.py /data/astro \
    --output /data/reduced \
    --temp-tol 3.0 \
    --stack-method sigma_clip_mean

Expected FITS headers if available
----------------------------------
OBJECT, IMAGETYP/FRAME/OBSTYPE, FILTER, EXPTIME, CCD-TEMP/SET-TEMP/TEMP,
XBINNING, YBINNING, DATE-OBS

Filename parsing fallback
-------------------------
The parser tries to extract likely tokens such as:
- subject/object
- frame type
- filter
- binning (e.g. 1x1, 2x2, bin1, bin2)
- temperature (e.g. -10C, temp-10)
- index
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.time import Time
from ccdproc import CCDData, Combiner
import astropy.units as u


# ----------------------------
# Data models
# ----------------------------

@dataclass
class FrameInfo:
    path: str
    frame_type: str                 # light, bias, dark, flat, flatdark
    subject: str
    filter_name: Optional[str]
    xbin: int
    ybin: int
    temperature_c: Optional[float]
    exposure_s: Optional[float]
    date_obs: Optional[str]
    topic: Optional[str] = None
    date_folder: Optional[str] = None
    image_type_folder: Optional[str] = None
    header_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectDescriptor:
    project_id: str
    subject: str
    filter_name: Optional[str]
    xbin: int
    ybin: int
    temperature_c: Optional[float]
    date_obs_min: Optional[str]
    date_obs_max: Optional[str]

    light_files: List[str] = field(default_factory=list)

    candidate_bias_files: List[str] = field(default_factory=list)
    candidate_dark_files: List[str] = field(default_factory=list)
    candidate_flat_files: List[str] = field(default_factory=list)
    candidate_flatdark_files: List[str] = field(default_factory=list)

    selected_master_bias: Optional[str] = None
    selected_master_dark: Optional[str] = None
    selected_master_flatdark: Optional[str] = None
    selected_master_flat: Optional[str] = None

    notes: List[str] = field(default_factory=list)


# ----------------------------
# Helpers: filename/header parsing
# ----------------------------

FRAME_TYPE_SYNONYMS = {
    "light": {"light", "light frame", "object", "science", "image", "img"},
    "bias": {"bias", "offset"},
    "dark": {"dark"},
    "flat": {"flat", "flat field"},
    "flatdark": {"flatdark", "flat-dark", "darkflat", "dark-flat", "flat dark"},
}

FILTER_ALIASES = {
    "l": "L",
    "lum": "L",
    "luminance": "L",
    "r": "R",
    "g": "G",
    "b": "B",
    "ha": "Ha",
    "halpha": "Ha",
    "o3": "OIII",
    "oiii": "OIII",
    "s2": "SII",
    "sii": "SII",
    "clear": "Clear",
}


def canonicalize_filter(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = str(name).strip().lower()
    return FILTER_ALIASES.get(s, str(name).strip())


def normalize_frame_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = str(value).strip().lower()
    for canon, vals in FRAME_TYPE_SYNONYMS.items():
        if s in vals:
            return canon
    for canon in FRAME_TYPE_SYNONYMS:
        if canon in s.replace("_", " ").replace("-", " "):
            return canon
    if "object" in s or "science" in s:
        return "light"
    return None


def safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def safe_int(v: Any, default: int = 1) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def parse_binning_from_text(text: str) -> Tuple[Optional[int], Optional[int]]:
    s = text.lower()
    m = re.search(r'(\d)\s*x\s*(\d)', s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r'bin(?:ning)?[_\- ]?(\d)', s)
    if m:
        b = int(m.group(1))
        return b, b
    return None, None


def parse_temperature_from_text(text: str) -> Optional[float]:
    s = text.lower()
    patterns = [
        r'temp[_\- ]?(-?\d+(?:\.\d+)?)c?',
        r'(-?\d+(?:\.\d+)?)c',
        r'ccd[_\- ]?temp[_\- ]?(-?\d+(?:\.\d+)?)',
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return safe_float(m.group(1))
    return None


def parse_exposure_from_text(text: str) -> Optional[float]:
    s = text.lower()
    patterns = [
        r'(\d+(?:\.\d+)?)s(?:ec)?',
        r'exp[_\- ]?(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)sec',
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return safe_float(m.group(1))
    return None


def parse_filter_from_text(text: str) -> Optional[str]:
    s = text.lower()
    tokens = re.split(r'[_\-\s\.]+', s)
    for t in tokens:
        if t in FILTER_ALIASES:
            return canonicalize_filter(t)
    for k, v in FILTER_ALIASES.items():
        if re.search(rf'(^|[_\-\s]){re.escape(k)}([_\-\s]|$)', s):
            return v
    return None


def parse_frame_type_from_text(text: str) -> Optional[str]:
    s = text.lower().replace("_", " ").replace("-", " ")
    # check flatdark before dark/flat to avoid collisions
    if any(k in s for k in ["flatdark", "flat dark", "darkflat", "dark flat"]):
        return "flatdark"
    for canon, vals in FRAME_TYPE_SYNONYMS.items():
        for v in vals:
            if v in s:
                return canon
    return None


def parse_subject_from_text(text: str, frame_type: str) -> str:
    stem = Path(text).stem
    s = re.sub(r'\.fits?$', '', stem, flags=re.IGNORECASE)
    s = re.sub(r'[_\-]+', ' ', s)

    removable = [
        "light", "object", "science", "image", "img",
        "bias", "offset", "dark", "flat", "flatdark", "flat dark",
        "luminance", "lum", "clear", "ha", "halpha", "oiii", "o3", "sii", "s2",
    ]
    for token in removable:
        s = re.sub(rf'\b{re.escape(token)}\b', ' ', s, flags=re.IGNORECASE)

    s = re.sub(r'\b\d+x\d+\b', ' ', s, flags=re.IGNORECASE)
    s = re.sub(r'\bbin(?:ning)?\s*\d+\b', ' ', s, flags=re.IGNORECASE)
    s = re.sub(r'\btemp\s*-?\d+(\.\d+)?c?\b', ' ', s, flags=re.IGNORECASE)
    s = re.sub(r'\b-?\d+(\.\d+)?c\b', ' ', s, flags=re.IGNORECASE)
    s = re.sub(r'\b\d+(\.\d+)?s(ec)?\b', ' ', s, flags=re.IGNORECASE)
    s = re.sub(r'\b\d+\b', ' ', s)

    s = re.sub(r'\s+', ' ', s).strip()
    if frame_type in {"bias", "dark", "flat", "flatdark"}:
        return frame_type.upper()
    return s if s else "UNKNOWN_SUBJECT"


def get_header_value(header: fits.Header, keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in header:
            return header[k]
    return default


def read_frame_info(path: Path) -> Optional[FrameInfo]:
    try:
        with fits.open(path, memmap=False) as hdul:
            hdr = hdul[0].header

        frame_type = normalize_frame_type(
            get_header_value(hdr, ["IMAGETYP", "FRAME", "OBSTYPE", "IMGTYPE"])
        )
        if frame_type is None:
            frame_type = parse_frame_type_from_text(path.name)
        if frame_type is None:
            frame_type = "light"

        subject = get_header_value(hdr, ["OBJECT", "OBJNAME"], None)
        if subject is None:
            subject = parse_subject_from_text(path.name, frame_type)

        filter_name = canonicalize_filter(get_header_value(hdr, ["FILTER", "FILTNAM"], None))
        if filter_name is None:
            filter_name = parse_filter_from_text(path.name)

        xbin = safe_int(get_header_value(hdr, ["XBINNING", "XBIN"], None), default=1)
        ybin = safe_int(get_header_value(hdr, ["YBINNING", "YBIN"], None), default=xbin)
        if xbin == 1 and ybin == 1:
            fx, fy = parse_binning_from_text(path.name)
            if fx is not None:
                xbin, ybin = fx, fy

        temp = safe_float(get_header_value(hdr, ["CCD-TEMP", "CCDTEMP", "SET-TEMP", "TEMP"], None))
        if temp is None:
            temp = parse_temperature_from_text(path.name)

        exptime = safe_float(get_header_value(hdr, ["EXPTIME", "EXPOSURE"], None))
        if exptime is None:
            exptime = parse_exposure_from_text(path.name)

        date_obs = get_header_value(hdr, ["DATE-OBS", "DATE"], None)
        if date_obs:
            try:
                date_obs = Time(date_obs, format='isot', scale='utc').isot
            except Exception:
                try:
                    date_obs = Time(date_obs).isot
                except Exception:
                    date_obs = str(date_obs)

        parts = path.parts
        topic = parts[-4] if len(parts) >= 4 else None
        date_folder = parts[-3] if len(parts) >= 3 else None
        image_type_folder = parts[-2] if len(parts) >= 2 else None

        return FrameInfo(
            path=str(path),
            frame_type=frame_type,
            subject=str(subject).strip() if subject else "UNKNOWN_SUBJECT",
            filter_name=filter_name,
            xbin=xbin,
            ybin=ybin,
            temperature_c=temp,
            exposure_s=exptime,
            date_obs=date_obs,
            topic=topic,
            date_folder=date_folder,
            image_type_folder=image_type_folder,
            header_summary={
                "OBJECT": get_header_value(hdr, ["OBJECT", "OBJNAME"], None),
                "IMAGETYP": get_header_value(hdr, ["IMAGETYP", "FRAME", "OBSTYPE", "IMGTYPE"], None),
                "FILTER": get_header_value(hdr, ["FILTER", "FILTNAM"], None),
                "XBINNING": get_header_value(hdr, ["XBINNING", "XBIN"], None),
                "YBINNING": get_header_value(hdr, ["YBINNING", "YBIN"], None),
                "CCD-TEMP": get_header_value(hdr, ["CCD-TEMP", "CCDTEMP", "SET-TEMP", "TEMP"], None),
                "EXPTIME": get_header_value(hdr, ["EXPTIME", "EXPOSURE"], None),
                "DATE-OBS": get_header_value(hdr, ["DATE-OBS", "DATE"], None),
            },
        )
    except Exception as e:
        print(f"[WARN] Could not read FITS header: {path} ({e})", file=sys.stderr)
        return None


# ----------------------------
# Directory scan and grouping
# ----------------------------

def scan_fits_tree(root: Path) -> List[FrameInfo]:
    exts = {".fits", ".fit", ".fts"}
    frames: List[FrameInfo] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            fi = read_frame_info(p)
            if fi is not None:
                frames.append(fi)
    return frames


def temperature_close(a: Optional[float], b: Optional[float], tol: float) -> bool:
    if a is None or b is None:
        return True
    return abs(a - b) <= tol


def exposure_close(a: Optional[float], b: Optional[float], tol: float = 0.5) -> bool:
    if a is None or b is None:
        return True
    return abs(a - b) <= tol


def sort_by_path(frames: List[FrameInfo]) -> List[FrameInfo]:
    return sorted(frames, key=lambda x: x.path)


def group_light_projects(frames: List[FrameInfo]) -> Dict[str, List[FrameInfo]]:
    """
    Group light frames into project buckets by:
    subject + filter + binning

    Date is preserved inside the descriptor rather than splitting too aggressively.
    """
    projects = defaultdict(list)
    for fi in frames:
        if fi.frame_type != "light":
            continue
        key = f"{fi.subject}__{fi.filter_name or 'NOFILTER'}__{fi.xbin}x{fi.ybin}"
        projects[key].append(fi)
    return dict(projects)


# ----------------------------
# Calibration matching
# ----------------------------

def median_temperature(frames: List[FrameInfo]) -> Optional[float]:
    vals = [f.temperature_c for f in frames if f.temperature_c is not None]
    if not vals:
        return None
    return float(np.median(vals))


def minmax_date(frames: List[FrameInfo]) -> Tuple[Optional[str], Optional[str]]:
    vals = [f.date_obs for f in frames if f.date_obs]
    if not vals:
        return None, None
    return min(vals), max(vals)


def select_candidate_biases(
    lights: List[FrameInfo],
    all_biases: List[FrameInfo],
    temp_tol: float
) -> List[FrameInfo]:
    if not lights:
        return []
    xbin, ybin = lights[0].xbin, lights[0].ybin
    filt = lights[0].filter_name
    temp = median_temperature(lights)

    candidates = [
        b for b in all_biases
        if b.xbin == xbin and b.ybin == ybin and temperature_close(b.temperature_c, temp, temp_tol)
    ]

    same_filter = [b for b in candidates if b.filter_name == filt]
    if same_filter:
        return sort_by_path(same_filter)
    return sort_by_path(candidates)


def select_candidate_darks(
    lights: List[FrameInfo],
    all_darks: List[FrameInfo],
    temp_tol: float,
    exp_tol: float = 0.5,
) -> List[FrameInfo]:
    if not lights:
        return []
    xbin, ybin = lights[0].xbin, lights[0].ybin

    unique_light_exposures = sorted({round(f.exposure_s or -1, 3) for f in lights if f.exposure_s is not None})
    light_temp = median_temperature(lights)

    candidates = [
        d for d in all_darks
        if d.xbin == xbin and d.ybin == ybin and temperature_close(d.temperature_c, light_temp, temp_tol)
    ]

    selected = []
    for exp in unique_light_exposures:
        exact = [d for d in candidates if exposure_close(d.exposure_s, exp, exp_tol)]
        if exact:
            selected.extend(exact)
        else:
            nearest = sorted(
                [d for d in candidates if d.exposure_s is not None],
                key=lambda d: abs((d.exposure_s or 0) - exp)
            )[:10]
            selected.extend(nearest)

    uniq = {x.path: x for x in selected}
    return sort_by_path(list(uniq.values()))


def select_candidate_flats(
    lights: List[FrameInfo],
    all_flats: List[FrameInfo],
    temp_tol: float
) -> List[FrameInfo]:
    if not lights:
        return []
    xbin, ybin = lights[0].xbin, lights[0].ybin
    filt = lights[0].filter_name
    temp = median_temperature(lights)

    candidates = [
        f for f in all_flats
        if f.xbin == xbin and f.ybin == ybin and f.filter_name == filt
           and temperature_close(f.temperature_c, temp, temp_tol)
    ]
    return sort_by_path(candidates)


def select_candidate_flatdarks(
    flats: List[FrameInfo],
    all_flatdarks: List[FrameInfo],
    temp_tol: float,
    exp_tol: float = 0.5,
) -> List[FrameInfo]:
    if not flats:
        return []
    xbin, ybin = flats[0].xbin, flats[0].ybin
    temp = median_temperature(flats)
    unique_flat_exposures = sorted({round(f.exposure_s or -1, 3) for f in flats if f.exposure_s is not None})

    candidates = [
        fd for fd in all_flatdarks
        if fd.xbin == xbin and fd.ybin == ybin and temperature_close(fd.temperature_c, temp, temp_tol)
    ]

    selected = []
    for exp in unique_flat_exposures:
        exact = [fd for fd in candidates if exposure_close(fd.exposure_s, exp, exp_tol)]
        if exact:
            selected.extend(exact)
        else:
            nearest = sorted(
                [fd for fd in candidates if fd.exposure_s is not None],
                key=lambda d: abs((d.exposure_s or 0) - exp)
            )[:10]
            selected.extend(nearest)

    uniq = {x.path: x for x in selected}
    return sort_by_path(list(uniq.values()))


def build_project_descriptors(
    frames: List[FrameInfo],
    output_dir: Path,
    temp_tol: float,
) -> List[ProjectDescriptor]:
    lights = [f for f in frames if f.frame_type == "light"]
    biases = [f for f in frames if f.frame_type == "bias"]
    darks = [f for f in frames if f.frame_type == "dark"]
    flats = [f for f in frames if f.frame_type == "flat"]
    flatdarks = [f for f in frames if f.frame_type == "flatdark"]

    descriptors: List[ProjectDescriptor] = []

    grouped = group_light_projects(lights)
    for project_key, project_lights in grouped.items():
        project_lights = sort_by_path(project_lights)
        subject = project_lights[0].subject
        filt = project_lights[0].filter_name
        xbin, ybin = project_lights[0].xbin, project_lights[0].ybin
        dmin, dmax = minmax_date(project_lights)
        tmed = median_temperature(project_lights)

        project_id = re.sub(r'[^A-Za-z0-9_.-]+', '_', f"{subject}_{filt or 'NOFILTER'}_{xbin}x{ybin}")
        desc = ProjectDescriptor(
            project_id=project_id,
            subject=subject,
            filter_name=filt,
            xbin=xbin,
            ybin=ybin,
            temperature_c=tmed,
            date_obs_min=dmin,
            date_obs_max=dmax,
            light_files=[f.path for f in project_lights],
        )

        cbias = select_candidate_biases(project_lights, biases, temp_tol)
        cdark = select_candidate_darks(project_lights, darks, temp_tol)
        cflat = select_candidate_flats(project_lights, flats, temp_tol)
        cflatdark = select_candidate_flatdarks(cflat, flatdarks, temp_tol)

        desc.candidate_bias_files = [f.path for f in cbias]
        desc.candidate_dark_files = [f.path for f in cdark]
        desc.candidate_flat_files = [f.path for f in cflat]
        desc.candidate_flatdark_files = [f.path for f in cflatdark]

        if not cbias:
            desc.notes.append("No candidate bias frames found.")
        if not cdark:
            desc.notes.append("No candidate dark frames found.")
        if not cflat:
            desc.notes.append("No candidate flat frames found.")
        if cflat and not cflatdark:
            desc.notes.append("No flat-dark frames found; dark or bias-only flat correction may be used.")

        descriptors.append(desc)

        proj_dir = output_dir / "projects" / project_id
        proj_dir.mkdir(parents=True, exist_ok=True)
        with open(proj_dir / "project_descriptor.json", "w", encoding="utf-8") as f:
            json.dump(asdict(desc), f, indent=2)

    return descriptors


# ----------------------------
# FITS / CCD helpers
# ----------------------------

def load_ccd(path: str) -> CCDData:
    with fits.open(path, memmap=False) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data.astype(np.float32)
    return CCDData(data, unit="adu", meta=hdr)


def combine_ccds(
    file_paths: List[str],
    method: str = "median",
    sigma_clip: bool = True
) -> CCDData:
    ccds = [load_ccd(p) for p in file_paths]
    comb = Combiner(ccds)

    if sigma_clip:
        comb.sigma_clipping(
            low_thresh=3.0,
            high_thresh=3.0,
            func=np.ma.median
        )

    if method == "median":
        master = comb.median_combine()
    elif method == "average":
        master = comb.average_combine()
    else:
        raise ValueError(f"Unsupported combine method: {method}")

    return master


def write_ccd(ccd: CCDData, path: Path, extra_header: Optional[Dict[str, Any]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = ccd.meta.copy()
    if extra_header:
        for k, v in extra_header.items():
            try:
                hdr[k] = v
            except Exception:
                pass
    fits.writeto(path, np.asarray(ccd.data, dtype=np.float32), header=hdr, overwrite=True)


def normalize_flat(ccd: CCDData) -> CCDData:
    data = np.asarray(ccd.data, dtype=np.float32)
    med = np.nanmedian(data[np.isfinite(data)])
    if not np.isfinite(med) or med == 0:
        raise RuntimeError("Cannot normalize flat: invalid median.")
    out = ccd.copy()
    out.data = data / med
    out.meta["NORMFLAT"] = True
    out.meta["FLATMED"] = float(med)
    return out


# ----------------------------
# Master selection/building
# ----------------------------

def exposure_groups(frames: List[FrameInfo], tol: float = 0.5) -> Dict[float, List[FrameInfo]]:
    groups: Dict[float, List[FrameInfo]] = {}
    for f in frames:
        if f.exposure_s is None:
            continue
        placed = False
        for k in list(groups.keys()):
            if abs(k - f.exposure_s) <= tol:
                groups[k].append(f)
                placed = True
                break
        if not placed:
            groups[f.exposure_s] = [f]
    return groups


def nearest_exposure_group(frames: List[FrameInfo], target_exp: Optional[float], tol: float = 0.5) -> List[FrameInfo]:
    if target_exp is None:
        return frames
    groups = exposure_groups(frames, tol=tol)
    if not groups:
        return []
    best_key = min(groups.keys(), key=lambda k: abs(k - target_exp))
    return groups[best_key]


def build_master_bias(
    bias_files: List[str],
    out_path: Path
) -> Optional[Path]:
    if not bias_files:
        return None
    master = combine_ccds(bias_files, method="median", sigma_clip=True)
    master.meta["IMAGETYP"] = "MASTER_BIAS"
    master.meta["NCOMBINE"] = len(bias_files)
    write_ccd(master, out_path)
    return out_path


def build_master_dark(
    dark_files: List[str],
    out_path: Path,
    master_bias_path: Optional[Path] = None
) -> Optional[Path]:
    if not dark_files:
        return None

    dark_ccds = [load_ccd(p) for p in dark_files]
    if master_bias_path and master_bias_path.exists():
        mb = load_ccd(str(master_bias_path))
        for ccd in dark_ccds:
            ccd.data = np.asarray(ccd.data, dtype=np.float32) - np.asarray(mb.data, dtype=np.float32)

    comb = Combiner(dark_ccds)
    comb.sigma_clipping(low_thresh=3.0, high_thresh=3.0, func=np.ma.median)
    master = comb.median_combine()
    master.meta["IMAGETYP"] = "MASTER_DARK"
    master.meta["NCOMBINE"] = len(dark_files)

    exps = []
    for p in dark_files:
        fi = read_frame_info(Path(p))
        if fi and fi.exposure_s is not None:
            exps.append(fi.exposure_s)
    if exps:
        master.meta["EXPTIME"] = float(np.median(exps))

    write_ccd(master, out_path)
    return out_path


def build_master_flatdark(
    flatdark_files: List[str],
    out_path: Path,
    master_bias_path: Optional[Path] = None
) -> Optional[Path]:
    if not flatdark_files:
        return None

    fd_ccds = [load_ccd(p) for p in flatdark_files]
    if master_bias_path and master_bias_path.exists():
        mb = load_ccd(str(master_bias_path))
        for ccd in fd_ccds:
            ccd.data = np.asarray(ccd.data, dtype=np.float32) - np.asarray(mb.data, dtype=np.float32)

    comb = Combiner(fd_ccds)
    comb.sigma_clipping(low_thresh=3.0, high_thresh=3.0, func=np.ma.median)
    master = comb.median_combine()
    master.meta["IMAGETYP"] = "MASTER_FLATDARK"
    master.meta["NCOMBINE"] = len(flatdark_files)

    exps = []
    for p in flatdark_files:
        fi = read_frame_info(Path(p))
        if fi and fi.exposure_s is not None:
            exps.append(fi.exposure_s)
    if exps:
        master.meta["EXPTIME"] = float(np.median(exps))

    write_ccd(master, out_path)
    return out_path


def build_master_flat(
    flat_files: List[str],
    out_path: Path,
    master_bias_path: Optional[Path] = None,
    master_dark_or_flatdark_path: Optional[Path] = None,
) -> Optional[Path]:
    if not flat_files:
        return None

    mb = load_ccd(str(master_bias_path)) if master_bias_path and master_bias_path.exists() else None
    md = load_ccd(str(master_dark_or_flatdark_path)) if master_dark_or_flatdark_path and master_dark_or_flatdark_path.exists() else None

    flat_ccds: List[CCDData] = []
    for p in flat_files:
        ccd = load_ccd(p)
        data = np.asarray(ccd.data, dtype=np.float32)

        if mb is not None:
            data = data - np.asarray(mb.data, dtype=np.float32)

        if md is not None:
            md_data = np.asarray(md.data, dtype=np.float32)
            flat_info = read_frame_info(Path(p))
            md_exp = safe_float(md.meta.get("EXPTIME", None))
            flat_exp = flat_info.exposure_s if flat_info else None

            if md_exp and flat_exp and md_exp > 0:
                scale = flat_exp / md_exp
            else:
                scale = 1.0
            data = data - md_data * scale

        ccd.data = data
        flat_ccds.append(ccd)

    comb = Combiner(flat_ccds)
    comb.sigma_clipping(low_thresh=3.0, high_thresh=3.0, func=np.ma.median)
    master = comb.median_combine()
    master = normalize_flat(master)
    master.meta["IMAGETYP"] = "MASTER_FLAT"
    master.meta["NCOMBINE"] = len(flat_files)
    write_ccd(master, out_path)
    return out_path


# ----------------------------
# Calibration and stacking
# ----------------------------

def calibrate_light(
    light_path: str,
    out_path: Path,
    master_bias_path: Optional[Path],
    master_dark_path: Optional[Path],
    master_flat_path: Optional[Path],
) -> Path:
    light = load_ccd(light_path)
    data = np.asarray(light.data, dtype=np.float32)

    li = read_frame_info(Path(light_path))
    light_exp = li.exposure_s if li else None

    if master_bias_path and master_bias_path.exists():
        mb = load_ccd(str(master_bias_path))
        data = data - np.asarray(mb.data, dtype=np.float32)

    if master_dark_path and master_dark_path.exists():
        md = load_ccd(str(master_dark_path))
        md_data = np.asarray(md.data, dtype=np.float32)
        md_exp = safe_float(md.meta.get("EXPTIME", None))
        if md_exp and light_exp and md_exp > 0:
            scale = light_exp / md_exp
        else:
            scale = 1.0
        data = data - md_data * scale

    if master_flat_path and master_flat_path.exists():
        mf = load_ccd(str(master_flat_path))
        flat_data = np.asarray(mf.data, dtype=np.float32)
        good = np.isfinite(flat_data) & (flat_data > 0)
        out = np.zeros_like(data, dtype=np.float32)
        out[good] = data[good] / flat_data[good]
        out[~good] = 0.0
        data = out

    light.data = data
    light.meta["CALIBRAT"] = True
    light.meta["HISTORY"] = "Bias, dark, and flat calibration applied."

    write_ccd(light, out_path)
    return out_path


def stack_calibrated_images(
    calibrated_paths: List[Path],
    out_path: Path,
    method: str = "sigma_clip_mean"
) -> Optional[Path]:
    if not calibrated_paths:
        return None

    ccds = [load_ccd(str(p)) for p in calibrated_paths]
    comb = Combiner(ccds)

    if method == "sigma_clip_mean":
        comb.sigma_clipping(low_thresh=3.0, high_thresh=3.0, func=np.ma.median)
        stacked = comb.average_combine()
    elif method == "median":
        comb.sigma_clipping(low_thresh=3.0, high_thresh=3.0, func=np.ma.median)
        stacked = comb.median_combine()
    elif method == "mean":
        stacked = comb.average_combine()
    else:
        raise ValueError(f"Unknown stack method: {method}")

    stacked.meta["IMAGETYP"] = "STACKED_LIGHT"
    stacked.meta["NCOMBINE"] = len(calibrated_paths)
    write_ccd(stacked, out_path)
    return out_path


# ----------------------------
# Project processing
# ----------------------------

def load_descriptor(path: Path) -> ProjectDescriptor:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return ProjectDescriptor(**d)


def update_descriptor_file(path: Path, desc: ProjectDescriptor) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(desc), f, indent=2)


def process_project(
    descriptor_path: Path,
    output_root: Path,
    temp_tol: float,
    stack_method: str,
) -> None:
    desc = load_descriptor(descriptor_path)
    project_dir = descriptor_path.parent
    masters_dir = project_dir / "masters"
    calibrated_dir = project_dir / "calibrated"
    stacked_dir = project_dir / "stacked"

    masters_dir.mkdir(parents=True, exist_ok=True)
    calibrated_dir.mkdir(parents=True, exist_ok=True)
    stacked_dir.mkdir(parents=True, exist_ok=True)

    # Rehydrate candidate frame infos for exposure-aware subset selection
    bias_infos = [read_frame_info(Path(p)) for p in desc.candidate_bias_files]
    bias_infos = [x for x in bias_infos if x is not None]

    dark_infos = [read_frame_info(Path(p)) for p in desc.candidate_dark_files]
    dark_infos = [x for x in dark_infos if x is not None]

    flat_infos = [read_frame_info(Path(p)) for p in desc.candidate_flat_files]
    flat_infos = [x for x in flat_infos if x is not None]

    flatdark_infos = [read_frame_info(Path(p)) for p in desc.candidate_flatdark_files]
    flatdark_infos = [x for x in flatdark_infos if x is not None]

    # Build master bias
    master_bias_path = None
    if bias_infos:
        master_bias_path = masters_dir / "master_bias.fits"
        build_master_bias([x.path for x in bias_infos], master_bias_path)
        desc.selected_master_bias = str(master_bias_path)

    # Build one master dark for the dominant light exposure, or nearest available
    light_infos = [read_frame_info(Path(p)) for p in desc.light_files]
    light_infos = [x for x in light_infos if x is not None]

    light_exps = [x.exposure_s for x in light_infos if x.exposure_s is not None]
    dominant_light_exp = float(np.median(light_exps)) if light_exps else None

    selected_dark_infos = nearest_exposure_group(dark_infos, dominant_light_exp, tol=0.5)
    master_dark_path = None
    if selected_dark_infos:
        master_dark_path = masters_dir / "master_dark.fits"
        build_master_dark(
            [x.path for x in selected_dark_infos],
            master_dark_path,
            master_bias_path=master_bias_path,
        )
        desc.selected_master_dark = str(master_dark_path)

    # Build master flat-dark matched to flat exposure, else allow dark fallback
    flat_exps = [x.exposure_s for x in flat_infos if x.exposure_s is not None]
    dominant_flat_exp = float(np.median(flat_exps)) if flat_exps else None

    selected_flatdark_infos = nearest_exposure_group(flatdark_infos, dominant_flat_exp, tol=0.5)
    master_flatdark_path = None
    if selected_flatdark_infos:
        master_flatdark_path = masters_dir / "master_flatdark.fits"
        build_master_flatdark(
            [x.path for x in selected_flatdark_infos],
            master_flatdark_path,
            master_bias_path=master_bias_path,
        )
        desc.selected_master_flatdark = str(master_flatdark_path)

    # Build master flat
    master_flat_path = None
    if flat_infos:
        master_flat_path = masters_dir / "master_flat.fits"
        dark_for_flat = master_flatdark_path if master_flatdark_path else master_dark_path
        build_master_flat(
            [x.path for x in flat_infos],
            master_flat_path,
            master_bias_path=master_bias_path,
            master_dark_or_flatdark_path=dark_for_flat,
        )
        desc.selected_master_flat = str(master_flat_path)

    # Calibrate lights
    calibrated_paths: List[Path] = []
    for lp in desc.light_files:
        src = Path(lp)
        out = calibrated_dir / f"{src.stem}_cal.fits"
        calibrate_light(
            light_path=lp,
            out_path=out,
            master_bias_path=master_bias_path,
            master_dark_path=master_dark_path,
            master_flat_path=master_flat_path,
        )
        calibrated_paths.append(out)

    # Stack
    if calibrated_paths:
        stack_out = stacked_dir / f"{desc.project_id}_stacked.fits"
        stack_calibrated_images(calibrated_paths, stack_out, method=stack_method)

    update_descriptor_file(descriptor_path, desc)


def process_all_projects(
    output_root: Path,
    temp_tol: float,
    stack_method: str,
) -> None:
    project_root = output_root / "projects"
    for descriptor_path in sorted(project_root.rglob("project_descriptor.json")):
        print(f"[INFO] Processing {descriptor_path}")
        process_project(descriptor_path, output_root, temp_tol, stack_method)


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Astronomical FITS calibration and stacking pipeline.")
    parser.add_argument("root", type=str, help="Root directory containing FITS files.")
    parser.add_argument("--output", type=str, required=True, help="Output directory.")
    parser.add_argument("--temp-tol", type=float, default=3.0, help="Temperature tolerance in C for calibration matching.")
    parser.add_argument(
        "--stack-method",
        choices=["sigma_clip_mean", "median", "mean"],
        default="sigma_clip_mean",
        help="Stacking method for calibrated lights."
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only scan tree and write project descriptors; do not process."
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Scanning FITS tree: {root}")
    frames = scan_fits_tree(root)
    print(f"[INFO] Found {len(frames)} FITS files.")

    descriptors = build_project_descriptors(
        frames=frames,
        output_dir=output_dir,
        temp_tol=args.temp_tol,
    )
    print(f"[INFO] Wrote {len(descriptors)} project descriptors under {output_dir / 'projects'}")

    if not args.scan_only:
        process_all_projects(
            output_root=output_dir,
            temp_tol=args.temp_tol,
            stack_method=args.stack_method,
        )
        print("[INFO] Processing complete.")


if __name__ == "__main__":
    main()