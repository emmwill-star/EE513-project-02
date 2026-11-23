from collections.abc import Iterable, Mapping
import json
from copy import deepcopy
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from psu_imaging import srgb_to_linear

import rawpy
import sys
import os
from pathlib import Path
import importlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, ExifTags
import imageio
import tifffile

RAW_EXTENSIONS = {".dng", ".raw", ".nef", ".arw", ".cr2", ".cr3", ".raf", ".orf", ".rw2", ".pef", ".srw"}


class classicImagePipeline:
    """One RAW capture with explicit, chainable processing stages.

    ``image`` is the current float64 mosaic/RGB array, never a LibRaw preview.
    Methods modify this instance and return self; get_intrinsics returns a copy.
    Only the original mosaic and current image are retained automatically.
    Units/domain and stage history make experimental pipelines inspectable.
    """

    def __init__(self, imagePath: str | Path):
        self.imagePath = Path(imagePath)
        with rawpy.imread(str(self.imagePath)) as raw:
            if raw.raw_pattern is None or raw.raw_pattern.shape != (2, 2):
                raise ValueError("Pipeline requires a 2 x 2 Bayer RAW capture")
            mosaic = _validate_bayer_mosaic(raw.raw_image_visible, "RAW mosaic")
            self.rawMosaic = mosaic.copy()
            pattern = raw.raw_colors_visible[:2, :2].copy()
            colors = np.array([chr(raw.color_desc[int(index)]) for index in pattern.flat]).reshape(2, 2)
            if (sorted(colors.flat) != ["B", "G", "G", "R"] or
                    colors[0, 0] == colors[0, 1] or colors[0, 0] == colors[1, 0] or
                    colors[0, 1] == colors[1, 1] or colors[1, 0] == colors[1, 1]):
                raise ValueError("Unsupported visible Bayer phase")
            levels = np.asarray(raw.black_level_per_channel, dtype=float)[pattern]
            white = float(raw.white_level)
            if not np.isfinite(levels).all() or not np.isfinite(white) or np.any(levels >= white):
                raise ValueError("RAW white level must exceed finite site black levels")
            cameraGains = np.asarray(raw.camera_whitebalance, dtype=float)
            # Some decoders leave the second green's WB slot unused (zero).
            siteGains = cameraGains[pattern].copy()
            greenGains = siteGains[(colors == "G") & (siteGains > 0)]
            if greenGains.size:
                siteGains[(colors == "G") & (siteGains == 0)] = greenGains.mean()
                siteGains /= greenGains.mean()
            self.intrinsics = {
                "visiblePattern": pattern, "visibleColors": colors,
                "siteBlackLevels": levels.copy(), "whiteLevel": white,
                "shape": mosaic.shape,
                "visibleMargins": (raw.sizes.top_margin, raw.sizes.left_margin),
                "cameraWhiteBalance": cameraGains.copy(),
                "siteWhiteBalanceGains": siteGains,
            }
        self.rawMosaic.flags.writeable = False
        self.intrinsics["cameraId"] = _read_camera_id(self.imagePath)
        try:
            self.intrinsics.update(lens_id(self.imagePath))
        except (KeyError, TypeError, ValueError, IndexError):
            # RAW formats without TIFF EXIF still work with array-only stages.
            self.intrinsics.update(lensModel=None, focalLengthMm=None, iso=None,
                                   exposureTimeSeconds=None)
        self.reset()

    def get_intrinsics(self):
        """Return capture metadata, not a geometric camera calibration matrix."""
        return deepcopy(self.intrinsics)

    def reset(self):
        """Start another experiment from the original RAW, without rereading it."""
        self.image = self.rawMosaic.astype(np.float64)
        self.domain = "raw_dn"
        self.history = []
        self.saturationMask = self.rawMosaic >= self.intrinsics["whiteLevel"]
        self.hotMask = self.lowMask = self.shadingMap = self.whiteBalanceGains = None
        return self

    def _require(self, *domains):
        if self.domain not in domains:
            raise ValueError(f"Stage requires {domains}; current domain is {self.domain}")

    def _record(self, name, image, **parameters):
        self.image = image
        self.history.append({"stage": name, "parameters": parameters})
        return self

    def _once(self, name):
        if any(entry["stage"] == name for entry in self.history):
            raise ValueError(f"{name} has already been applied; reset before repeating it")

    def _compatible_capture(self, capture, *, dark=False, allowExposureMismatch=False):
        keys = ("shape", "visibleMargins", "cameraId", "lensModel", "focalLengthMm")
        for key in keys:
            value = self.intrinsics.get(key)
            if value is None or value == "unknown" or value == "Unknown camera":
                raise ValueError(f"Cannot verify calibration compatibility: missing {key}")
            if value != capture.intrinsics.get(key):
                raise ValueError(f"Calibration capture has incompatible {key}")
        if not np.array_equal(self.intrinsics["visibleColors"], capture.intrinsics["visibleColors"]):
            raise ValueError("Calibration capture has incompatible visible Bayer phase")
        if dark:
            import warnings
            mismatches = [key for key in ("iso", "exposureTimeSeconds")
                          if self.intrinsics.get(key) is None or capture.intrinsics.get(key) is None
                          or self.intrinsics[key] != capture.intrinsics[key]]
            if mismatches and not allowExposureMismatch:
                raise ValueError(f"Dark capture must match {mismatches}; select a matched dark")
            if mismatches:
                warnings.warn("Using an unmatched dark for exploratory defect candidates only: "
                              + ", ".join(mismatches), UserWarning, stacklevel=3)

    def subtract_black_level(self):
        self._require("raw_dn")
        image = subtract_black_level(self.image, self.intrinsics["siteBlackLevels"])
        self.domain = "black_subtracted_dn"
        return self._record("subtract_black_level", image)

    def normalize_saturation(self):
        self._require("black_subtracted_dn")
        result = normalize_saturation(self.image, self.intrinsics["siteBlackLevels"],
                                      self.intrinsics["whiteLevel"])
        self.domain = "linear_bayer"
        # Keep the acquisition saturation mask, even if defects were repaired first.
        return self._record("normalize_saturation", result["corrected"])

    def correct_defective_pixels(self, darkPath, *, defectThresholdDN,
                                 allowExposureMismatch=False):
        """Detect hot/low candidates from a dark; not proof of persistent defects.

        Temperature compatibility must be checked by the caller. Pixels stuck at
        normal black cannot be identified from a dark alone: use a lit flat too.
        """
        self._require("raw_dn", "black_subtracted_dn", "linear_bayer")
        self._once("correct_defective_pixels")
        dark = classicImagePipeline(darkPath)
        self._compatible_capture(dark, dark=True, allowExposureMismatch=allowExposureMismatch)
        defects = find_defective_pixels(dark.rawMosaic, defectThresholdDN)
        image = correct_defective_pixels(self.image, defects["hotMask"] | defects["lowMask"])
        self.hotMask, self.lowMask = defects["hotMask"], defects["lowMask"]
        return self._record("correct_defective_pixels", image, darkPath=str(darkPath),
                            defectThresholdDN=float(defectThresholdDN),
                            allowExposureMismatch=allowExposureMismatch)

    def correct_lens_shading(self, flatPath, *, roiSize=64, sigma=32):
        """Build and apply a per-site map; uniform flat illumination is required."""
        self._require("black_subtracted_dn", "linear_bayer")
        self._once("correct_lens_shading")
        flat = classicImagePipeline(flatPath)
        self._compatible_capture(flat)
        if flat.saturationMask.any():
            raise ValueError("Flat contains saturated samples")
        flat.subtract_black_level()
        shading = build_bayer_shading_map(flat.image, roiSize=roiSize, sigma=sigma)
        image = correct_bayer_lens_shading(self.image, shading["normalizedFlat"])
        self.shadingMap = shading
        return self._record("correct_lens_shading", image, flatPath=str(flatPath),
                            roiSize=roiSize, sigma=sigma)

    def white_balance(self, gains=None):
        """Multiply linear Bayer sites by RGB or explicit 2 x 2 gains.

        None explicitly selects camera metadata gains, normalized to green.
        Measured CCMs require the same gains/convention as their calibration.
        """
        self._require("linear_bayer")
        self._once("white_balance")
        gains = np.asarray(self.intrinsics["siteWhiteBalanceGains"] if gains is None else gains,
                           dtype=float)
        if gains.shape == (3,):
            colors = self.intrinsics["visibleColors"]
            gains = np.array([[gains["RGB".index(color)] for color in row] for row in colors])
        if gains.shape != (2, 2) or not np.isfinite(gains).all() or np.any(gains <= 0):
            raise ValueError("White-balance gains must be positive finite RGB or 2 x 2 values")
        image = self.image.copy()
        for row, column in np.ndindex(2, 2):
            image[row::2, column::2] *= gains[row, column]
        self.whiteBalanceGains = gains.copy()
        return self._record("white_balance", image, gains=gains.tolist())

    def demosaic(self, method="mhc"):
        """Select a from-scratch demosaicer. Skipping WB is an explicit ablation."""
        self._require("linear_bayer")
        methods = {"bilinear": bilinear_demosaic, "mhc": malvar_he_cutler_demosaic}
        if method not in methods:
            raise ValueError("Demosaicing method must be 'bilinear' or 'mhc'")
        image = methods[method](self.image, self.intrinsics["visibleColors"])
        self.domain = "linear_camera_rgb"
        return self._record("demosaic", image, method=method)

    def denoise(self, method="bilateral", *, diameter=5, sigmaColor=0.05, sigmaSpace=2):
        """Edge-aware RGB bilateral filtering or a callable(array) replacement.

        sigmaColor uses the current image's normalized linear units. The filter
        preserves negative values and highlight headroom; it does not clip.
        """
        self._require("linear_camera_rgb", "linear_srgb")
        if callable(method):
            image = np.asarray(method(self.image.copy()), dtype=float)
            if image.shape != self.image.shape or not np.isfinite(image).all():
                raise ValueError("Denoiser must return a finite RGB array of matching shape")
            return self._record("denoise", image.copy(), method=getattr(method, "__name__", "callable"))
        if method != "bilateral":
            raise ValueError("Denoiser must be 'bilateral' or a callable")
        if isinstance(diameter, bool) or not isinstance(diameter, (int, np.integer)) or diameter < 1:
            raise ValueError("diameter must be a positive integer")
        sigmaColor = _validate_real_scalar(sigmaColor, "sigmaColor")
        sigmaSpace = _validate_real_scalar(sigmaSpace, "sigmaSpace")
        if sigmaColor <= 0 or sigmaSpace <= 0:
            raise ValueError("Bilateral sigmas must be positive")
        image = cv2.bilateralFilter(self.image.astype(np.float32), int(diameter),
                                    sigmaColor, sigmaSpace).astype(np.float64)
        return self._record("denoise", image, method=method, diameter=diameter,
                            sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

    def color_correct(self, matrix, *, referenceIlluminant):
        """Apply measured XYZ = matrix @ cameraRGB, adapt white, then linear sRGB.

        Pass the matrix fitted for the already applied WB gains, not cameraToXyz
        (which embeds them). referenceIlluminant is the calibration white's xy.
        No display clipping or transfer function is applied here.
        """
        import colour
        self._require("linear_camera_rgb")
        matrix = np.asarray(matrix, dtype=float)
        illuminant = np.asarray(referenceIlluminant, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("matrix must be a finite 3 x 3 camera-to-XYZ matrix")
        if (illuminant.shape != (2,) or not np.isfinite(illuminant).all()
                or np.any(illuminant <= 0) or illuminant.sum() >= 1):
            raise ValueError("referenceIlluminant must be a valid white-point xy pair")
        xyz = self.image @ matrix.T
        image = colour.XYZ_to_sRGB(xyz, illuminant=illuminant,
                                   chromatic_adaptation_transform="Bradford",
                                   apply_cctf_encoding=False)
        if not np.isfinite(image).all():
            raise ValueError("Color correction produced nonfinite values")
        self.domain = "linear_srgb"
        return self._record("color_correct", image, matrix=matrix.tolist(),
                            referenceIlluminant=illuminant.tolist())

    def tone_curve(self, *, exposure=0.0, gamma=None):
        """Expose in stops, clip to [0,1], then sRGB encoding (or power gamma)."""
        self._require("linear_srgb")
        exposure = _validate_real_scalar(exposure, "exposure")
        if not -32 <= exposure <= 32:
            raise ValueError("exposure must be between -32 and 32 stops")
        if gamma is not None:
            gamma = _validate_real_scalar(gamma, "gamma")
            if gamma <= 0:
                raise ValueError("gamma must be positive")
        linear = np.clip(self.image * (2.0 ** exposure), 0, 1)
        if gamma is None:
            image = np.where(linear <= 0.0031308, 12.92 * linear,
                             1.055 * linear ** (1 / 2.4) - 0.055)
        else:
            image = linear ** (1 / gamma)
        self.domain = "display_rgb"
        return self._record("tone_curve", image, exposure=exposure, gamma=gamma)

    def sharpen(self, *, amount=0.5, sigma=1.0):
        """Unsharp masking in the selected RGB domain; clip only display output."""
        from scipy.ndimage import gaussian_filter
        self._require("linear_camera_rgb", "linear_srgb", "display_rgb")
        amount = _validate_real_scalar(amount, "amount")
        sigma = _validate_real_scalar(sigma, "sigma")
        if amount < 0 or sigma <= 0:
            raise ValueError("amount must be nonnegative and sigma must be positive")
        blurred = gaussian_filter(self.image, sigma=(sigma, sigma, 0), mode="reflect")
        image = self.image + amount * (self.image - blurred)
        if self.domain == "display_rgb":
            image = np.clip(image, 0, 1)
        return self._record("sharpen", image, amount=amount, sigma=sigma)


def update_json_values(jsonPath: str | Path, updates: Mapping[str, object]) -> None:
    """Replace values of existing top-level keys in a JSON file.

    Pass a mapping of keys to replacement values, for example:
        update_json_values("results.json", {
            "camera": "My camera",
            "focal_length_mm": [4.25, 0.02],
            "read_noise_e": None,
        })

    All keys are checked before writing; unknown keys raise KeyError.
    Unspecified values and the top-level key order are preserved. Values are
    replaced, not appended or merged. An empty mapping leaves the file unchanged.
    The file must already exist and contain a JSON object (otherwise ValueError).
    Output uses UTF-8 and two-space indentation; original whitespace is not retained.
    File errors, JSON decoding errors, and serialization errors propagate.
    Invalid updates never overwrite the file, including NaN/infinite values.
    """
    if not isinstance(updates, Mapping):
        raise TypeError("updates must be a mapping of existing keys to replacement values")

    jsonPath = Path(jsonPath)
    with jsonPath.open(encoding="utf-8") as jsonFile:
        data = json.load(jsonFile)
    if not isinstance(data, dict):
        raise ValueError("JSON file must contain a top-level object")

    for key in updates:
        if key not in data:
            raise KeyError(f"JSON key does not exist: {key!r}")
    if not updates:
        return

    data.update(updates)
    serialized = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    jsonPath.write_text(serialized + "\n", encoding="utf-8")



def lens_id(path):
    """Return a dictionary of lens and exposure metadata.

    Keys: lensModel, focalLengthMm, focalLength35Mm, imageSize (width, height),
    iso, and exposureTimeSeconds.
    Missing required EXIF tags raise KeyError; file reading errors propagate.
    """
    with tifffile.TiffFile(path) as imageFile:
        page = imageFile.pages[0]
        exif = page.tags["ExifTag"].value
        focalLength = exif["FocalLength"]
        iso = exif["ISOSpeedRatings"]
        exposureTime = exif["ExposureTime"]

        return {
            "lensModel": exif.get("LensModel", "unknown"),
            "focalLengthMm": round(focalLength[0] / focalLength[1], 3),
            "focalLength35Mm": exif.get("FocalLengthIn35mmFilm"),
            "imageSize": (page.tags["ImageWidth"].value, page.tags["ImageLength"].value),
            "iso": iso,
            "exposureTimeSeconds": exposureTime[0] / exposureTime[1],
        }

def _load_image(imagePath: Path) -> Image.Image:
	if imagePath.suffix.lower() in RAW_EXTENSIONS:
		import rawpy

		with rawpy.imread(str(imagePath)) as raw:
			return Image.fromarray(raw.postprocess(use_camera_wb=True))
	with Image.open(imagePath) as image:
		return ImageOps.exif_transpose(image).convert("RGB")


def _read_camera_id(imagePath: Path) -> str | None:
	if imagePath.suffix.lower() != ".dng":
		return None

	import tifffile

	with tifffile.TiffFile(imagePath) as dngFile:
		tags = dngFile.pages[0].tags
		cameraId = "Unknown camera"
		for tagCode in (272, 271):
			tag = tags.get(tagCode)
			if tag is not None:
				value = str(tag.value).strip("\x00 ")
				if value:
					cameraId = value
					break

		exifTag = tags.get(34665)
		exif = exifTag.value if exifTag is not None else {}
		cameraDescriptions = [exif.get("LensModel", "")]
		for tagCode in (50708, 42036):
			tag = tags.get(tagCode)
			if tag is not None:
				cameraDescriptions.append(tag.value)
		for description in cameraDescriptions:
			cameraNumber = re.search(
				r"\bcamera\s+(?:#\s*|(?:number|no\.?)\s*)?(\d+)\b(?!\.|\s*(?:mm|cm|m)\b)",
				str(description),
				re.IGNORECASE,
			)
			if cameraNumber is not None:
				return f"{cameraId} | Camera {cameraNumber.group(1)}"
		return cameraId


def get_noise_profile_status(imagePath: str | Path) -> dict:
	"""Return NoiseProfile presence and coefficient values for a TIFF/DNG."""
	import tifffile

	with tifffile.TiffFile(imagePath) as dngFile:
		for page in dngFile.pages:
			noiseProfile = page.tags.get(51041)
			if noiseProfile is not None:
				return {"present": True, "values": tuple(noiseProfile.value)}
	return {"present": False, "values": None}


def has_noise_profile(imagePath: str | Path) -> bool:
	"""Return whether a TIFF/DNG contains the DNG NoiseProfile tag."""
	return get_noise_profile_status(imagePath)["present"]


## Excerise 3
def read_raw_or_array(imageSource):
	if isinstance(imageSource, (str, Path)):
		with rawpy.imread(str(imageSource)) as raw:
			return raw.raw_image_visible[0::2, 1::2].astype(np.float64)
	imageArray = np.asarray(imageSource, dtype=np.float64)
	if imageArray.ndim != 2:
		raise ValueError("Flat-field inputs must be 2D raw mosaics")
	return imageArray

def measure_flat_field_pairs(imagePairs, blackLevel, roiSize=128):
	imagePairs = list(imagePairs)
	if not imagePairs:
		raise ValueError("At least one pair of flat-field images is required")
	if not isinstance(roiSize, int) or isinstance(roiSize, bool) or roiSize < 2:
		raise ValueError("roiSize must be an integer of at least 2 pixels")

	meanSignalDN = []
	noiseVarianceDN2 = []
	for imageSourceA, imageSourceB in imagePairs:
		imageA = read_raw_or_array(imageSourceA)
		imageB = read_raw_or_array(imageSourceB)
		if imageA.shape != imageB.shape:
			raise ValueError("Paired flat fields must have matching dimensions")

		height, width = imageA.shape
		if height < roiSize or width < roiSize:
			raise ValueError(f"Images must be at least {roiSize} x {roiSize} pixels")
		top = (height - roiSize) // 2
		left = (width - roiSize) // 2
		imageA = imageA[top:top + roiSize, left:left + roiSize]
		imageB = imageB[top:top + roiSize, left:left + roiSize]

		meanSignalDN.append((imageA.mean() + imageB.mean()) / 2 - blackLevel)
		noiseVarianceDN2.append(np.var(imageA - imageB, ddof=1) / 2)

	return np.asarray(meanSignalDN), np.asarray(noiseVarianceDN2)

def measure_flat_field_single(images, blackLevel, roiSize=128):
	images = list(images)
	if not images:
		raise ValueError("At least one flat-field image is required")
	if not isinstance(roiSize, int) or isinstance(roiSize, bool) or roiSize < 2:
		raise ValueError("roiSize must be an integer of at least 2 pixels")

	meanSignalDN = []
	noiseVarianceDN2 = []
	for imageSource in images:
		image = read_raw_or_array(imageSource)
		if image.ndim != 2:
			raise ValueError("Flat-field inputs must be 2D raw mosaics")

		height, width = image.shape
		if height < roiSize or width < roiSize:
			raise ValueError(f"Images must be at least {roiSize} x {roiSize} pixels")
		top = (height - roiSize) // 2
		left = (width - roiSize) // 2
		image = image[top:top + roiSize, left:left + roiSize]

		meanSignalDN.append(image.mean() - blackLevel)
		noiseVarianceDN2.append(np.var(image, ddof=1))

	return np.asarray(meanSignalDN), np.asarray(noiseVarianceDN2)

def plot_photon_transfer(
	meanSignalDN,
	noiseVarianceDN2,
	saturationDN,
	saturationFraction=0.7,
	fitRange=None,
	ax=None,
):
	meanSignalDN = np.asarray(meanSignalDN, dtype=float)
	noiseVarianceDN2 = np.asarray(noiseVarianceDN2, dtype=float)
	if meanSignalDN.ndim != 1 or noiseVarianceDN2.ndim != 1:
		raise ValueError("Photon-transfer measurements must be one-dimensional")
	if meanSignalDN.size != noiseVarianceDN2.size or meanSignalDN.size == 0:
		raise ValueError("Mean and variance arrays must have the same nonzero length")
	if not np.isfinite(meanSignalDN).all() or not np.isfinite(noiseVarianceDN2).all():
		raise ValueError("Photon-transfer measurements must be finite")
	if not np.isfinite(saturationDN) or saturationDN <= 0:
		raise ValueError("saturationDN must be a positive finite signal range")
	if not 0 < saturationFraction <= 1:
		raise ValueError("saturationFraction must be in (0, 1]")

	saturationLimitDN = saturationDN * saturationFraction
	belowSaturation = meanSignalDN < saturationLimitDN
	if np.count_nonzero(belowSaturation) < 3:
		raise ValueError("At least three measurements must be below the saturation cutoff")
	fitMask = belowSaturation.copy()
	if fitRange is not None:
		fitRange = np.asarray(fitRange, dtype=float)
		if fitRange.shape != (2,) or not np.isfinite(fitRange).all() or fitRange[0] >= fitRange[1]:
			raise ValueError("fitRange must contain increasing finite DN bounds")
		fitMask &= (meanSignalDN >= fitRange[0]) & (meanSignalDN <= fitRange[1])
	if np.count_nonzero(fitMask) < 3:
		raise ValueError("At least three plotted measurements must be inside fitRange")
	if ax is None:
		_, ax = plt.subplots()

	ax.scatter(meanSignalDN[belowSaturation], noiseVarianceDN2[belowSaturation])
	ax.set_xlabel("Mean signal, E[D] - B (DN)")
	ax.set_ylabel("Temporal variance (DN^2)")
	ax.set_title(f"Photon-transfer curve below {saturationFraction:.0%} saturation")

	slope, intercept = np.polyfit(meanSignalDN[fitMask], noiseVarianceDN2[fitMask], 1)
	# if slope <= 0 or intercept < 0:
	# 	raise ValueError("Fit must have a positive slope and nonnegative intercept")
	fitSignalDN = np.linspace(
		meanSignalDN[fitMask].min(), meanSignalDN[fitMask].max(), 100
	)
	ax.plot(fitSignalDN, slope * fitSignalDN + intercept, label="Linear fit")
	ax.legend()

	readNoiseDN = np.sqrt(intercept/slope**2)
	return {
		"slopeDNPerElectron": float(slope),
		"interceptDN2": float(intercept),
		"conversionGainElectronPerDN": float(1 / slope),
		"readNoiseDN": float(readNoiseDN),
		"readNoiseElectrons": float(readNoiseDN / slope),
		"saturationLimitDN": float(saturationLimitDN),
		"saturationFraction": float(saturationFraction),
		"fitRangeDN": (
			float(meanSignalDN[fitMask].min()),
			float(meanSignalDN[fitMask].max()),
		),
	}

def calculate_signal_to_noise(signalElectrons, readNoiseElectrons):
	signalElectrons = np.asarray(signalElectrons, dtype=float)
	readNoiseElectrons = np.asarray(readNoiseElectrons, dtype=float)
	if not np.isfinite(signalElectrons).all() or np.any(signalElectrons < 0):
		raise ValueError("signalElectrons must contain finite nonnegative values")
	if not np.isfinite(readNoiseElectrons).all() or np.any(readNoiseElectrons < 0):
		raise ValueError("readNoiseElectrons must be finite and nonnegative")

	denominator = np.sqrt(signalElectrons + readNoiseElectrons ** 2)
	snr = np.divide(
		signalElectrons,
		denominator,
		out=np.zeros_like(signalElectrons, dtype=float),
		where=denominator > 0,
	)
	return float(snr) if snr.ndim == 0 else snr


def plot_snr_vs_signal(meanSignalDN, conversionGainElectronPerDN, readNoiseElectrons, ax=None):
	"""Plot predicted shot-plus-read-noise SNR over the supplied signal range.

	Accept black-subtracted mean signal in DN, gain in electrons/DN, and
	read noise in electrons RMS. Draw a smooth model curve in electron units,
	not lines connecting the input samples. Return the Matplotlib axes.
	Raise ValueError for invalid signals, gain, or read noise.
	"""
	meanSignalDN = np.asarray(meanSignalDN, dtype=float)
	if meanSignalDN.ndim != 1 or meanSignalDN.size == 0:
		raise ValueError("meanSignalDN must be a nonempty one-dimensional array")
	if not np.isfinite(meanSignalDN).all() or np.any(meanSignalDN < 0):
		raise ValueError("meanSignalDN must contain finite nonnegative values")
	gainArray = np.asarray(conversionGainElectronPerDN, dtype=float)
	if gainArray.ndim != 0 or not np.isfinite(gainArray) or gainArray <= 0:
		raise ValueError("conversionGainElectronPerDN must be a finite positive scalar")
	signalElectrons = meanSignalDN * float(gainArray)
	if not np.isfinite(signalElectrons).all():
		raise ValueError("Gain-scaled signals must be finite")
	readNoiseArray = np.asarray(readNoiseElectrons, dtype=float)
	if readNoiseArray.ndim != 0 or not np.isfinite(readNoiseArray) or readNoiseArray < 0:
		raise ValueError("readNoiseElectrons must be a finite nonnegative scalar")
	readNoiseElectrons = float(readNoiseArray)

	positiveSignal = signalElectrons > 0
	if not np.any(positiveSignal):
		raise ValueError("At least one positive signal is required for logarithmic axes")
	curveSignalElectrons = np.geomspace(
		signalElectrons[positiveSignal].min(), signalElectrons[positiveSignal].max(), 400,
	)
	curveSnr = calculate_signal_to_noise(curveSignalElectrons, readNoiseElectrons)
	if ax is None:
		_, ax = plt.subplots()

	ax.loglog(curveSignalElectrons, curveSnr, label="Shot + read noise prediction")
	ax.set_xlabel("Signal (electrons)")
	ax.set_ylabel("Signal-to-noise ratio")
	ax.set_title("Signal-to-noise ratio versus signal")
	if readNoiseElectrons > 0:
		kneeSignalElectrons = readNoiseElectrons ** 2
		ax.axvline(
			kneeSignalElectrons,
			color="tab:orange",
			linestyle="--",
			label=f"Knee = {kneeSignalElectrons:.2g} e-",
		)
	ax.legend()
	return ax


def plot_dark_current(exposureTimesSeconds, darkSignalElectrons, ax=None):
	exposureTimesSeconds = np.asarray(exposureTimesSeconds, dtype=float)
	darkSignalElectrons = np.asarray(darkSignalElectrons, dtype=float)
	if exposureTimesSeconds.ndim != 1 or darkSignalElectrons.ndim != 1:
		raise ValueError("Exposure times and dark signals must be one-dimensional")
	if exposureTimesSeconds.size != darkSignalElectrons.size or exposureTimesSeconds.size < 2:
		raise ValueError("At least two matching exposure and dark-signal values are required")
	if not np.isfinite(exposureTimesSeconds).all() or np.any(exposureTimesSeconds < 0):
		raise ValueError("Exposure times must be finite and nonnegative")
	if not np.isfinite(darkSignalElectrons).all():
		raise ValueError("Dark signal values must be finite")
	if np.unique(exposureTimesSeconds).size < 2:
		raise ValueError("At least two distinct exposure times are required")

	order = np.argsort(exposureTimesSeconds)
	exposureTimesSeconds = exposureTimesSeconds[order]
	darkSignalElectrons = darkSignalElectrons[order]
	slope, intercept = np.polyfit(exposureTimesSeconds, darkSignalElectrons, 1)
	if ax is None:
		_, ax = plt.subplots()

	ax.plot(exposureTimesSeconds, darkSignalElectrons, "o", label="Measured dark signal")
	fitTimes = np.linspace(exposureTimesSeconds.min(), exposureTimesSeconds.max(), 100)
	ax.plot(fitTimes, slope * fitTimes + intercept, label="Linear fit")
	ax.set_xlabel("Exposure time (s)")
	ax.set_ylabel("Mean dark signal above black (electrons)")
	ax.set_title("Dark signal versus exposure time")
	ax.legend()
	return {
		"darkCurrentElectronsPerSecond": float(slope),
		"interceptElectrons": float(intercept),
		"ax": ax,
	}
 
## Exercise 4 

def _validate_bayer_mosaic(image, name):
    image = np.asarray(image)
    if (
        image.ndim != 2
        or min(image.shape) < 4
        or not np.issubdtype(image.dtype, np.number)
        or np.iscomplexobj(image)
        or not np.isfinite(image).all()
    ):
        raise ValueError(f"{name} must be a finite real 2D array at least 4 x 4")
    return image


def _validate_site_levels(blackLevel):
    levels = np.asarray(blackLevel)
    if (
        levels.shape not in ((), (2, 2))
        or not np.issubdtype(levels.dtype, np.number)
        or np.iscomplexobj(levels)
        or not np.isfinite(levels).all()
    ):
        raise ValueError("blackLevel must be a finite scalar or 2 x 2 array")
    return np.broadcast_to(levels, (2, 2)).astype(np.float64)


def _validate_real_scalar(value, name):
    value = np.asarray(value)
    if (
        value.ndim != 0
        or not np.issubdtype(value.dtype, np.number)
        or np.iscomplexobj(value)
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


def subtract_black_level(bayerMosaic: np.ndarray, blackLevel) -> np.ndarray:
    """Subtract scalar or 2 x 2 visible-site black levels from a Bayer mosaic.

    Inputs and output are in DN. Return float64 values without clipping,
    preserving negative noise samples and leaving the input unchanged.
    Raise ValueError for invalid mosaic or black-level inputs.
    """
    mosaic = _validate_bayer_mosaic(bayerMosaic, "bayerMosaic")
    levels = _validate_site_levels(blackLevel)
    corrected = mosaic.astype(np.float64)
    for row, column in np.ndindex(2, 2):
        corrected[row::2, column::2] -= levels[row, column]
    return corrected


def normalize_saturation(
    blackCorrectedMosaic: np.ndarray, blackLevel, saturationValue: float,
) -> dict:
    """Normalize black-subtracted DN and clip at the measured RAW white level.

    blackLevel is a scalar or 2 x 2 visible-site array in DN and
    saturationValue is the original RAW white level in DN. Return a float64
    mosaic under ``corrected`` and a boolean ``saturationMask`` marking samples
    at or above white before clipping. Negative values remain unchanged apart
    from scaling. Raise ValueError for invalid inputs.
    """
    mosaic = _validate_bayer_mosaic(blackCorrectedMosaic, "blackCorrectedMosaic")
    levels = _validate_site_levels(blackLevel)
    white = _validate_real_scalar(saturationValue, "saturationValue")
    if np.any(levels >= white):
        raise ValueError("saturationValue must exceed every black level")

    corrected = np.empty(mosaic.shape, dtype=np.float64)
    saturationMask = np.empty(mosaic.shape, dtype=bool)
    for row, column in np.ndindex(2, 2):
        site = (slice(row, None, 2), slice(column, None, 2))
        normalized = mosaic[site] / (white - levels[row, column])
        saturationMask[site] = normalized >= 1
        corrected[site] = np.minimum(normalized, 1)
    return {"corrected": corrected, "saturationMask": saturationMask}


def find_defective_pixels(darkMosaic: np.ndarray, defectThresholdDN: float) -> dict:
    """Find hot and low candidates using a dark Bayer frame in raw DN.

    Compare each pixel with the median of its 3 x 3 same-site neighborhood.
    Return boolean ``hotMask`` and ``lowMask`` arrays. Choose the positive
    threshold from measured noise and verify candidates using matched dark
    frames. A pixel stuck at normal black requires a lit flat to identify.
    Raise ValueError for invalid inputs.
    """
    from scipy.ndimage import median_filter

    dark = _validate_bayer_mosaic(darkMosaic, "darkMosaic")
    threshold = _validate_real_scalar(defectThresholdDN, "defectThresholdDN")
    if threshold <= 0:
        raise ValueError("defectThresholdDN must be positive")
    hotMask = np.zeros(dark.shape, dtype=bool)
    lowMask = np.zeros(dark.shape, dtype=bool)
    for row, column in np.ndindex(2, 2):
        site = (slice(row, None, 2), slice(column, None, 2))
        darkPlane = dark[site].astype(np.float64)
        residual = darkPlane - median_filter(darkPlane, size=3, mode="reflect")
        hotMask[site] = residual > threshold
        lowMask[site] = residual < -threshold
    return {"hotMask": hotMask, "lowMask": lowMask}


def correct_defective_pixels(bayerMosaic: np.ndarray, defectMask: np.ndarray) -> np.ndarray:
    """Replace masked Bayer pixels with the median of valid same-site neighbors.

    Preserve the input units and return a float64 copy. The boolean defect mask
    must match the mosaic's sensor coordinates. Raise ValueError for invalid
    inputs or a defect without valid neighbors.
    """
    mosaic = _validate_bayer_mosaic(bayerMosaic, "bayerMosaic")
    defectMask = np.asarray(defectMask)
    if defectMask.shape != mosaic.shape or defectMask.dtype != np.bool_:
        raise ValueError("defectMask must be a boolean array matching bayerMosaic")
    corrected = mosaic.astype(np.float64)
    for row, column in np.argwhere(defectMask):
        plane = corrected[row % 2::2, column % 2::2]
        planeDefects = defectMask[row % 2::2, column % 2::2]
        planeRow, planeColumn = row // 2, column // 2
        top, bottom = max(0, planeRow - 1), min(plane.shape[0], planeRow + 2)
        left, right = max(0, planeColumn - 1), min(plane.shape[1], planeColumn + 2)
        neighbors = plane[top:bottom, left:right]
        valid = ~planeDefects[top:bottom, left:right]
        if not valid.any():
            raise ValueError(f"No valid same-site neighbors for defective pixel ({row}, {column})")
        corrected[row, column] = np.median(neighbors[valid])
    return corrected


def correct_raw_sensor(
    bayerMosaic: np.ndarray,
    darkMosaic: np.ndarray,
    blackLevel,
    saturationValue: float,
    *,
    defectThresholdDN: float,
) -> dict:
    """Apply the black, saturation and dark-frame defect stages in sequence.

    Both mosaics must be aligned visible sensor arrays from the same camera,
    lens, crop and CFA phase. The caller must verify metadata compatibility and
    use a dark frame captured at suitable ISO, exposure and temperature.
    Return the normalized corrected mosaic and the original saturation, hot
    candidate and low candidate masks. See the individual stages for units and
    limitations. Raise ValueError for incompatible or invalid inputs.
    """
    mosaic = _validate_bayer_mosaic(bayerMosaic, "bayerMosaic")
    dark = _validate_bayer_mosaic(darkMosaic, "darkMosaic")
    if mosaic.shape != dark.shape:
        raise ValueError("bayerMosaic and darkMosaic must have matching shapes")

    blackCorrected = subtract_black_level(mosaic, blackLevel)
    saturation = normalize_saturation(blackCorrected, blackLevel, saturationValue)
    defects = find_defective_pixels(dark, defectThresholdDN)
    corrected = correct_defective_pixels(
        saturation["corrected"], defects["hotMask"] | defects["lowMask"],
    )
    return {
        "corrected": corrected,
        "hotMask": defects["hotMask"],
        "lowMask": defects["lowMask"],
        "saturationMask": saturation["saturationMask"],
    }


def build_bayer_shading_map(
    flatMosaic: np.ndarray, *, roiSize: int = 64, sigma: float = 32,
) -> dict:
    """Build a smooth, center-normalized response map from a Bayer flat.

    flatMosaic must be black-subtracted, unsaturated and from the same sensor
    crop and CFA phase as the target. Each of the four Bayer sites is smoothed
    separately with a Gaussian of sigma site-plane pixels; sigma=0 disables
    smoothing. roiSize is the width of the central square in site-plane pixels.
    Return the full-resolution response under ``normalizedFlat`` and the four
    central means under ``centerMeans`` in visible 2 x 2 site order.

    The flat must be uniformly illuminated; smoothing cannot distinguish lamp
    falloff from lens shading. The caller must check for saturated samples and
    validate the calibration using independent flats. Raise ValueError for
    invalid inputs, odd dimensions or a nonpositive smoothed response.
    """
    from scipy.ndimage import gaussian_filter

    flat = _validate_bayer_mosaic(flatMosaic, "flatMosaic")
    if any(size % 2 for size in flat.shape):
        raise ValueError("flatMosaic must have even dimensions")
    sigma = _validate_real_scalar(sigma, "sigma")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")

    planes = np.stack(
        [flat[row::2, column::2] for row, column in np.ndindex(2, 2)],
        axis=-1,
    ).astype(np.float64)
    if sigma:
        planes = gaussian_filter(planes, sigma=(sigma, sigma, 0), mode="reflect")
    normalizedPlanes, centerMeans = _normalize_flat_planes(planes, roiSize)
    if np.any(normalizedPlanes <= 0):
        raise ValueError("Smoothed flat response must be positive at every site")
    normalizedFlat = np.empty(flat.shape, dtype=np.float64)
    for index, (row, column) in enumerate(np.ndindex(2, 2)):
        normalizedFlat[row::2, column::2] = normalizedPlanes[..., index]
    return {"normalizedFlat": normalizedFlat, "centerMeans": centerMeans.reshape(2, 2)}


def correct_bayer_lens_shading(
    bayerMosaic: np.ndarray, normalizedFlat: np.ndarray,
) -> np.ndarray:
    """Divide a Bayer mosaic by its matching positive, per-site response map.

    The input may be black-subtracted DN or normalized linear values. Return a
    float64 mosaic in the same units without clipping; values above one and
    negative noise samples are preserved. The caller must verify that the map
    and image share sensor, lens, crop and CFA phase. Raise ValueError for
    mismatched or invalid arrays or a nonpositive response.
    """
    mosaic = _validate_bayer_mosaic(bayerMosaic, "bayerMosaic")
    response = _validate_bayer_mosaic(normalizedFlat, "normalizedFlat")
    if mosaic.shape != response.shape:
        raise ValueError("bayerMosaic and normalizedFlat must have matching shapes")
    if np.any(response <= 0):
        raise ValueError("normalizedFlat must be positive at every site")
    return mosaic.astype(np.float64) / response


def raw_to_linear_rgb(imagePath: str | Path, saturationValue: float):
    """Return quarter-pixel-count linear camera RGB from a Bayer RAW file.

    Subtract each site's metadata black level and divide by
    (saturationValue - black level), using the measured saturation DN.
    Each visible 2 x 2 block contributes R, mean(G1, G2), B without
    interpolation, white balance, color conversion, gamma, or rotation.
    The float64 output has shape (height // 2, width // 2, 3).
    Negative values and values above one are preserved, not clipped.

    Require an even-sized visible Bayer mosaic containing R, G, G, B.
    Raise ValueError for unsupported layouts or invalid normalization levels.
    File reading and RAW decoding errors propagate.
    """
    if not np.isscalar(saturationValue) or not np.isreal(saturationValue):
        raise ValueError("saturationValue must be a finite real scalar")
    saturationValue = float(saturationValue)
    if not np.isfinite(saturationValue):
        raise ValueError("saturationValue must be a finite real scalar")

    with rawpy.imread(str(imagePath)) as rawFile:
        if rawFile.raw_pattern is None or rawFile.raw_pattern.shape != (2, 2):
            raise ValueError("Linear RGB extraction requires a 2 x 2 Bayer pattern")
        mosaic = rawFile.raw_image_visible
        if mosaic.ndim != 2 or min(mosaic.shape) < 2 or any(size % 2 for size in mosaic.shape):
            raise ValueError("The visible RAW mosaic must have positive even dimensions")

        # Visible color indices account for sensor margins shifting the CFA origin.
        pattern = rawFile.raw_colors_visible[:2, :2]
        colorNames = [chr(rawFile.color_desc[int(index)]) for index in pattern.flat]
        if sorted(colorNames) != ["B", "G", "G", "R"]:
            raise ValueError("The visible Bayer block must contain R, G, G, B")
        blackLevels = np.asarray(rawFile.black_level_per_channel, dtype=np.float64)
        siteBlackLevels = blackLevels[pattern]
        if not np.isfinite(siteBlackLevels).all() or np.any(saturationValue <= siteBlackLevels):
            raise ValueError("saturationValue must exceed every finite site black level")

        sitePlanes = []
        for row, column in np.ndindex(2, 2):
            blackLevel = siteBlackLevels[row, column]
            sitePlanes.append(
                (mosaic[row::2, column::2].astype(np.float64) - blackLevel)
                / (saturationValue - blackLevel)
            )

    redPlane = sitePlanes[colorNames.index("R")]
    greenPlanes = [plane for plane, color in zip(sitePlanes, colorNames) if color == "G"]
    bluePlane = sitePlanes[colorNames.index("B")]
    output = {
		'stack': np.stack((redPlane, (greenPlanes[0] + greenPlanes[1]) / 2, bluePlane), axis=-1),
		'saturation': saturationValue,
		'blackLevel': siteBlackLevels,
		'pattern': pattern,
		'colorNames': colorNames,
 		}
    return output

def bilinear_demosaic(
    bayerMosaic: np.ndarray,
    pattern=(("B", "G"), ("G", "R")),
) -> np.ndarray:
    """Reconstruct RGB from a Bayer mosaic using bilinear interpolation.

    pattern gives the R/G/B layout at the mosaic's top-left 2 x 2 block;
    the default is BGGR. Each channel is interpolated only from its measured
    samples. At frame borders, weights are renormalized over available samples.
    Return a float64 H x W x 3 RGB array.
    """
    from scipy.ndimage import convolve

    bayerMosaic = np.asarray(bayerMosaic)
    if (
        bayerMosaic.ndim != 2
        or min(bayerMosaic.shape) < 2
        or not np.issubdtype(bayerMosaic.dtype, np.number)
        or np.iscomplexobj(bayerMosaic)
    ):
        raise ValueError("bayerMosaic must be a real numeric 2D array at least 2 x 2")
    if not np.isfinite(bayerMosaic).all():
        raise ValueError("bayerMosaic must contain only finite values")

    pattern = np.asarray(pattern)
    if (
        pattern.shape != (2, 2)
        or not np.isin(pattern, ("R", "G", "B")).all()
        or sorted(pattern.flat) != ["B", "G", "G", "R"]
    ):
        raise ValueError("pattern must be a 2 x 2 Bayer layout containing R, G, G, B")

    bayerMosaic = bayerMosaic.astype(np.float64, copy=False)
    redMask = np.zeros(bayerMosaic.shape, dtype=bool)
    greenMask = np.zeros(bayerMosaic.shape, dtype=bool)
    blueMask = np.zeros(bayerMosaic.shape, dtype=bool)
    redHorizontalMask = np.zeros(bayerMosaic.shape, dtype=bool)
    redVerticalMask = np.zeros(bayerMosaic.shape, dtype=bool)
    blueHorizontalMask = np.zeros(bayerMosaic.shape, dtype=bool)
    blueVerticalMask = np.zeros(bayerMosaic.shape, dtype=bool)

    for row, column in np.ndindex(2, 2):
        site = (slice(row, None, 2), slice(column, None, 2))
        channelName = pattern[row, column]
        if channelName == "R":
            redMask[site] = True
        elif channelName == "B":
            blueMask[site] = True
        else:
            greenMask[site] = True
            if pattern[row, (column + 1) % 2] == "R":
                redHorizontalMask[site] = True
                blueVerticalMask[site] = True
            else:
                blueHorizontalMask[site] = True
                redVerticalMask[site] = True

    greenKernel = np.array([
        [0, 0.25, 0],
        [0.25, 0, 0.25],
        [0, 0.25, 0],
    ])
    horizontalKernel = np.array([[0, 0, 0], [0.5, 0, 0.5], [0, 0, 0]])
    verticalKernel = horizontalKernel.T
    diagonalKernel = np.array([
        [0.25, 0, 0.25],
        [0, 0, 0],
        [0.25, 0, 0.25],
    ])

    def interpolate(kernel, sourceMask, targetMask):
        sourceWeights = sourceMask.astype(np.float64)
        neighborWeights = convolve(sourceWeights, kernel, mode="constant", cval=0.0)
        if np.any(targetMask & (neighborWeights == 0)):
            raise ValueError("pattern and image dimensions leave pixels without channel samples")
        weightedSamples = convolve(
            bayerMosaic * sourceWeights,
            kernel,
            mode="constant",
            cval=0.0,
        )
        interpolated = np.zeros_like(bayerMosaic)
        np.divide(weightedSamples, neighborWeights, out=interpolated, where=neighborWeights > 0)
        return interpolated

    greenAtRedBlue = interpolate(greenKernel, greenMask, redMask | blueMask)
    redAtGreenHorizontal = interpolate(horizontalKernel, redMask, redHorizontalMask)
    redAtGreenVertical = interpolate(verticalKernel, redMask, redVerticalMask)
    redAtBlue = interpolate(diagonalKernel, redMask, blueMask)
    blueAtGreenHorizontal = interpolate(horizontalKernel, blueMask, blueHorizontalMask)
    blueAtGreenVertical = interpolate(verticalKernel, blueMask, blueVerticalMask)
    blueAtRed = interpolate(diagonalKernel, blueMask, redMask)

    rgbImage = np.empty((*bayerMosaic.shape, 3), dtype=np.float64)
    rgbImage[..., 0][redMask] = bayerMosaic[redMask]
    rgbImage[..., 0][redHorizontalMask] = redAtGreenHorizontal[redHorizontalMask]
    rgbImage[..., 0][redVerticalMask] = redAtGreenVertical[redVerticalMask]
    rgbImage[..., 0][blueMask] = redAtBlue[blueMask]
    rgbImage[..., 1][greenMask] = bayerMosaic[greenMask]
    rgbImage[..., 1][redMask | blueMask] = greenAtRedBlue[redMask | blueMask]
    rgbImage[..., 2][blueMask] = bayerMosaic[blueMask]
    rgbImage[..., 2][blueHorizontalMask] = blueAtGreenHorizontal[blueHorizontalMask]
    rgbImage[..., 2][blueVerticalMask] = blueAtGreenVertical[blueVerticalMask]
    rgbImage[..., 2][redMask] = blueAtRed[redMask]

    return rgbImage

def malvar_he_cutler_demosaic(
    bayerMosaic: np.ndarray,
    pattern=(("B", "G"), ("G", "R")),
) -> np.ndarray:
    """Apply Malvar-He-Cutler corrections to a bilinear Bayer reconstruction.

    pattern gives the R/G/B layout at the mosaic's top-left 2 x 2 block;
    the default is BGGR. Eight CFA-phase masks select fixed 5 x 5 correction
    filters. Reflection padding supplies filter support at frame borders.
    Return a float64 H x W x 3 RGB array.
    """
    from scipy.ndimage import convolve

    pattern = np.asarray(pattern)
    rgbImage = bilinear_demosaic(bayerMosaic, pattern)
    bayerMosaic = np.asarray(bayerMosaic, dtype=np.float64)

    malvarGreenKernel = np.array([
        [0, 0, -1, 0, 0],
        [0, 0, 2, 0, 0],
        [-1, 2, 4, 2, -1],
        [0, 0, 2, 0, 0],
        [0, 0, -1, 0, 0],
    ], dtype=np.float64) / 8

    malvarHorizontalKernel = np.array([
        [0, 0, 0.5, 0, 0],
        [0, -1, 0, -1, 0],
        [-1, 4, 5, 4, -1],
        [0, -1, 0, -1, 0],
        [0, 0, 0.5, 0, 0],
    ], dtype=np.float64) / 8

    malvarOppositeKernel = np.array([
        [0, 0, -1.5, 0, 0],
        [0, 2, 0, 2, 0],
        [-1.5, 0, 6, 0, -1.5],
        [0, 2, 0, 2, 0],
        [0, 0, -1.5, 0, 0],
    ], dtype=np.float64) / 8

    # Whole-sample reflection preserves Bayer parity beyond each image edge.
    # Apply complete MHC kernels, rather than combining incompatible padding
    # policies for a bilinear base and its gradient correction.
    greenCorrection = convolve(bayerMosaic, malvarGreenKernel, mode="mirror")
    horizontalCorrection = convolve(bayerMosaic, malvarHorizontalKernel, mode="mirror")
    verticalCorrection = convolve(bayerMosaic, malvarHorizontalKernel.T, mode="mirror")
    oppositeCorrection = convolve(bayerMosaic, malvarOppositeKernel, mode="mirror")

    redMask = np.zeros(bayerMosaic.shape, dtype=bool)
    blueMask = np.zeros(bayerMosaic.shape, dtype=bool)
    redHorizontalMask = np.zeros(bayerMosaic.shape, dtype=bool)
    redVerticalMask = np.zeros(bayerMosaic.shape, dtype=bool)
    blueHorizontalMask = np.zeros(bayerMosaic.shape, dtype=bool)
    blueVerticalMask = np.zeros(bayerMosaic.shape, dtype=bool)
    for row, column in np.ndindex(2, 2):
        siteMask = np.zeros(bayerMosaic.shape, dtype=bool)
        siteMask[row::2, column::2] = True
        if pattern[row, column] == "R":
            redMask |= siteMask
        elif pattern[row, column] == "B":
            blueMask |= siteMask
        elif pattern[row, (column + 1) % 2] == "R":
            redHorizontalMask |= siteMask
            blueVerticalMask |= siteMask
        else:
            blueHorizontalMask |= siteMask
            redVerticalMask |= siteMask

    correctionCases = (
        (1, greenCorrection, redMask),
        (1, greenCorrection, blueMask),
        (2, oppositeCorrection, redMask),
        (0, oppositeCorrection, blueMask),
        (0, horizontalCorrection, redHorizontalMask),
        (0, verticalCorrection, redVerticalMask),
        (2, horizontalCorrection, blueHorizontalMask),
        (2, verticalCorrection, blueVerticalMask),
    )
    for channelIndex, correction, siteMask in correctionCases:
        rgbImage[..., channelIndex][siteMask] = correction[siteMask]

    return rgbImage

def plot_quarter_resolution():
    pass

def white_level_histogram(imagePath, binMin, binMax, binRange):
    """histogram the raw values and look for a spike at the top of the range indicating the white level"""
    with rawpy.imread(str(imagePath)) as rawFile:
        raw = rawFile.raw_image_visible.copy()
    hist, bin_edges = np.histogram(raw, bins=binMax-binMin, range=binRange)
    return hist, bin_edges

def extract_colorchecker_means(
    linearRgb: np.ndarray,
    *,
    interiorFraction: float = 0.5,
) -> dict:
    """Locate a 24-patch ColorChecker and measure its linear camera RGB.

    Accept an H x W x 3 floating-point RGB image. Detect on an 8-bit
    preview, but measure original pixels without clipping or interpolation.
    interiorFraction controls the retained linear size of each patch.

    Return meanRgb (24 x 3), patchPolygons (24 x 4 x 2), and
    pixelCounts (24,). Patch order follows OpenCV's MCC24 model.
    Raise ValueError for invalid inputs, detection failure, or invalid ROIs.
    Require the pinned OpenCV 4.11 contrib build; its MCC24 detector is
    validated on the assignment charts. OpenCV 5.0 detection was inconsistent.
    """
    linearRgb = np.asarray(linearRgb)
    if (
        linearRgb.ndim != 3
        or linearRgb.shape[2] != 3
        or min(linearRgb.shape[:2]) < 2
        or not np.issubdtype(linearRgb.dtype, np.floating)
    ):
        raise ValueError("linearRgb must be a floating-point H x W x 3 RGB image")
    if not np.isfinite(linearRgb).all():
        raise ValueError("linearRgb must contain only finite values")
    if not np.isfinite(interiorFraction) or not 0 < interiorFraction <= 1:
        raise ValueError("interiorFraction must be in (0, 1]")
    if not hasattr(cv2, "mcc"):
        raise RuntimeError("ColorChecker detection requires OpenCV's mcc contrib module")

    # Adjust only the detection preview; preserve the measurement array.
    previewRgb = np.clip(linearRgb, 0.0, 1.0)
    previewRgb = np.where(
        previewRgb <= 0.0031308,
        12.92 * previewRgb,
        1.055 * previewRgb ** (1 / 2.4) - 0.055,
    )
    previewBgr = cv2.cvtColor(
        np.rint(previewRgb * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )

    detector = cv2.mcc.CCheckerDetector_create()
    if not detector.process(previewBgr, cv2.mcc.MCC24):
        raise ValueError("No 24-patch ColorChecker was detected")
    checker = detector.getBestColorChecker()
    if checker is None:
        raise ValueError("The detector returned no ColorChecker")

    patchPoints = np.asarray(checker.getColorCharts(), dtype=np.float64)
    if patchPoints.size != 24 * 4 * 2 or not np.isfinite(patchPoints).all():
        raise ValueError("The detector did not return 24 valid patch polygons")
    patchPolygons = patchPoints.reshape(24, 4, 2)
    centers = patchPolygons.mean(axis=1, keepdims=True)
    patchPolygons = centers + interiorFraction * (patchPolygons - centers)

    height, width = linearRgb.shape[:2]
    meanRgb = np.empty((24, 3), dtype=np.float64)
    pixelCounts = np.empty(24, dtype=np.int64)

    for patchIndex, polygon in enumerate(patchPolygons):
        if (
            np.any(polygon < 0)
            or np.any(polygon[:, 0] > width - 1)
            or np.any(polygon[:, 1] > height - 1)
        ):
            raise ValueError(f"Patch {patchIndex + 1} extends outside the image")

        left, top = np.floor(polygon.min(axis=0)).astype(int)
        right, bottom = np.ceil(polygon.max(axis=0)).astype(int)
        patchMask = np.zeros((bottom - top + 1, right - left + 1), dtype=np.uint8)
        localPolygon = polygon - [left, top]
        cv2.fillConvexPoly(
            patchMask,
            np.rint(localPolygon * 256).astype(np.int32),
            1,
            shift=8,
        )
        patchPixels = linearRgb[top:bottom + 1, left:right + 1][patchMask != 0]
        if patchPixels.shape[0] == 0:
            raise ValueError(f"Patch {patchIndex + 1} contains no measurement pixels")

        meanRgb[patchIndex] = patchPixels.mean(axis=0, dtype=np.float64)
        pixelCounts[patchIndex] = patchPixels.shape[0]

    return {
        "meanRgb": meanRgb,
        "patchPolygons": patchPolygons,
        "pixelCounts": pixelCounts,
    }


def plot_colorchecker_patches(
    linearRgb: np.ndarray,
    patchPolygons: np.ndarray,
    ax=None,
):
    """Overlay 24 numbered measurement polygons on a linear RGB preview.

    Pass the same H x W x 3 image used by extract_colorchecker_means and its
    patchPolygons array of shape (24, 4, 2), in (x, y) coordinates.
    Clip and apply approximate display gamma only to a new preview array.
    Return the Matplotlib axes without calling show or modifying inputs.
    Raise ValueError for invalid images, polygons, or out-of-image coordinates.
    """
    from matplotlib.patches import Polygon

    linearRgb = np.asarray(linearRgb)
    patchPolygons = np.asarray(patchPolygons, dtype=np.float64)
    if (
        linearRgb.ndim != 3
        or linearRgb.shape[2] != 3
        or min(linearRgb.shape[:2]) < 1
        or not np.issubdtype(linearRgb.dtype, np.floating)
        or not np.isfinite(linearRgb).all()
    ):
        raise ValueError("linearRgb must be a finite floating-point H x W x 3 RGB image")
    if patchPolygons.shape != (24, 4, 2) or not np.isfinite(patchPolygons).all():
        raise ValueError("patchPolygons must be a finite array of shape (24, 4, 2)")
    height, width = linearRgb.shape[:2]
    if (
        np.any(patchPolygons < 0)
        or np.any(patchPolygons[..., 0] > width - 1)
        or np.any(patchPolygons[..., 1] > height - 1)
    ):
        raise ValueError("Patch polygons must lie inside the image")

    previewRgb = np.clip(linearRgb, 0, 1) ** (1 / 2.2)
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(previewRgb, origin="upper")
    for patchNumber, polygon in enumerate(patchPolygons, start=1):
        ax.add_patch(
            Polygon(polygon, closed=True, fill=False, edgecolor="lime", linewidth=1.5)
        )
        centerX, centerY = polygon.mean(axis=0)
        ax.text(
            centerX,
            centerY,
            str(patchNumber),
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1),
        )
    ax.set_title("ColorChecker measurement regions")
    ax.axis("off")
    return ax

def srgb_baseline_color_error(
    meanRgb: np.ndarray,
    referenceXYZ: np.ndarray,
    referenceIlluminant: np.ndarray,
    *,
    whiteIndex: int = 18,
    chromaticAdaptation: str = "Bradford",
) -> dict:
    """Report chart color error before fitting any camera matrix.

    meanRgb (N x 3) is linear camera RGB, e.g. extract_colorchecker_means
    output. Per-channel gains map patch whiteIndex to neutral RGB equal to
    the reference white luminance Y. The balanced RGB is treated as linear
    sRGB (no decoding), converted to XYZ with standard sRGB primaries, and
    adapted from D65 to referenceIlluminant (CIE xy, e.g. the chart's D50).
    Both XYZ sets are converted to Lab using referenceIlluminant.

    Return whiteBalanceGains (3,), balancedRgb, testXYZ, referenceLab,
    testLab (N x 3), deltaE (N,) CIEDE2000, and meanDeltaE, maxDeltaE,
    whiteDeltaE floats. Raise ValueError for invalid shapes, non-finite
    values, a bad whiteIndex, or a non-positive camera white patch.
    """
    import colour

    meanRgb = np.asarray(meanRgb, dtype=np.float64)
    referenceXYZ = np.asarray(referenceXYZ, dtype=np.float64)
    referenceIlluminant = np.asarray(referenceIlluminant, dtype=np.float64)
    if meanRgb.ndim != 2 or meanRgb.shape[1] != 3 or meanRgb.shape != referenceXYZ.shape:
        raise ValueError("meanRgb and referenceXYZ must both be N x 3 arrays")
    if referenceIlluminant.shape != (2,):
        raise ValueError("referenceIlluminant must be CIE xy chromaticity of shape (2,)")
    if not all(np.isfinite(array).all() for array in (meanRgb, referenceXYZ, referenceIlluminant)):
        raise ValueError("Inputs must contain only finite values")
    if isinstance(whiteIndex, bool) or not isinstance(whiteIndex, (int, np.integer)):
        raise ValueError("whiteIndex must be an integer")
    if not 0 <= whiteIndex < meanRgb.shape[0]:
        raise ValueError(f"whiteIndex must be in [0, {meanRgb.shape[0] - 1}]")

    cameraWhite = meanRgb[whiteIndex]
    if np.any(cameraWhite <= 0):
        raise ValueError("The camera white patch must be positive in every channel")
    whiteBalanceGains = referenceXYZ[whiteIndex, 1] / cameraWhite  # White -> neutral at reference Y
    balancedRgb = meanRgb * whiteBalanceGains

    testXYZ = colour.RGB_to_XYZ(
        balancedRgb,
        "sRGB",
        illuminant=referenceIlluminant,  # Adapt sRGB's D65 white to the chart illuminant
        chromatic_adaptation_transform=chromaticAdaptation,
    )
    referenceLab, testLab, deltaE = _chart_delta_e(referenceXYZ, testXYZ, referenceIlluminant)

    return {
        "whiteBalanceGains": whiteBalanceGains,
        "balancedRgb": balancedRgb,
        "testXYZ": testXYZ,
        "referenceLab": referenceLab,
        "testLab": testLab,
        "deltaE": deltaE,
        "meanDeltaE": float(deltaE.mean()),
        "maxDeltaE": float(deltaE.max()),
        "whiteDeltaE": float(deltaE[whiteIndex]),
    }

def _chart_delta_e(referenceXYZ, testXYZ, referenceIlluminant):
    """Return (referenceLab, testLab, CIEDE2000 deltaE) under one illuminant."""
    import colour

    referenceLab = colour.XYZ_to_Lab(referenceXYZ, illuminant=referenceIlluminant)
    testLab = colour.XYZ_to_Lab(testXYZ, illuminant=referenceIlluminant)
    deltaE = np.asarray(colour.delta_E(referenceLab, testLab, method="CIE 2000"))
    return referenceLab, testLab, deltaE

def fit_color_correction_matrix(
    meanRgb: np.ndarray,
    referenceXYZ: np.ndarray,
    referenceIlluminant: np.ndarray,
    *,
    colorGain: np.ndarray | None = None,
    clipLevel: float = 0.98,
    operateOnAllValues: bool = False,
) -> dict:
    """Least-squares fit of a 3 x 3 matrix from white-balanced RGB to XYZ.

    meanRgb (N x 3) is normalized linear camera RGB, where 1 is saturation.
    colorGain (3,) holds per-channel white-balance gains (default ones).
    Patches with any raw channel >= clipLevel are excluded from the fit,
    since clipped values do not follow the linear model.

    Solve min ||balancedRgb @ colorMatrix.T - referenceXYZ|| over the
    unclipped patches, so XYZ = colorMatrix @ (colorGain * rgb). The scale
    is absorbed by the matrix; no exposure normalization is needed.

    Return colorMatrix (3 x 3), cameraToXyz = colorMatrix @ diag(colorGain),
    fitMask (N,), testXYZ, referenceLab, testLab (N x 3), deltaE (N,)
    CIEDE2000 under referenceIlluminant, and meanDeltaE/maxDeltaE over
    fitted patches. Raise ValueError for invalid inputs, fewer than 3
    unclipped patches, or a rank-deficient fit.
    """
    meanRgb = np.asarray(meanRgb, dtype=np.float64)
    referenceXYZ = np.asarray(referenceXYZ, dtype=np.float64)
    colorGain = np.ones(3) if colorGain is None else np.asarray(colorGain, dtype=np.float64)
    if meanRgb.ndim != 2 or meanRgb.shape[1] != 3 or meanRgb.shape != referenceXYZ.shape:
        raise ValueError("meanRgb and referenceXYZ must both be N x 3 arrays")
    if colorGain.shape != (3,) or np.any(colorGain <= 0):
        raise ValueError("colorGain must contain three positive gains")
    if not all(np.isfinite(array).all() for array in (meanRgb, referenceXYZ, colorGain)):
        raise ValueError("Inputs must contain only finite values")
    
    if(operateOnAllValues):
        clipLevel = np.inf
    fitMask = np.all(meanRgb < clipLevel, axis=1)  # Clipping is judged on raw, not balanced, RGB
    if fitMask.sum() < 3:
        raise ValueError("At least 3 unclipped patches are required for the fit")
    balancedRgb = meanRgb * colorGain
    solution, _, rank, _ = np.linalg.lstsq(
        balancedRgb[fitMask], referenceXYZ[fitMask], rcond=None
    )
    if rank != 3:
        raise ValueError("Unclipped patches cannot determine a full-rank matrix")

    testXYZ = balancedRgb @ solution
    referenceLab, testLab, deltaE = _chart_delta_e(referenceXYZ, testXYZ, referenceIlluminant)
    colorMatrix = solution.T  # Column-vector form: xyz = colorMatrix @ rgb
    return {
        "colorMatrix": colorMatrix,
        "cameraToXyz": colorMatrix @ np.diag(colorGain),
        "fitMask": fitMask,
        "testXYZ": testXYZ,
        "referenceLab": referenceLab,
        "testLab": testLab,
        "deltaE": deltaE,
        "meanDeltaE": float(deltaE[fitMask].mean()),
        "maxDeltaE": float(deltaE[fitMask].max()),
    }

def summarize_delta_e(deltaE: np.ndarray, patchNames, mask=None, worstCount: int = 3) -> dict:
    """Return mean, max, and the worstCount worst (name, deltaE) over masked patches."""
    deltaE = np.asarray(deltaE, dtype=np.float64)
    mask = np.ones(deltaE.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if deltaE.ndim != 1 or mask.shape != deltaE.shape or len(patchNames) != deltaE.size:
        raise ValueError("deltaE, mask, and patchNames must all have length N")
    if not mask.any():
        raise ValueError("mask must select at least one patch")
    if isinstance(worstCount, bool) or not isinstance(worstCount, int) or worstCount < 1:
        raise ValueError("worstCount must be a positive integer")

    candidates = np.flatnonzero(mask)
    worstIndices = candidates[np.argsort(deltaE[candidates])[::-1][:worstCount]]
    return {
        "meanDeltaE": float(deltaE[mask].mean()),
        "maxDeltaE": float(deltaE[mask].max()),
        "worstPatches": [
            (int(index) + 1, patchNames[index], float(deltaE[index])) for index in worstIndices
        ],
    }

def plot_delta_e_before_after(
    beforeDeltaE: np.ndarray,
    afterDeltaE: np.ndarray,
    patchNames,
    *,
    statsMask=None,
    worstCount: int = 3,
    ax=None,
):
    """Grouped bar chart of per-patch CIEDE2000 before and after correction.

    statsMask (N,) selects the patches used for mean/max/worst statistics,
    e.g. the fit's fitMask; excluded patches are hatched and labeled in the
    plot so clipped patches remain visible but do not distort the summary.
    Dashed lines mark each condition's mean over the selected patches.

    Return (ax, summary) where summary maps "before"/"after" to
    summarize_delta_e output. Raise ValueError for mismatched inputs.
    """
    beforeDeltaE = np.asarray(beforeDeltaE, dtype=np.float64)
    afterDeltaE = np.asarray(afterDeltaE, dtype=np.float64)
    if beforeDeltaE.shape != afterDeltaE.shape:
        raise ValueError("beforeDeltaE and afterDeltaE must have the same shape")
    if statsMask is None:
        statsMask = np.ones(beforeDeltaE.shape, dtype=bool)
    statsMask = np.asarray(statsMask, dtype=bool)
    summary = {
        "before": summarize_delta_e(beforeDeltaE, patchNames, statsMask, worstCount),
        "after": summarize_delta_e(afterDeltaE, patchNames, statsMask, worstCount),
    }
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 5))

    positions = np.arange(beforeDeltaE.size)
    barWidth = 0.4
    conditions = [
        ("before", beforeDeltaE, -barWidth / 2, "tab:red"),
        ("after", afterDeltaE, barWidth / 2, "tab:blue"),
    ]
    for label, values, offset, color in conditions:
        bars = ax.bar(
            positions + offset,
            values,
            barWidth,
            color=color,
            label=f"{label.capitalize()} (mean {summary[label]['meanDeltaE']:.2f}, "
                  f"max {summary[label]['maxDeltaE']:.2f})",
        )
        for bar, isUsed in zip(bars, statsMask):
            if not isUsed:  # Hatch patches excluded from statistics
                bar.set_hatch("//")
                bar.set_alpha(0.45)
        ax.axhline(summary[label]["meanDeltaE"], color=color, linestyle="--", linewidth=1)

    tickLabels = [
        f"{index + 1} {name}" + ("" if statsMask[index] else " (clipped)")
        for index, name in enumerate(patchNames)
    ]
    ax.set_xticks(positions, tickLabels, rotation=60, ha="right", fontsize=8)
    ax.set_xlim(-0.6, beforeDeltaE.size - 0.4)
    ax.set_ylabel("CIEDE2000 ΔE")
    ax.set_title("Per-patch color error before and after matrix correction")
    ax.legend()
    ax.figure.tight_layout()
    return ax, summary

def flat_field_correct(linearRgb: np.ndarray, flatRgb: np.ndarray, roiSize: int = 64) -> dict:
    """Divide an image by a per-channel, center-normalized flat field.

    Each channel of flatRgb (H x W x 3, e.g. raw_to_linear_rgb(...)['stack'])
    is divided by the mean of its own central roiSize x roiSize window, so the
    flat equals 1 at the center in R, G, and B independently. linearRgb is
    then divided by that normalized flat, correcting color-dependent falloff
    without a single gray gain. Non-positive flat pixels yield NaN.

    Return corrected (H x W x 3), normalizedFlat (H x W x 3), and
    centerMeans (3,). Raise ValueError for mismatched or invalid inputs.
    """
    linearRgb = np.asarray(linearRgb, dtype=np.float64)
    normalizedFlat, centerMeans = normalize_flat_field(flatRgb, roiSize)
    if linearRgb.shape != normalizedFlat.shape:
        raise ValueError("linearRgb and flatRgb must have matching shapes")

    corrected = np.full_like(linearRgb, np.nan)
    np.divide(linearRgb, normalizedFlat, out=corrected, where=normalizedFlat > 0)
    return {
        "corrected": corrected,
        "normalizedFlat": normalizedFlat,
        "centerMeans": centerMeans,
    }

def normalize_flat_field(flatRgb: np.ndarray, roiSize: int = 64):
    """Scale each channel of an H x W x 3 flat to 1 at its own central ROI.

    Return (normalizedFlat, centerMeans) where centerMeans has shape (3,).
    Raise ValueError for invalid shapes, ROI sizes, or non-positive centers.
    """
    flatRgb = np.asarray(flatRgb, dtype=np.float64)
    if flatRgb.ndim != 3 or flatRgb.shape[2] != 3:
        raise ValueError("flatRgb must be an H x W x 3 RGB image")
    return _normalize_flat_planes(flatRgb, roiSize)


def _normalize_flat_planes(flatPlanes: np.ndarray, roiSize: int):
    if not isinstance(roiSize, int) or isinstance(roiSize, bool) or roiSize < 1:
        raise ValueError("roiSize must be a positive integer")
    height, width = flatPlanes.shape[:2]
    if height < roiSize or width < roiSize:
        raise ValueError(f"Images must be at least {roiSize} x {roiSize} pixels")

    top = (height - roiSize) // 2
    left = (width - roiSize) // 2
    centerMeans = flatPlanes[top:top + roiSize, left:left + roiSize].mean(axis=(0, 1))
    if not np.all(np.isfinite(centerMeans)) or np.any(centerMeans <= 0):
        raise ValueError("Flat-field center must be positive in every channel")
    return flatPlanes / centerMeans, centerMeans

def plot_flat_field_profiles(
    flatSources: dict,
    saturationValue: float | None = None,
    roiSize: int = 64,
    bandHeight: int = 16,
    ax=None,
):
    """Overlay center-row profiles of per-channel normalized white sheets.

    flatSources maps a round label to a RAW path or an H x W x 3 linear RGB
    array. Paths are converted with raw_to_linear_rgb(path, saturationValue).
    Every flat is normalized with normalize_flat_field, then averaged over
    bandHeight rows centered vertically to suppress noise. Plotted with
    seaborn lineplot: hue encodes round and style encodes channel. The x axis
    is the fraction of image width so differently sized inputs still align.

    Lens falloff is fixed to the frame, so its profiles coincide across
    rounds; lamp falloff moves with the scene and makes them differ.

    Return (ax, profiles), where profiles maps each label to a (W, 3) array.
    Raise ValueError for empty sources, missing saturationValue, or bad bands.
    """
    if not flatSources:
        raise ValueError("At least one flat-field source is required")
    if not isinstance(bandHeight, int) or isinstance(bandHeight, bool) or bandHeight < 1:
        raise ValueError("bandHeight must be a positive integer")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    profiles = {}
    profileFrames = []
    for label, source in flatSources.items():
        if isinstance(source, (str, Path)):
            if saturationValue is None:
                raise ValueError("saturationValue is required when passing RAW paths")
            source = raw_to_linear_rgb(source, saturationValue)["stack"]
        normalizedFlat, _ = normalize_flat_field(source, roiSize)
        height, width = normalizedFlat.shape[:2]
        if bandHeight > height:
            raise ValueError(f"bandHeight must not exceed image height ({height})")
        top = (height - bandHeight) // 2
        profile = normalizedFlat[top:top + bandHeight].mean(axis=0)
        profiles[label] = profile

        xPosition = (np.arange(width) + 0.5) / width
        for channelIndex, channelName in enumerate("RGB"):
            profileFrames.append(pd.DataFrame({
                "position": xPosition,
                "signal": profile[:, channelIndex],
                "round": label,
                "channel": channelName,
            }))

    sns.lineplot(
        data=pd.concat(profileFrames, ignore_index=True),
        x="position",
        y="signal",
        hue="round",
        style="channel",
        style_order=["R", "G", "B"],
        estimator=None,  # One sample per x per line; skip aggregation
        linewidth=1.2,
        ax=ax,
    )
    ax.axhline(1.0, color="gray", linewidth=0.8)
    ax.set_xlabel("Horizontal position (fraction of image width)")
    ax.set_ylabel("Normalized signal (center = 1)")
    ax.set_title("Center-row flat-field profiles by round")
    sns.move_legend(ax, "lower center", ncol=2, fontsize=8)
    return ax, profiles


## Deliverable 1: Classic Image Pipeline
