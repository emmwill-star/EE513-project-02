# EE513-project-02

## Initial RAW sensor correction

`helper/utils.py` provides independent black subtraction, saturation handling
and defect detection/correction stages. Pass visible 2D Bayer mosaics, a scalar
or visible 2-by-2 array of black levels, the measured white level, and an
explicitly chosen defect threshold in DN:

```python
blackCorrected = utils.subtract_black_level(sceneMosaic, siteBlackLevels)
saturation = utils.normalize_saturation(blackCorrected, siteBlackLevels, whiteLevel)
defects = utils.find_defective_pixels(darkMosaic, 8 * measuredReadNoiseDN)
correctedMosaic = utils.correct_defective_pixels(
    saturation["corrected"], defects["hotMask"] | defects["lowMask"],
)
```

The stages can be skipped or reordered for controlled experiments.
`correct_raw_sensor` remains available as a convenience wrapper for these same
stages in the order shown. The saturation mask describes original samples at or
above white, including defective samples.

Obtain the mosaics with `rawpy.imread(...).raw_image_visible.copy()` and the
site black levels using `raw.black_level_per_channel` indexed by
`raw.raw_colors_visible[:2, :2]`. Verify that the dark and scene captures use
the same sensor, lens, crop and visible CFA phase; select a dark frame at
appropriate ISO, exposure and temperature. The Project 1 ISO 40 read noise is
approximately 1.798 DN RMS; remeasure or validate it for other capture settings.

Defect detection compares each dark pixel with the local median of its own Bayer
site and returns hot and low *candidate* masks, plus a mask of original samples
at or above white. It replaces candidates with non-defective same-site
neighbors. It retains negative black-subtracted values and clips at white before
replacement. Check candidates against additional dark frames: one frame can
contain noise outliers, and a pixel stuck at the normal black level cannot be
identified from a dark frame alone. A flat field at a different signal level is
needed to detect that case.

An initial check of the Project 1 `dark2.DNG` capture (ISO 40, 1 s) with
`defectThresholdDN=8 * 1.798` found three high candidates at visible
row/column coordinates `(666, 1268)`, `(1074, 778)` and `(2662, 1898)`, and
no low candidates. These are not confirmed defects: the other available dark
frames have different exposure times. Repeat the check with multiple matched
dark frames before treating the candidates as a reusable defect map.

From the course repository root, run the focused tests with:

```text
uv run --no-sync pytest -q projects/02_project/tests/test_sensor_correction.py
```

## Bayer lens shading

The notebook builds a provisional map from Project 1's
`images/IMG_0297.DNG`. It expects the Project 1 and Project 2 directories to
be siblings; change `project1FlatPath` if the capture is stored elsewhere.
Each of the four Bayer-site planes is smoothed and normalized to its own
central mean before correcting the scene:

```python
flatBlackCorrected = utils.subtract_black_level(flatMosaic, flatBlackLevels)
shadingMap = utils.build_bayer_shading_map(flatBlackCorrected, roiSize=64, sigma=32)
shadingCorrectedMosaic = utils.correct_bayer_lens_shading(
    normalizedMosaic, shadingMap["normalizedFlat"],
)
```

`roiSize` and `sigma` are in half-resolution Bayer-site pixels. The notebook
checks the flat's lens, crop, visible CFA phase and saturation; it plots each
site's response for inspection. Correction does not clip the scene, so values
above one are retained. The original sensor saturation mask remains relevant.

The Project 1 flat shows strong, asymmetric falloff. A single flat cannot
separate lens shading from uneven illumination; compare additional independent
uniformly illuminated captures and recapture if needed before relying on this
map or using it for color calibration. In an initial check using Project 2's
`IMG_0293.DNG` as a second flat, corrected corner-to-center ratios range from
about 0.71 to 1.79 across sites. The second flat may itself have uneven
illumination, so this is not ground truth, but it does not validate the map.
The current notebook skips defect correction until an appropriate matched dark
frame is available.