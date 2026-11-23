import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def utils_module(monkeypatch):
    helperPath = Path(__file__).resolve().parents[1] / "helper"
    monkeypatch.syspath_prepend(str(helperPath))
    spec = importlib.util.spec_from_file_location("project2_utils", helperPath / "utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalization_defects_and_saturation(utils_module):
    levels = np.array([[100, 200], [300, 400]])
    mosaic = np.tile(levels, (4, 4)).astype(np.uint16)
    dark = mosaic.copy()
    mosaic[0::2, 0::2] += 450
    mosaic[0::2, 1::2] += 400
    mosaic[1::2, 0::2] += 350
    mosaic[1::2, 1::2] += 300
    dark[0, 0] = 900
    dark[5, 5] = 0
    mosaic[0, 0] = 1000
    mosaic[5, 5] = 400
    mosaic[2, 4] = 1100
    mosaic[2, 3] = 190
    original = mosaic.copy()

    result = utils_module.correct_raw_sensor(
        mosaic, dark, levels, 1000, defectThresholdDN=100,
    )

    assert result["hotMask"].sum() == 1
    assert result["hotMask"][0, 0]
    assert result["lowMask"].sum() == 1
    assert result["lowMask"][5, 5]
    assert result["saturationMask"].sum() == 2
    assert result["saturationMask"][0, 0]
    assert result["saturationMask"][2, 4]
    assert result["corrected"][0, 0] == pytest.approx(0.5)
    assert result["corrected"][5, 5] == pytest.approx(0.5)
    assert result["corrected"][2, 4] == 1
    assert result["corrected"][2, 3] == pytest.approx(-10 / 800)
    np.testing.assert_array_equal(mosaic, original)


def test_neighboring_defects_are_excluded_from_replacement(utils_module):
    dark = np.full((10, 10), 100.0)
    mosaic = np.full((10, 10), 500.0)
    dark[4, 4] = 900
    dark[4, 6] = 900
    mosaic[4, 4] = 1000
    mosaic[4, 6] = 0

    result = utils_module.correct_raw_sensor(
        mosaic, dark, 100, 1000, defectThresholdDN=100,
    )

    assert result["hotMask"].sum() == 2
    assert result["corrected"][4, 4] == pytest.approx(400 / 900)
    assert result["corrected"][4, 6] == pytest.approx(400 / 900)


def test_separate_stages_match_combined_helper(utils_module):
    levels = np.array([[100, 200], [300, 400]])
    mosaic = np.tile(levels, (4, 4)).astype(np.uint16) + 200
    dark = np.tile(levels, (4, 4)).astype(np.uint16)
    dark[2, 2] = 900
    mosaic[2, 2] = 1000
    mosaic[1, 3] = 390

    blackCorrected = utils_module.subtract_black_level(mosaic, levels)
    saturation = utils_module.normalize_saturation(blackCorrected, levels, 1000)
    defects = utils_module.find_defective_pixels(dark, 100)
    corrected = utils_module.correct_defective_pixels(
        saturation["corrected"], defects["hotMask"] | defects["lowMask"],
    )
    combined = utils_module.correct_raw_sensor(
        mosaic, dark, levels, 1000, defectThresholdDN=100,
    )

    assert blackCorrected[1, 3] == -10
    assert saturation["corrected"][1, 3] == pytest.approx(-10 / 600)
    assert saturation["saturationMask"][2, 2]
    np.testing.assert_array_equal(corrected, combined["corrected"])
    np.testing.assert_array_equal(saturation["saturationMask"], combined["saturationMask"])
    np.testing.assert_array_equal(defects["hotMask"], combined["hotMask"])
    np.testing.assert_array_equal(defects["lowMask"], combined["lowMask"])


def test_standalone_defect_correction_rejects_invalid_mask(utils_module):
    with pytest.raises(ValueError, match="defectMask"):
        utils_module.correct_defective_pixels(np.zeros((4, 4)), np.ones((4, 4)))


def test_bayer_shading_corrects_each_site_independently(utils_module):
    flat = np.empty((8, 8))
    scene = np.empty_like(flat)
    response = np.array([
        [0.5, 0.6, 0.7, 0.8],
        [0.6, 0.8, 1.0, 0.9],
        [0.7, 1.0, 1.2, 1.0],
        [0.8, 0.9, 1.0, 1.1],
    ])
    for row, column in np.ndindex(2, 2):
        site = (slice(row, None, 2), slice(column, None, 2))
        siteResponse = response ** (1 + row * 2 + column)
        flat[site] = (1 + row * 2 + column) * siteResponse
        scene[site] = (0.25 + row + column) * siteResponse
    original = scene.copy()

    shadingMap = utils_module.build_bayer_shading_map(flat, roiSize=2, sigma=0)
    corrected = utils_module.correct_bayer_lens_shading(scene, shadingMap["normalizedFlat"])

    for row, column in np.ndindex(2, 2):
        site = (slice(row, None, 2), slice(column, None, 2))
        center = response[1:3, 1:3] ** (1 + row * 2 + column)
        np.testing.assert_allclose(
            corrected[site], (0.25 + row + column) * center.mean(),
        )
    assert corrected.max() > 1
    np.testing.assert_array_equal(scene, original)


def test_smoothing_does_not_mix_bayer_sites(utils_module):
    flat = np.ones((10, 10))
    flat[4, 4] = 10
    result = utils_module.build_bayer_shading_map(flat, roiSize=1, sigma=1)

    assert result["normalizedFlat"][4, 4] > result["normalizedFlat"][0, 0]
    for row, column in ((0, 1), (1, 0), (1, 1)):
        np.testing.assert_allclose(result["normalizedFlat"][row::2, column::2], 1)


@pytest.mark.parametrize(
    "flat,roiSize,sigma",
    [
        (np.ones((5, 4)), 1, 0),
        (np.zeros((4, 4)), 1, 0),
        (np.ones((4, 4)), 3, 0),
        (np.ones((4, 4)), 1, -1),
    ],
)
def test_invalid_bayer_shading_maps_raise(utils_module, flat, roiSize, sigma):
    with pytest.raises(ValueError):
        utils_module.build_bayer_shading_map(flat, roiSize=roiSize, sigma=sigma)


def test_lens_shading_rejects_mismatched_or_nonpositive_response(utils_module):
    with pytest.raises(ValueError, match="matching shapes"):
        utils_module.correct_bayer_lens_shading(np.ones((4, 4)), np.ones((4, 6)))
    response = np.ones((4, 4))
    response[0, 0] = 0
    with pytest.raises(ValueError, match="positive"):
        utils_module.correct_bayer_lens_shading(np.ones((4, 4)), response)


@pytest.mark.parametrize(
    "mosaic,dark,black,white,threshold",
    [
        (np.zeros((4, 4)), np.zeros((4, 5)), 0, 100, 5),
        (np.zeros((3, 4)), np.zeros((3, 4)), 0, 100, 5),
        (np.full((4, 4), np.nan), np.zeros((4, 4)), 0, 100, 5),
        (np.zeros((4, 4)), np.zeros((4, 4)), [1, 2], 100, 5),
        (np.zeros((4, 4)), np.zeros((4, 4)), 100, 100, 5),
        (np.zeros((4, 4)), np.zeros((4, 4)), 0, 100, 0),
        (np.zeros((4, 4)), np.zeros((4, 4)), 0, np.inf, 5),
    ],
)
def test_invalid_inputs_raise(utils_module, mosaic, dark, black, white, threshold):
    with pytest.raises(ValueError):
        utils_module.correct_raw_sensor(
            mosaic, dark, black, white, defectThresholdDN=threshold,
        )
