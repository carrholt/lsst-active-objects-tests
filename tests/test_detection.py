import numpy as np
import pytest

from lsst_aot.detection import MAX_BBOX_SIZE, MIN_BBOX_SIZE, compute_bbox_size


def test_no_detection_floors_to_min_bbox():
    image = np.zeros((50, 50))
    assert compute_bbox_size(image, noise_sigma=1.0, dilation_radius=0.0) == MIN_BBOX_SIZE


def test_bbox_size_matches_expected_extent():
    image = np.zeros((60, 60))
    # corners of a known footprint, well above the 5-sigma threshold
    for y, x in [(0, 0), (40, 0), (0, 25), (40, 25)]:
        image[y, x] = 100.0

    bbox = compute_bbox_size(image, noise_sigma=1.0, dilation_radius=5.0)
    # y extent 40 + 2*5 = 50, x extent 25 + 2*5 = 35 -> max is 50
    assert bbox == pytest.approx(50.0)


def test_bbox_size_clips_to_max():
    image = np.full((200, 200), 1000.0)
    bbox = compute_bbox_size(image, noise_sigma=1.0, dilation_radius=1000.0)
    assert bbox == MAX_BBOX_SIZE


def test_larger_dilation_increases_bbox_size():
    image = np.zeros((60, 60))
    image[0, 0] = 100.0
    image[20, 20] = 100.0  # 20-pixel extent in both x and y

    small_dilation = compute_bbox_size(image, noise_sigma=1.0, dilation_radius=0.0)
    large_dilation = compute_bbox_size(image, noise_sigma=1.0, dilation_radius=10.0)
    assert large_dilation > small_dilation


def test_threshold_sigma_controls_detection():
    image = np.zeros((60, 60))
    image[0, 0] = 100.0  # always above threshold
    noise_sigma = 2.0  # 5-sigma threshold = 10.0

    image[50, 50] = 10.5  # just above threshold -> included in footprint
    detected = compute_bbox_size(image, noise_sigma=noise_sigma, dilation_radius=0.0)
    assert detected == pytest.approx(50.0)

    image[50, 50] = 9.5  # just below threshold -> excluded
    not_detected = compute_bbox_size(image, noise_sigma=noise_sigma, dilation_radius=0.0)
    assert not_detected == MIN_BBOX_SIZE
