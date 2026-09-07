import numpy as np
import pytest

from lsst_aot.detection import MAX_BBOX_SIZE, MIN_BBOX_SIZE, compute_bbox_size


def test_no_detection_floors_to_min_bbox():
    image = np.zeros((50, 50))
    assert compute_bbox_size(image, center=(25, 25), noise_sigma=1.0, dilation_radius=0.0) == MIN_BBOX_SIZE


def test_bbox_size_matches_symmetric_extent():
    image = np.zeros((60, 60))
    center = (20, 20)
    # footprint symmetric about center, 15 px out in each direction
    for y, x in [(5, 20), (35, 20), (20, 5), (20, 35)]:
        image[y, x] = 100.0

    bbox = compute_bbox_size(image, center=center, noise_sigma=1.0, dilation_radius=5.0)
    # half-extent 15 + dilation 5 = 20, doubled -> 40
    assert bbox == pytest.approx(40.0)


def test_bbox_extends_symmetrically_even_when_footprint_is_offcenter():
    """
    A footprint entirely on one side of the centroid still forces the box
    to extend equally far on the *opposite* side too, since the cutout
    must keep the centroid at its center -- unlike a plain tight bounding
    box, which would only extend as far as the footprint itself.
    """
    image = np.zeros((100, 100))
    center = (50, 50)
    image[50, 70] = 100.0  # 20 px right of center, nothing on the left

    bbox = compute_bbox_size(image, center=center, noise_sigma=1.0, dilation_radius=0.0)
    # naive tight bbox would be ~0 (single pixel); centered box must be 2*20
    assert bbox == pytest.approx(40.0)


def test_bbox_size_clips_to_max():
    image = np.full((200, 200), 1000.0)
    bbox = compute_bbox_size(image, center=(100, 100), noise_sigma=1.0, dilation_radius=1000.0)
    assert bbox == MAX_BBOX_SIZE


def test_larger_dilation_increases_bbox_size():
    image = np.zeros((60, 60))
    center = (10, 10)
    image[10, 20] = 100.0  # 10 px from center

    small_dilation = compute_bbox_size(image, center=center, noise_sigma=1.0, dilation_radius=0.0)
    large_dilation = compute_bbox_size(image, center=center, noise_sigma=1.0, dilation_radius=10.0)
    assert large_dilation > small_dilation


def test_threshold_sigma_controls_detection():
    image = np.zeros((110, 110))
    center = (50, 50)
    image[50, 70] = 100.0  # always above threshold, 20 px from center
    noise_sigma = 2.0  # 5-sigma threshold = 10.0

    image[50, 90] = 10.5  # just above threshold -> included, 40 px from center
    detected = compute_bbox_size(image, center=center, noise_sigma=noise_sigma, dilation_radius=0.0)
    assert detected == pytest.approx(80.0)

    image[50, 90] = 9.5  # just below threshold -> excluded
    not_detected = compute_bbox_size(image, center=center, noise_sigma=noise_sigma, dilation_radius=0.0)
    assert not_detected == pytest.approx(40.0)
