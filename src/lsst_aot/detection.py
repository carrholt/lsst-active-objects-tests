"""
Detection footprint utilities: computing the alert bounding box for a
model comet image.
"""
import numpy as np

MIN_BBOX_SIZE = 30  # pixels
MAX_BBOX_SIZE = 102  # pixels


def compute_bbox_size(image, noise_sigma, dilation_radius, threshold_sigma=5.0):
    """
    Compute bboxSize, the square alert bounding-box side length (pixels).

    The footprint is every pixel above threshold_sigma * noise_sigma. Its
    bounding box is then padded by dilation_radius on each side -- the
    same effect as dilating the footprint with a disk of that radius,
    since a disk dilation pushes each extremal edge outward by exactly
    its radius. The larger of the padded x/y extents is the bbox size,
    clipped to [MIN_BBOX_SIZE, MAX_BBOX_SIZE].
    """
    mask = image > threshold_sigma * noise_sigma
    if not np.any(mask):
        bbox_size = 0
    else:
        ys, xs = np.nonzero(mask)
        x_extent = xs.max() - xs.min() + 2 * dilation_radius
        y_extent = ys.max() - ys.min() + 2 * dilation_radius
        bbox_size = max(x_extent, y_extent)

    return np.clip(bbox_size, MIN_BBOX_SIZE, MAX_BBOX_SIZE)
