"""
Detection footprint utilities: computing the alert bounding box for a
model comet image.
"""
import numpy as np

MIN_BBOX_SIZE = 30  # pixels
MAX_BBOX_SIZE = 102  # pixels


def compute_bbox_size(image, center, noise_sigma, dilation_radius, threshold_sigma=5.0):
    """
    Compute bboxSize, the square alert bounding-box side length (pixels).

    The real bboxSize is the smallest square that both contains the full
    footprint (every pixel above threshold_sigma * noise_sigma) and keeps
    the source centroid at its center -- so the box must reach as far past
    center as the footprint's farthest pixel in that direction, even on
    the opposite side where nothing was detected. This can only make the
    box bigger than a plain tight bounding box, never smaller. Padded by
    dilation_radius on each side -- the same effect as dilating the
    footprint with a disk of that radius, since a disk dilation pushes
    each extremal edge outward by exactly its radius -- then clipped to
    [MIN_BBOX_SIZE, MAX_BBOX_SIZE].
    """
    mask = image > threshold_sigma * noise_sigma
    if not np.any(mask):
        bbox_size = 0
    else:
        ys, xs = np.nonzero(mask)
        half_extent = max(np.abs(ys - center[0]).max(), np.abs(xs - center[1]).max())
        bbox_size = 2 * (half_extent + dilation_radius)

    return np.clip(bbox_size, MIN_BBOX_SIZE, MAX_BBOX_SIZE)
