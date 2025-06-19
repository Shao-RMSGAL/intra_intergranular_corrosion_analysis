import numpy as np
from numba import jit, prange


@jit(nopython=True)
def fast_calculate_statistics(x1, y1, x2, y2, eds_original, max_eds):
    """Calculate mean and standard deviation statistics for a rectangular ROI
    in EDS data.

    Extracts a rectangular region of interest from the original EDS data array
    and computes normalized statistical measures scaled to the maximum EDS
    value.

    Parameters
    ----------
    x1, x2 : int
        X-coordinates defining the horizontal bounds of the ROI rectangle.
    y1, y2 : int
        Y-coordinates defining the vertical bounds of the ROI rectangle.
    eds_original : numpy.ndarray
        2D array containing the original EDS data values, typically in range
        [0, 255].
    max_eds : float
        Maximum EDS value used for scaling the normalized statistics.

    Returns
    -------
    tuple[float, float]
        A tuple containing (mean_eds, std_eds) where:
        - mean_eds: Mean EDS value in the ROI, normalized and scaled to max_eds
        - std_eds: Standard deviation of EDS values in the ROI, normalized and
        scaled to max_eds

    Notes
    -----
    The function assumes the input EDS data is in the range [0, 255] and
    normalizes by dividing by 255 before scaling to the maximum EDS value. The
    @jit decorator indicates this function is compiled for performance optimization.
    """
    x_start, x_end = sorted([x1, x2])
    y_start, y_end = sorted([y1, y2])
    roi = eds_original[y_start:y_end, x_start:x_end]
    mean_eds = np.mean(roi) / 255 * max_eds
    std_eds = np.std(roi) / 255 * max_eds
    return mean_eds, std_eds


@jit(nopython=True, parallel=True, cache=True)
def fast_generate_intragranular_mask(
    eds_original,
    eds_gray,
    eds_mask,
    exclusion_mask,
    intragranular_mask,
):
    intragranular_mask = np.zeros(eds_original.shape[:2], dtype=np.bool_)

    height, width = eds_gray.shape
    has_exclusion = exclusion_mask.shape != (0, 0)

    # Parallelize over rows
    for y in prange(height):
        for x in range(width):
            if not eds_mask[y, x]:
                if has_exclusion:
                    if not exclusion_mask[y, x]:
                        intragranular_mask[y, x] = True
                else:
                    intragranular_mask[y, x] = True

    return intragranular_mask


@jit(nopython=True, parallel=True, cache=True)
def fast_apply_masks(
    sem_data, sem_mask, exclusion_mask, eds_data, eds_mask, intragranular_mask
):
    sem_display = sem_data.copy()
    eds_display = eds_data.copy()

    height, width = sem_data.shape[:2]

    # Parallelize mask application over rows
    for i in prange(height):
        for j in range(width):
            # Apply red overlay for sem_mask
            if sem_mask.size > 0 and sem_mask[i, j]:
                sem_display[i, j, 0] = 0
                sem_display[i, j, 1] = 0
                sem_display[i, j, 2] = 255

            # Apply blue overlay for exclusion areas
            if exclusion_mask.size > 0 and exclusion_mask[i, j]:
                sem_display[i, j, 0] = 255
                sem_display[i, j, 1] = 0
                sem_display[i, j, 2] = 0

            # Apply green overlay to EDS image for intergranular corrosion
            if eds_mask.size > 0 and eds_mask[i, j]:
                eds_display[i, j, 0] = 0
                eds_display[i, j, 1] = 255
                eds_display[i, j, 2] = 0

            # Apply yellow overlay for intragranular corrosion
            if intragranular_mask.size > 0 and intragranular_mask[i, j]:
                eds_display[i, j, 0] = 0
                eds_display[i, j, 1] = 255
                eds_display[i, j, 2] = 255

    return sem_display, eds_display


# Additional optimized helper functions for better performance
@jit(nopython=True, parallel=True, cache=True)
def fast_circular_average(eds_gray, center_points, radius, exclusion_mask):
    """Compute circular averages around multiple center points in parallel."""
    n_points = center_points.shape[0]
    averages = np.zeros(n_points)

    radius_sq = radius * radius

    for idx in prange(n_points):
        i, j = center_points[idx]
        sum_value = 0.0
        count = 0

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius_sq:
                    ny, nx = i + dy, j + dx
                    if 0 <= ny < eds_gray.shape[0] and 0 <= nx < eds_gray.shape[1] and not exclusion_mask[ny, nx]:
                        sum_value += eds_gray[ny, nx]
                        count += 1

        if count > 0:
            averages[idx] = sum_value / count
        else:
            averages[idx] = 0.0

    return averages


@ jit(nopython=True, cache=True)
def fast_generate_eds_mask(
    threshold_percent, eds_original, mean_eds, sem_mask, eds_gray, exclusion_mask
):
    """Optimized version using batch processing for better parallelization."""
    threshold_value = (threshold_percent / 100.0) * mean_eds * 255.0
    eds_mask = np.zeros(eds_original.shape[:2], dtype=np.bool_)

    # Collect all valid center points
    center_points = []
    for i in range(sem_mask.shape[0]):
        for j in range(sem_mask.shape[1]):
            if sem_mask[i, j]:
                center_points.append((i, j))

    if len(center_points) == 0:
        return eds_mask

    center_points_array = np.array(center_points)
    max_radius = min(eds_gray.shape) // 4
    has_exclusion = exclusion_mask.shape != (0, 0)

    # Process each radius level
    processed = np.zeros(len(center_points), dtype=np.bool_)

    for radius in range(1, max_radius + 1):
        # Find unprocessed points
        unprocessed_indices = []
        unprocessed_points = []

        for idx in range(len(center_points)):
            if not processed[idx]:
                unprocessed_indices.append(idx)
                unprocessed_points.append(center_points[idx])

        if len(unprocessed_points) == 0:
            break

        unprocessed_array = np.array(unprocessed_points)

        # Compute averages for all unprocessed points at this radius
        averages = fast_circular_average(
            eds_gray, unprocessed_array, radius, exclusion_mask)

        # Determine which points meet the threshold
        meets_threshold = averages >= threshold_value

        # Update mask for points that meet threshold
        if np.any(meets_threshold):
            valid_points = unprocessed_array[meets_threshold]
            valid_values = np.ones(len(valid_points), dtype=np.bool_)
            fast_batch_mask_update_with_exclusion(
                eds_mask,
                valid_points,
                radius,
                valid_values,
                exclusion_mask,
                has_exclusion,
            )

            # Mark these points as processed
            for i, meets in enumerate(meets_threshold):
                if meets:
                    processed[unprocessed_indices[i]] = True

    return eds_mask


@ jit(nopython=True, parallel=True, cache=True)
def fast_batch_mask_update_with_exclusion(
    mask, center_points, radius, values, exclusion_mask, has_exclusion
):
    """Update mask with circular regions around center points in parallel, excluding pixels in exclusion_mask."""
    n_points = center_points.shape[0]
    radius_sq = radius * radius

    for idx in prange(n_points):
        if not values[idx]:  # Skip if condition not met
            continue

        i, j = center_points[idx]

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius_sq:
                    ny, nx = i + dy, j + dx
                    if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                        # Skip pixels that are in the exclusion mask
                        if has_exclusion and exclusion_mask[ny, nx]:
                            continue
                        mask[ny, nx] = True

    return mask
