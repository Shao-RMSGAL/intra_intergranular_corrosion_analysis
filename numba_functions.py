from numba import jit
import numpy as np


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
    @jit decorator
    indicates this function is compiled for performance optimization.
    """
    x_start, x_end = sorted([x1, x2])
    y_start, y_end = sorted([y1, y2])
    roi = eds_original[y_start:y_end, x_start:x_end]
    mean_eds = np.mean(roi) / 255 * max_eds
    std_eds = np.std(roi) / 255 * max_eds
    return mean_eds, std_eds


@jit(nopython=True)
def fast_generate_eds_mask_python(threshold_percent, eds_original, max_eds,
                                  sem_mask, eds_gray):
    threshold_value = (threshold_percent / 100.0) * max_eds * (255.0 / max_eds)
    eds_mask = np.zeros(eds_original.shape[:2], dtype=np.bool_)

    for i in range(sem_mask.shape[0]):
        for j in range(sem_mask.shape[1]):
            if not sem_mask[i, j]:
                continue

            radius = 1
            max_radius = min(eds_gray.shape) // 4
            found = False

            while radius <= max_radius and not found:
                sum_value = 0.0
                count = 0
                radius_sq = radius * radius

                # Single pass: calculate average and prepare for final application
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        dist_sq = dx*dx + dy*dy
                        if dist_sq <= radius_sq:
                            ny, nx = i + dy, j + dx
                            if 0 <= ny < eds_gray.shape[0] and 0 <= nx < eds_gray.shape[1]:
                                sum_value += eds_gray[ny, nx]
                                count += 1

                if count > 0:
                    avg_value = sum_value / count
                    if avg_value >= threshold_value:
                        found = True
                        # Apply the circle to the mask immediately
                        for dy in range(-radius, radius + 1):
                            for dx in range(-radius, radius + 1):
                                if dx*dx + dy*dy <= radius_sq:
                                    ny, nx = i + dy, j + dx
                                    if 0 <= ny < eds_mask.shape[0] and 0 <= nx < eds_mask.shape[1]:
                                        eds_mask[ny, nx] = True

                radius += 1

    return eds_mask


@jit(nopython=True)
def fast_generate_intragranular_mask_python(threshold_percent, eds_original,
                                            eds_gray, max_eds, eds_mask,
                                            exclusion_mask, intragranular_mask):
    intragranular_mask = np.zeros(
        eds_original.shape[:2], dtype=np.bool_)

    # Calculate threshold value as percentage of max EDS value
    threshold_value = (threshold_percent / 100.0) * max_eds

    # Find pixels below threshold that are not in intergranular mask and not excluded
    for y in range(eds_gray.shape[0]):
        for x in range(eds_gray.shape[1]):
            if (not eds_mask[y, x] and (exclusion_mask.shape != (0, 0) and not exclusion_mask[y, x])):
                intragranular_mask[y, x] = True

    return intragranular_mask


@jit(nopython=True)
def fast_apply_masks(sem_data, sem_mask, exclusion_mask, eds_data, eds_mask,
                     intragranular_mask):
    sem_display = sem_data.copy()

    # Apply red overlay for sem_mask
    if sem_mask.size > 0:
        for i in range(sem_mask.shape[0]):
            for j in range(sem_mask.shape[1]):
                if sem_mask[i, j]:
                    sem_display[i, j, 0] = 0
                    sem_display[i, j, 1] = 0
                    sem_display[i, j, 2] = 255

    # Apply blue overlay for exclusion areas
    if exclusion_mask.size > 0:
        for i in range(exclusion_mask.shape[0]):
            for j in range(exclusion_mask.shape[1]):
                if exclusion_mask[i, j]:
                    sem_display[i, j, 0] = 255
                    sem_display[i, j, 1] = 0
                    sem_display[i, j, 2] = 0

    # Apply green overlay to EDS image for intergranular corrosion
    eds_display = eds_data.copy()
    if eds_mask.size > 0:
        for i in range(eds_mask.shape[0]):
            for j in range(eds_mask.shape[1]):
                if eds_mask[i, j]:
                    eds_display[i, j, 0] = 0
                    eds_display[i, j, 1] = 255
                    eds_display[i, j, 2] = 0

    # Apply yellow overlay for intragranular corrosion
    if intragranular_mask.size > 0:
        for i in range(intragranular_mask.shape[0]):
            for j in range(intragranular_mask.shape[1]):
                if intragranular_mask[i, j]:
                    eds_display[i, j, 0] = 0
                    eds_display[i, j, 1] = 255
                    eds_display[i, j, 2] = 255

    return sem_display, eds_display
