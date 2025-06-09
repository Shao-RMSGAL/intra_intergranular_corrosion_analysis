import math

import numpy as np
from numba import cuda, jit, roc

# GPU kernel for EDS mask generation


@cuda.jit

def cuda_generate_eds_mask_kernel(threshold_value, eds_gray, max_radius, sem_mask, eds_mask):
    """CUDA kernel for generating EDS mask with circular filtering"""
    i, j = cuda.grid(2)

    if i >= sem_mask.shape[0] or j >= sem_mask.shape[1]:
        return

    if not sem_mask[i, j]:
        return

    radius = 1
    found = False

    while radius <= max_radius and not found:
        sum_value = 0.0
        count = 0
        radius_sq = radius * radius

        # Calculate average in circular region
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
                # Apply the circle to the mask
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dx*dx + dy*dy <= radius_sq:
                            ny, nx = i + dy, j + dx
                            if 0 <= ny < eds_mask.shape[0] and 0 <= nx < eds_mask.shape[1]:
                                eds_mask[ny, nx] = True

        radius += 1

# ROCm kernel (same logic as CUDA)


@roc.jit
def roc_generate_eds_mask_kernel(threshold_value, eds_gray, max_radius, sem_mask, eds_mask):
    """ROCm kernel for generating EDS mask with circular filtering"""
    i, j = roc.grid(2)

    if i >= sem_mask.shape[0] or j >= sem_mask.shape[1]:
        return

    if not sem_mask[i, j]:
        return

    radius = 1
    found = False

    while radius <= max_radius and not found:
        sum_value = 0.0
        count = 0
        radius_sq = radius * radius

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
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dx*dx + dy*dy <= radius_sq:
                            ny, nx = i + dy, j + dx
                            if 0 <= ny < eds_mask.shape[0] and 0 <= nx < eds_mask.shape[1]:
                                eds_mask[ny, nx] = True

        radius += 1

# GPU kernel for intragranular mask generation


@cuda.jit
def cuda_generate_intragranular_mask_kernel(eds_mask, exclusion_mask, intragranular_mask):
    """CUDA kernel for generating intragranular mask"""
    i, j = cuda.grid(2)

    if i >= eds_mask.shape[0] or j >= eds_mask.shape[1]:
        return

    if not eds_mask[i, j]:
        if exclusion_mask.shape[0] == 0 or not exclusion_mask[i, j]:
            intragranular_mask[i, j] = True


@roc.jit
def roc_generate_intragranular_mask_kernel(eds_mask, exclusion_mask, intragranular_mask):
    """ROCm kernel for generating intragranular mask"""
    i, j = roc.grid(2)

    if i >= eds_mask.shape[0] or j >= eds_mask.shape[1]:
        return

    if not eds_mask[i, j]:
        if exclusion_mask.shape[0] == 0 or not exclusion_mask[i, j]:
            intragranular_mask[i, j] = True

# GPU kernel for applying masks


@cuda.jit
def cuda_apply_masks_kernel(sem_data, sem_mask, exclusion_mask, eds_data,
                            eds_mask, intragranular_mask, sem_display, eds_display):
    """CUDA kernel for applying color overlays to masks"""
    i, j = cuda.grid(2)

    if i >= sem_data.shape[0] or j >= sem_data.shape[1]:
        return

    # Copy original data
    for c in range(3):
        sem_display[i, j, c] = sem_data[i, j, c]
        eds_display[i, j, c] = eds_data[i, j, c]

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

    # Apply green overlay for intergranular corrosion
    if eds_mask.size > 0 and eds_mask[i, j]:
        eds_display[i, j, 0] = 0
        eds_display[i, j, 1] = 255
        eds_display[i, j, 2] = 0

    # Apply yellow overlay for intragranular corrosion
    if intragranular_mask.size > 0 and intragranular_mask[i, j]:
        eds_display[i, j, 0] = 0
        eds_display[i, j, 1] = 255
        eds_display[i, j, 2] = 255


@roc.jit
def roc_apply_masks_kernel(sem_data, sem_mask, exclusion_mask, eds_data,
                           eds_mask, intragranular_mask, sem_display, eds_display):
    """ROCm kernel for applying color overlays to masks"""
    i, j = roc.grid(2)

    if i >= sem_data.shape[0] or j >= sem_data.shape[1]:
        return

    # Copy original data
    for c in range(3):
        sem_display[i, j, c] = sem_data[i, j, c]
        eds_display[i, j, c] = eds_data[i, j, c]

    # Apply overlays (same logic as CUDA)
    if sem_mask.size > 0 and sem_mask[i, j]:
        sem_display[i, j, 0] = 0
        sem_display[i, j, 1] = 0
        sem_display[i, j, 2] = 255

    if exclusion_mask.size > 0 and exclusion_mask[i, j]:
        sem_display[i, j, 0] = 255
        sem_display[i, j, 1] = 0
        sem_display[i, j, 2] = 0

    if eds_mask.size > 0 and eds_mask[i, j]:
        eds_display[i, j, 0] = 0
        eds_display[i, j, 1] = 255
        eds_display[i, j, 2] = 0

    if intragranular_mask.size > 0 and intragranular_mask[i, j]:
        eds_display[i, j, 0] = 0
        eds_display[i, j, 1] = 255
        eds_display[i, j, 2] = 255

# GPU reduction kernel for statistics calculation


@cuda.jit
def cuda_calculate_statistics_kernel(roi, results):
    """CUDA kernel for calculating mean and std using reduction"""
    i = cuda.grid(1)
    if i >= roi.size:
        return

    # Simple reduction - each thread contributes to sum and sum_sq
    val = roi.flat[i]
    cuda.atomic.add(results, 0, val)  # sum
    cuda.atomic.add(results, 1, val * val)  # sum of squares
    cuda.atomic.add(results, 2, 1.0)  # count


@roc.jit
def roc_calculate_statistics_kernel(roi, results):
    """ROCm kernel for calculating mean and std using reduction"""
    i = roc.grid(1)
    if i >= roi.size:
        return

    val = roi.flat[i]
    roc.atomic.add(results, 0, val)
    roc.atomic.add(results, 1, val * val)
    roc.atomic.add(results, 2, 1.0)

# Helper function to detect GPU backend


def detect_gpu_backend():
    """Detect available GPU backend (CUDA or ROCm)"""
    try:
        cuda.detect()
        return 'cuda'
    except:
        try:
            roc.detect()
            return 'rocm'
        except:
            return None

# Main GPU-optimized functions


def gpu_calculate_statistics(x1, y1, x2, y2, eds_original, max_eds):
    """GPU-accelerated statistics calculation"""
    backend = detect_gpu_backend()

    x_start, x_end = sorted([x1, x2])
    y_start, y_end = sorted([y1, y2])
    roi = eds_original[y_start:y_end, x_start:x_end].astype(np.float32)

    if backend == 'cuda':
        # CUDA implementation
        d_roi = cuda.to_device(roi)
        d_results = cuda.to_device(np.zeros(3, dtype=np.float32))

        threads_per_block = 256
        blocks_per_grid = (roi.size + threads_per_block -
                           1) // threads_per_block

        cuda_calculate_statistics_kernel[blocks_per_grid, threads_per_block](
            d_roi, d_results)
        results = d_results.copy_to_host()

    elif backend == 'rocm':
        # ROCm implementation
        d_roi = roc.to_device(roi)
        d_results = roc.to_device(np.zeros(3, dtype=np.float32))

        threads_per_block = 256
        blocks_per_grid = (roi.size + threads_per_block -
                           1) // threads_per_block

        roc_calculate_statistics_kernel[blocks_per_grid, threads_per_block](
            d_roi, d_results)
        results = d_results.copy_to_host()

    else:
        # Fallback to CPU
        return fast_calculate_statistics(x1, y1, x2, y2, eds_original, max_eds)

    # Calculate mean and std from reduction results
    total_sum, sum_sq, count = results
    mean_val = total_sum / count
    std_val = math.sqrt((sum_sq / count) - (mean_val * mean_val))

    mean_eds = mean_val / 255 * max_eds
    std_eds = std_val / 255 * max_eds

    return mean_eds, std_eds


def gpu_generate_eds_mask(threshold_percent, eds_original, max_eds, sem_mask, eds_gray):
    """GPU-accelerated EDS mask generation"""
    backend = detect_gpu_backend()

    threshold_value = (threshold_percent / 100.0) * max_eds * (255.0 / max_eds)
    eds_mask = np.zeros(eds_original.shape[:2], dtype=np.bool_)
    max_radius = min(eds_gray.shape) // 4

    if backend == 'cuda':
        # CUDA implementation
        d_eds_gray = cuda.to_device(eds_gray.astype(np.float32))
        d_sem_mask = cuda.to_device(sem_mask)
        d_eds_mask = cuda.to_device(eds_mask)

        threads_per_block = (16, 16)
        blocks_per_grid_x = (
            sem_mask.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (
            sem_mask.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        cuda_generate_eds_mask_kernel[blocks_per_grid, threads_per_block](
            threshold_value, d_eds_gray, max_radius, d_sem_mask, d_eds_mask)

        return d_eds_mask.copy_to_host()

    elif backend == 'rocm':
        # ROCm implementation
        d_eds_gray = roc.to_device(eds_gray.astype(np.float32))
        d_sem_mask = roc.to_device(sem_mask)
        d_eds_mask = roc.to_device(eds_mask)

        threads_per_block = (16, 16)
        blocks_per_grid_x = (
            sem_mask.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (
            sem_mask.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        roc_generate_eds_mask_kernel[blocks_per_grid, threads_per_block](
            threshold_value, d_eds_gray, max_radius, d_sem_mask, d_eds_mask)

        return d_eds_mask.copy_to_host()

    else:
        # Fallback to CPU
        return fast_generate_eds_mask_python(threshold_percent, eds_original, max_eds, sem_mask, eds_gray)


def gpu_generate_intragranular_mask(threshold_percent, eds_original, eds_gray,
                                    max_eds, eds_mask, exclusion_mask, intragranular_mask):
    """GPU-accelerated intragranular mask generation"""
    backend = detect_gpu_backend()

    intragranular_mask = np.zeros(eds_original.shape[:2], dtype=np.bool_)

    if backend == 'cuda':
        d_eds_mask = cuda.to_device(eds_mask)
        d_exclusion_mask = cuda.to_device(exclusion_mask)
        d_intragranular_mask = cuda.to_device(intragranular_mask)

        threads_per_block = (16, 16)
        blocks_per_grid_x = (
            eds_mask.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (
            eds_mask.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        cuda_generate_intragranular_mask_kernel[blocks_per_grid, threads_per_block](
            d_eds_mask, d_exclusion_mask, d_intragranular_mask)

        return d_intragranular_mask.copy_to_host()

    elif backend == 'rocm':
        d_eds_mask = roc.to_device(eds_mask)
        d_exclusion_mask = roc.to_device(exclusion_mask)
        d_intragranular_mask = roc.to_device(intragranular_mask)

        threads_per_block = (16, 16)
        blocks_per_grid_x = (
            eds_mask.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (
            eds_mask.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        roc_generate_intragranular_mask_kernel[blocks_per_grid, threads_per_block](
            d_eds_mask, d_exclusion_mask, d_intragranular_mask)

        return d_intragranular_mask.copy_to_host()

    else:
        # Fallback to CPU
        return fast_generate_intragranular_mask_python(threshold_percent, eds_original,
                                                       eds_gray, max_eds, eds_mask,
                                                       exclusion_mask, intragranular_mask)


def gpu_apply_masks(sem_data, sem_mask, exclusion_mask, eds_data, eds_mask, intragranular_mask):
    """GPU-accelerated mask application"""
    backend = detect_gpu_backend()

    sem_display = np.zeros_like(sem_data)
    eds_display = np.zeros_like(eds_data)

    if backend == 'cuda':
        d_sem_data = cuda.to_device(sem_data)
        d_sem_mask = cuda.to_device(sem_mask)
        d_exclusion_mask = cuda.to_device(exclusion_mask)
        d_eds_data = cuda.to_device(eds_data)
        d_eds_mask = cuda.to_device(eds_mask)
        d_intragranular_mask = cuda.to_device(intragranular_mask)
        d_sem_display = cuda.to_device(sem_display)
        d_eds_display = cuda.to_device(eds_display)

        threads_per_block = (16, 16)
        blocks_per_grid_x = (
            sem_data.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (
            sem_data.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        cuda_apply_masks_kernel[blocks_per_grid, threads_per_block](
            d_sem_data, d_sem_mask, d_exclusion_mask, d_eds_data,
            d_eds_mask, d_intragranular_mask, d_sem_display, d_eds_display)

        return d_sem_display.copy_to_host(), d_eds_display.copy_to_host()

    elif backend == 'rocm':
        d_sem_data = roc.to_device(sem_data)
        d_sem_mask = roc.to_device(sem_mask)
        d_exclusion_mask = roc.to_device(exclusion_mask)
        d_eds_data = roc.to_device(eds_data)
        d_eds_mask = roc.to_device(eds_mask)
        d_intragranular_mask = roc.to_device(intragranular_mask)
        d_sem_display = roc.to_device(sem_display)
        d_eds_display = roc.to_device(eds_display)

        threads_per_block = (16, 16)
        blocks_per_grid_x = (
            sem_data.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (
            sem_data.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        roc_apply_masks_kernel[blocks_per_grid, threads_per_block](
            d_sem_data, d_sem_mask, d_exclusion_mask, d_eds_data,
            d_eds_mask, d_intragranular_mask, d_sem_display, d_eds_display)

        return d_sem_display.copy_to_host(), d_eds_display.copy_to_host()

    else:
        # Fallback to CPU
        return fast_apply_masks(sem_data, sem_mask, exclusion_mask, eds_data,
                                eds_mask, intragranular_mask)

# Keep original CPU functions as fallbacks


@jit(nopython=True)
def fast_calculate_statistics(x1, y1, x2, y2, eds_original, max_eds):
    """Original CPU version as fallback"""
    x_start, x_end = sorted([x1, x2])
    y_start, y_end = sorted([y1, y2])
    roi = eds_original[y_start:y_end, x_start:x_end]
    mean_eds = np.mean(roi) / 255 * max_eds
    std_eds = np.std(roi) / 255 * max_eds
    return mean_eds, std_eds


@jit(nopython=True)
def fast_generate_eds_mask_python(threshold_percent, eds_original, max_eds,
                                  sem_mask, eds_gray):
    """Original CPU version as fallback"""
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
    """Original CPU version as fallback"""
    intragranular_mask = np.zeros(eds_original.shape[:2], dtype=np.bool_)

    for y in range(eds_gray.shape[0]):
        for x in range(eds_gray.shape[1]):
            if (not eds_mask[y, x] and (exclusion_mask.shape != (0, 0) and not exclusion_mask[y, x])):
                intragranular_mask[y, x] = True

    return intragranular_mask


@jit(nopython=True)
def fast_apply_masks(sem_data, sem_mask, exclusion_mask, eds_data, eds_mask,
                     intragranular_mask):
    """Original CPU version as fallback"""
    sem_display = sem_data.copy()

    if sem_mask.size > 0:
        for i in range(sem_mask.shape[0]):
            for j in range(sem_mask.shape[1]):
                if sem_mask[i, j]:
                    sem_display[i, j, 0] = 0
                    sem_display[i, j, 1] = 0
                    sem_display[i, j, 2] = 255

    if exclusion_mask.size > 0:
        for i in range(exclusion_mask.shape[0]):
            for j in range(exclusion_mask.shape[1]):
                if exclusion_mask[i, j]:
                    sem_display[i, j, 0] = 255
                    sem_display[i, j, 1] = 0
                    sem_display[i, j, 2] = 0

    eds_display = eds_data.copy()
    if eds_mask.size > 0:
        for i in range(eds_mask.shape[0]):
            for j in range(eds_mask.shape[1]):
                if eds_mask[i, j]:
                    eds_display[i, j, 0] = 0
                    eds_display[i, j, 1] = 255
                    eds_display[i, j, 2] = 0

    if intragranular_mask.size > 0:
        for i in range(intragranular_mask.shape[0]):
            for j in range(intragranular_mask.shape[1]):
                if intragranular_mask[i, j]:
                    eds_display[i, j, 0] = 0
                    eds_display[i, j, 1] = 255
                    eds_display[i, j, 2] = 255

    return sem_display, eds_display


# Example usage and benchmarking
if __name__ == "__main__":
    import time


    # Example data
    height, width = 1024, 1024
    eds_original = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    eds_gray = eds_original.astype(np.float32)
    sem_mask = np.random.choice([True, False], (height, width))
    exclusion_mask = np.random.choice([True, False], (height, width))

    # Create sample color images
    sem_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    eds_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    print(f"GPU Backend: {detect_gpu_backend()}")

    # Benchmark GPU vs CPU
    start_time = time.time()
    gpu_mask = gpu_generate_eds_mask(
        50.0, eds_original, 255.0, sem_mask, eds_gray)
    gpu_time = time.time() - start_time
    print(f"GPU EDS mask generation: {gpu_time:.4f} seconds")

    start_time = time.time()
    cpu_mask = fast_generate_eds_mask_python(
        50.0, eds_original, 255.0, sem_mask, eds_gray)
    cpu_time = time.time() - start_time
    print(f"CPU EDS mask generation: {cpu_time:.4f} seconds")

    if cpu_time > 0:
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")
