# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_eds_mask_fast(cnp.uint8_t[:, :] eds_gray, 
                          cnp.uint8_t[:, :] sem_mask,
                          double max_eds,
                          double threshold_percent):
    """
    Fast Cython implementation of EDS mask generation
    
    Parameters:
    -----------
    eds_gray : 2D uint8 array
        Grayscale EDS image
    sem_mask : 2D uint8 array  
        Boolean SEM mask (0 or 255)
    max_eds : double
        Maximum EDS value
    threshold_percent : double
        Threshold percentage
        
    Returns:
    --------
    eds_mask : 2D uint8 array
        Boolean EDS mask (0 or 255)
    """
    cdef int height = eds_gray.shape[0]
    cdef int width = eds_gray.shape[1]
    cdef cnp.uint8_t[:, :] eds_mask = np.zeros((height, width), dtype=np.uint8)
    
    cdef double threshold_value = (threshold_percent / 100.0) * max_eds * (255.0 / max_eds)
    cdef int max_radius = min(height, width) // 4
    
    cdef int y, x, radius
    cdef int cy, cx  # circle coordinates
    cdef int pixel_count
    cdef double sum_value, avg_value
    cdef int dy, dx, dist_sq, radius_sq
    
    # Iterate through all pixels in SEM mask
    for y in range(height):
        for x in range(width):
            if sem_mask[y, x] > 0:  # If this pixel is in the SEM mask
                # Start with radius 1 and expand until condition is met
                radius = 1
                
                while radius <= max_radius:
                    # Calculate average value in circle
                    sum_value = 0.0
                    pixel_count = 0
                    radius_sq = radius * radius
                    
                    # Iterate through bounding box of circle
                    for cy in range(max(0, y - radius), min(height, y + radius + 1)):
                        for cx in range(max(0, x - radius), min(width, x + radius + 1)):
                            dy = cy - y
                            dx = cx - x
                            dist_sq = dy * dy + dx * dx
                            
                            if dist_sq <= radius_sq:
                                sum_value += eds_gray[cy, cx]
                                pixel_count += 1
                    
                    if pixel_count > 0:
                        avg_value = sum_value / pixel_count
                        
                        # If average is >= threshold, stop expanding
                        if avg_value >= threshold_value:
                            break
                    
                    radius += 1
                
                # Add final circle to EDS mask
                radius_sq = radius * radius
                for cy in range(max(0, y - radius), min(height, y + radius + 1)):
                    for cx in range(max(0, x - radius), min(width, x + radius + 1)):
                        dy = cy - y
                        dx = cx - x
                        dist_sq = dy * dy + dx * dx
                        
                        if dist_sq <= radius_sq:
                            eds_mask[cy, cx] = 255
    
    return np.asarray(eds_mask)


@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_region_statistics_fast(cnp.uint8_t[:, :, :] roi, double max_eds):
    """
    Fast calculation of region statistics
    
    Parameters:
    -----------
    roi : 3D uint8 array
        Region of interest from EDS image
    max_eds : double
        Maximum EDS value
        
    Returns:
    --------
    tuple : (mean, std)
        Mean and standard deviation
    """
    cdef int height = roi.shape[0]
    cdef int width = roi.shape[1]
    cdef int channels = roi.shape[2]
    
    cdef double sum_val = 0.0
    cdef double sum_sq = 0.0
    cdef int total_pixels = height * width * channels
    cdef double pixel_val
    cdef int y, x, c
    
    # Calculate sum and sum of squares
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                pixel_val = roi[y, x, c] / 255.0 * max_eds
                sum_val += pixel_val
                sum_sq += pixel_val * pixel_val
    
    cdef double mean = sum_val / total_pixels
    cdef double variance = (sum_sq / total_pixels) - (mean * mean)
    cdef double std_dev = sqrt(variance) if variance > 0 else 0.0
    
    return mean, std_dev


@cython.boundscheck(False)
@cython.wraparound(False)
def create_circle_mask_fast(int height, int width, int center_x, int center_y, int radius):
    """
    Fast creation of circular mask
    
    Parameters:
    -----------
    height : int
        Image height
    width : int
        Image width
    center_x : int
        Circle center x coordinate
    center_y : int
        Circle center y coordinate
    radius : int
        Circle radius
        
    Returns:
    --------
    mask : 2D uint8 array
        Boolean mask (0 or 255)
    """
    cdef cnp.uint8_t[:, :] mask = np.zeros((height, width), dtype=np.uint8)
    cdef int y, x
    cdef int dx, dy
    cdef int radius_sq = radius * radius
    
    for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
        for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
            dx = x - center_x
            dy = y - center_y
            if dx * dx + dy * dy <= radius_sq:
                mask[y, x] = 255
    
    return np.asarray(mask)
