import pandas as pd
import cv2
import numpy as np
from io import StringIO

# Load a CSV file containing EDS data from a filepath. Assumes a square image.


def load_csv(filepath):

    # Clean training commas
    with open(filepath, "r") as f:
        lines = [line.rstrip(',\n') + '\n' for line in f]

    # Read CSV without headers
    eds_df = pd.read_csv(StringIO(''.join(lines)), header=None)
    matrix = eds_df.values

    # Check shape of the matrix.
    shape = np.shape(matrix)
    if shape[0] != shape[1]:
        raise ValueError(
            f"Incorrect dimensions of matrix. Must be square. Shape is {shape}")

    # Convert to float first to handle any potential NaN values
    if matrix.min() < 0:
        raise ValueError("Matrix contains negative values.")

    # Handle NaN values (replace with 0 or interpolate)
    if np.isnan(matrix).any():
        raise ValueError("Matrix contains nan values.")

    #  # Normalize the data to 0-255 range if needed
    #  if matrix.max() > 255 or matrix.min() < 0:
    #      matrix = matrix - matrix.min()  # Shift to 0
    #      matrix = (255 * (matrix / matrix.max())).astype(np.uint8)
    #  else:
    #      matrix = matrix.astype(np.uint8)

    cv2.imshow('Matrix', matrix * 255 / matrix.max())
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Global variables
drawing = False
ix, iy = -1, -1  # initial coordinates
fx, fy = -1, -1  # final coordinates


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, img, img_copy


#  def load_sem(filepath):
load_csv("./assets/original/Cr Kα1.csv")
