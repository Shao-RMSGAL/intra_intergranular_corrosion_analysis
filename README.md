# Inter-intragranular EDS Analysis

This program uses Scanning Electron Microscopy (SEM) and Energy-Dispersive
x-ray Spectroscopy (EDS) images for chromium concentration in metallic samples
to identify intergranular and intragranular corrosion in molten-salt corroded
structural metals. First, the SEM and EDS images are loaded. The SEM files are
.tiff files, while the EDS images are .csv files with raw EDS count data. 
The EDS image is parsed and displayed on the left, while the SEM images is
shown on the right. The user can then select a TRIM that can be applied to the 
TIFF image to trim any non-image portions (metadata, for example). 

Then the user can select an area on the EDS image which is representative of
the bulk concentration of the EDS element, usually chromium in the case of 
fluoride salt corrosion. The average value of the EDS counts in this area
are calculated, and used as a baseline for corrosion calculations.

An additional feature is also provided which allows the user to select areas on
the SEM image to be excluded from the analysis. For example, the dealloying
region of a corroded sample near the surface can be excluded, because this 
region does not exhibit intergranular attack, but rather complete bulk corrosion.
Because of this, it is unecessary to apply the analysis to this region. 

In addition, the contrast-based void selection mechanism can often be confused
by darker deposits, such as molybdenum deposits, so these areas may be excluded
as well. The user will have to manually select these regions to be excluded.

Finally, once the image and selected areas are chosen, the user can then apply
the threshold option, which determines the maximum grayscale pixel value that
can be considered a void. This value ranges from 0 to 256, where 0 is black 
and 256 is white. All pixels under the chosen threshold will be considered
a void, and will be used in the subsequent analysis. The user should 
select a threshold that highlights voids, but does not highlight the bulk
material.

Once the void pixels are identified, these pixels are then mapped to the
corresponding image in the EDS data. Because of this, it is very important
that the EDS and SEM images line up perfectly. For each void pixel in the 
SEM image, a circle is drawn around the corresponding pixel in the EDS data.
This pixel starts out at 1 pixel, and the radius is expanded by incrementing
the circle radius by one-pixel increments until the mean value of the pixels
enclosed in the circle surpass some fraction of the bulk EDS concentration. 
This fraction is chosen by the user, but a value of around 5% works fairly well.

The enclosed pixels are then added to a map (shown in green) that is overlayed
onto the EDS image. These pixels are considered intergranular corrosion
areas. All remaining pixels are considered as intragranular corrosion areas.

For the plotting, the pixels are split row-by-row, and the value of the EDS 
counts in each row are averaged for the intragranular pixesl. 
This is done through the whole EDS dataset.
This is then compared to the bulk Cr concentration, and the fraction of the
bulk is calculated. This is then subtracted from unity to provide the 
intragranular Cr depletion. The process is then repeated, only for the 
intergranular corrosion pixels. This give the intergarnular depletion. 
Finally, the intergranular and intragranular depletion are summed to provide
the total Cr depletion. These are then provied four plots, one showing total 
depletion, another showing intergranular corrosion, another showing 
intagranular corrosion, and finally a plot of all three values together.

Note that there is a user-provided value for the conversion of pixels to depth,
so that the plots are of depletion versus depth in micrometers instead of 
pixels. In addition, there is a feature that bins the row data into some user-
selected number of bins which allows for a reduction in noise in the data.

## Installation and Setup

### Prerequisites

1. **Python dependencies:**
   ```bash
   pip install pandas opencv-python numpy pillow tkinter numba
   ```

