# 14.2 An Everyday Image
'''
1. Default display:
'''
from scipy import datasets
import numpy as np
import matplotlib.pyplot as plt

print("\n14.2: An Everyday Image")

face = datasets.face(gray=True)

print("Problem 1: Default display of face")
plt.figure()
plt.imshow(face, cmap="gray")
plt.title("1. Default display")
plt.axis("off")

'''
2. Min/max and two rescaled displays
'''
vmin = float(face.min())
vmax = float(face.max())
print(f"\nProblem 2: Min/max and two rescaled displays\nMinimum pixel = {vmin}\nMaximum pixel = {vmax}")

# (min, max/2.0)
plt.figure()
plt.imshow(face, cmap="gray", vmin=vmin, vmax=vmax/2.0)
plt.title("2. Scaled to (min, max/2)")
plt.axis("off")

# (min*2.0, max)
plt.figure()
plt.imshow(face, cmap="gray", vmin=vmin*2.0, vmax=vmax)
plt.title("2. Scaled to (min*2, max)")
plt.axis("off")

plt.show()

'''
3. Print any 30x30 element section to the screen
'''
r0, c0 = 200, 300
block = face[r0:r0+30, c0:c0+30]
np.set_printoptions(linewidth=200, suppress=True) # https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html
print(f"\nProblem 3: Print any 30x30 element section to the screen\n30x30 block starting at (row {r0}, col {c0}):")
print(block)

'''
4) Remove 100px border and compare
'''
cropped = face[100:-100, 100:-100]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(face, cmap="gray")
plt.title("4. Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cropped, cmap="gray")
plt.title("4. Cropped (100 pixel border removed)")
plt.axis("off")

plt.tight_layout()
plt.show()

'''
5 Row-wise sort (in ascending order) and compare to original
'''
sorted_rows = face.copy()
for i in range(sorted_rows.shape[0]):
    sorted_rows[i, :] = np.sort(sorted_rows[i, :])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(face, cmap="gray")
plt.title("5. Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sorted_rows, cmap="gray")
plt.title("5. Row-wise sorted")
plt.axis("off")

plt.tight_layout()
plt.show()

'''
6. Index sorted rows
'''
# https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html
original = face  # use the original image
H, W = original.shape
weird = np.empty_like(original)

# Sort ascending
first_order = np.argsort(original[0])
weird[0] = original[0][first_order]

# For each next row i, apply the permutation that would sort row i-1 (of the ORIGINAL)
for i in range(1, H):
    order = np.argsort(original[i-1])
    weird[i] = original[i][order]

# Show original vs. index-sorted
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(original, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(weird, cmap="gray")
plt.title("Sorted by previous row's order")
plt.axis("off")

plt.tight_layout()
plt.show()

# 14.5 FITS file Exercises
'''
1 & 2. Examining Bband.fits
'''
# Load the Bband.fits file from Data/images into python. examine the hdulist carefully. 
# What sort of data does this file contain? How many separate arrays and headers?

from astropy.io import fits

# https://docs.astropy.org/en/stable/io/fits/index.html
# hdul = fits.open("Bband.fits")
# data = fits.getdata("Bband.fits")   # primary image array
# hdr  = fits.getheader("Bband.fits") # primary header
# print(data)
# print(hdul)
# print(hdr)

# https://docs.astropy.org/en/stable/io/fits/api/files.html
# https://docs.astropy.org/en/latest/io/fits/api/headers.html
header = fits.open("Bband.fits")[0].header
header.remove("HISTORY", remove_all=True) # gets rid of the long file history
header.remove("COMMENT", remove_all=True) # gets rid of associated comments (don't care)

# https://docs.astropy.org/en/stable/_modules/astropy/io/fits/header.html
print(header.tostring(sep="\n", padding=True))  # 'sep' for clean line breaks + aligned columns
print("\nProblems 1 & 2: Examining Bband.fits (see above for FITS info)")
print("\nThe Bband.fits file contains a single HDU and a single data array. The 2009x2009 array is a 2-D image of M16 " \
      "(OBJECT='M16') through the B filter (FILTER='B') stored as 32-bit floats (BITPIX = -32). The header info (from " \
      "hdulist.info) indicates that this is a PrimaryHDU with 914 cards. Image was captured on 2015-07-27T09:33:04.368 (DATE-OBS) " \
      "with a 120 second exposure (EXPTIME) on the LCOGT 2-m telescope at Haleakala Observatory using instrument fs02 " \
      "(TELESCOP='2m0-01', INSTRUME='fs02'). There are other metrics, such as the pixel scale being 0.30104 arcsec per pixel " \
      "(PIXSCALE/SECPIX=0.30104) and detector characteristics like gain (GAIN) and read noise (RDNOISE). It also contains " \
      "a ridiculously long historical documentation with associated comments.")

'''
3. Displaying Bband.fits
'''
# You should have discovered that the file contains an image. 
# Display the image, using default parameters. What do you see?

from astropy.io import fits
import matplotlib.pyplot as plt

plt.imshow(fits.getdata('Bband.fits'))
plt.show()

# the default settings do a horrendous job of displaying the image, better below

# from astropy.io import fits
# import numpy as np
# import matplotlib.pyplot as plt

# data = fits.getdata("Bband.fits")

# # https://numpy.org/doc/stable/reference/generated/numpy.percentile.html using percentile to clip outliers since I don't know the actual best min/max range
# plt.imshow(data, origin="lower", interpolation="nearest", cmap="magma", vmin=np.percentile(data, 0.5), vmax=np.percentile(data, 99.5)) # using origin="lower" so 0,0 is bottom left instead of top right
# plt.colorbar(label="counts")
# plt.title("M16 (B filter with 0.5-99.5% stretch)")
# plt.show()

# print("\nProblem 3: Displaying Bband.fits\nI see what appears to be a stellar nebula with numerous stars and clouds. I first opened the image without using any specific plt.show() parameters " \
#       "but I could only see a few faint stars. I played with the min/max percentiles a little bit and I think 0.5-99.5 is a decent render. From the FITS data, " \
#       "I know that this is an image of the M16 Eagle Nebula.")

'''
4. Statistics on the image array
'''
# Let’s now do some statistics on the image array. Calculate the mean, median, and standard deviation 
# of the elements in the array. Just for fun, calculate the vase 10 logarithm of the elements in the array. 
# There are simple numpy commands to do all of these tasks. Finally, make a histogram of the (original)
# array values (matplotlib has a quick way to do this, as does seaborn if you prefer) and output this histogram as a pdf.

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

data = fits.getdata("Bband.fits")

mean = np.mean(data)
median = np.median(data)
std = np.std(data)
log10_data = np.log10(data)

print(f"\nProblem 4: Histogram of Bband.fits\nMean={mean:.3f}\nMedian={median:.3f}\nStandard deviation={std:.3f}\nBase 10 logarithms=\n{log10_data}") # using 3 decimal places because why not

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
plt.figure()
# found .reshape method for 2-D array interpretation: https://stackoverflow.com/questions/65486157/what-does-matplotlib-hist-do-with-a-2-d-numpy-array-input
plt.hist(data.reshape(-1), bins='auto', range=(270, 430))
plt.xlim(270, 440) # mean ~ 352, left tail dies around 300, right tail has scattered points further
plt.xlabel("Counts")
plt.ylabel("Pixels")
plt.title("Bband.fits Pixel Histogram")
plt.tight_layout()
plt.savefig("bband_histogram.pdf")
plt.show()

'''
5. Histogram analysis
'''
# Look carefully at this histogram. Bearing in mind that astronomical images tend to be faint background 
# noide plus a few bright sources, use the histogram to estimate the min and max values to translate 
# the elements in the array to a colormap, and redisplay the image with these values. What do you see? 
# Hint: pick a colormap with plenty of variety.

from astropy.io import fits
import matplotlib.pyplot as plt

data = fits.getdata("Bband.fits")

# chosen by eye from the histogram
vmin, vmax = 300, 430

print("\nProblem 5: Histogram analysis\nI see many bright stars and gas clouds scattered all throughout the image. There are several very bright stars dominating certain areas, but the noise makes it difficult to differentiate the fainter objects.")

plt.imshow(data, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest") # could declare vmin/max internally but for the sake of the problem I declared them like a boss. viridis is nice
plt.colorbar(label="counts")
plt.title(f"M16 (B filter) — stretch [{vmin}, {vmax}]")
plt.tight_layout()
plt.show()

'''
6. Scale experimentation
'''
# Experiment with different python commands to achieve a pretty scaling of the image. What astronomical object 
# is this an image of? Things you might want to experiment with include:
# • Scale the image to start from zero. You can get the minimum pixel value
# of the image via minimage = np.min(image)
# • Raise the image to a power, e.g. DispData = image**(0.5)
# • Take the logarithm of the pixel values
# The best way to do this is to play with ideas - you are in essence looking for the best match of the
# relevant parts of the image histogram to the most ”informative” part of the colormap

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
# https://docs.astropy.org/en/stable/api/astropy.visualization.AsinhStretch.html way better than the other stuff I was trying
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch

image = fits.getdata("Bband.fits")

''' FIRST TRY - SUPER HUGE ULTRA MEGA WASTE OF TIME. FOUND ASINHSTRETCH FROM ASTROPY AND ITS WAY BETTER. '''
# min_image = np.min(image) # from problem example
# zero_based_image = image - min_image
# sqrt_stretch = zero_based_image**0.5 # from problem example
# # https://numpy.org/doc/stable/reference/generated/numpy.log1p.html
# log_stretch  = np.log1p(zero_based_image) # tried np.log(zero_based_image) first but log(0) problems; using log1p instead of log(1+x) # https://www.youtube.com/watch?v=y5IzIcvCM18&t=42s
# sqrt_stretch = zero_based_image ** 0.5
# plt.imshow(disp, origin="lower", cmap="magma", interpolation="nearest", vmin=0, vmax=np.percentile(log_stretch, 99.5))
# plt.colorbar(label="scaled intensity")
# plt.title("M16 (B band) — zero-shift + sqrt/log stretch")
# plt.tight_layout()
# plt.show()
''' '''

norm = ImageNormalize(image, interval=PercentileInterval(99.5), stretch=AsinhStretch(a=0.05))

print("\nProblem 6. Scale experimentation\nAfter much experimentation and furious googling, I ended up finding Astropy's own 'asinhstretch' method that works great.")
plt.imshow(image, origin="lower", cmap="magma", norm=norm, interpolation="nearest") # magma looks epic
plt.colorbar(label="stretched counts")
plt.title("M16 (B band) — asinh stretch")
plt.tight_layout()
plt.show()

'''
7. H alpha image inclusion
'''
# From the same directory, load the Halpha image into python as a separate image. Display this image with 
# an appropriate colormap scaling. Then, divide the B band image by the Halpha image to make a new image. 
# Display this new image with an appropriate scaling.

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch

b_image  = fits.getdata("Bband.fits")
ha_image = fits.getdata("Halpha.fits")

ha_image_normalize = ImageNormalize(ha_image, interval=PercentileInterval(99.5), stretch=AsinhStretch(a=0.05)) # same method as before

print("\nProblem 7: H alpha image inclusion")
plt.imshow(ha_image, origin="lower", cmap="magma", norm=ha_image_normalize, interpolation="nearest")
plt.colorbar(label="H alpha counts (stretched)")
plt.title("H alpha image (asinh + 99.5%)")
plt.tight_layout()
plt.show()
# https://numpy.org/doc/2.0/reference/generated/numpy.divide.html 
ratio_image = np.divide(b_image, ha_image, out=np.zeros_like(b_image, dtype=float), where=ha_image!=0) # need 'where=ha_image!=0' because can't divide by 0 obviously (oops)

ratio_image_normalize = ImageNormalize(ratio_image, interval=PercentileInterval(99), stretch=AsinhStretch(a=0.05))
plt.imshow(ratio_image, origin="lower", cmap="viridis", norm=ratio_image_normalize, interpolation="nearest")
plt.colorbar(label="B / H alpha (stretched)")
plt.title("B / H alpha ratio (asinh + 99%)")
plt.tight_layout()
plt.show()

'''
8. The Final FITS
'''
# Load the B band and Halpha images, as before. Multoply the Halpha image by a randomly chosen float, say α, 
# between 2.1 and 27.9. Divide the B band image by this scaled Halpha image, to create a new image,
# Let’s call it BovH.

# PART 1
from astropy.io import fits
import numpy as np

b_image  = fits.getdata("Bband.fits")
ha_image = fits.getdata("Halpha.fits")

rng = np.random.default_rng()   
alpha = float(rng.uniform(2.1, 27.9)) #https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html clips random range

# https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html
ha_scaled = alpha * ha_image.astype(float) # using astype to ensure float values
BovH = np.divide(b_image.astype(float), ha_scaled, out=np.zeros_like(b_image, dtype=float), where=ha_scaled != 0) # same method as problem 7

print(f"alpha = {alpha:.4f}")

# PART 2
# Create a new image that is the square root of BovH and multiply it by a randomly chosen number, say β, between 1.057 and 1.553. 
# Let’s call this new image BovHerr. We are going to treat this as the uncertainty array for BovH, though of course it is 
# not the uncertainty array in any realistic sense.

BovH_positive_only = np.clip(BovH, 0, None)

beta = float(rng.uniform(1.057, 1.553))

BovHerr = beta * np.sqrt(BovH_positive_only) # per the problem

print(f"beta = {beta:.4f}")

# PART 3
# Create a mask array for BovH. Create a numpy array of the same dimensions as BovH of all zeros. 
# Then. set 7 randomly chosen elements in this array equal to 1. Call this array BovHmask.

import numpy as np
# https://numpy.org/doc/stable/reference/maskedarray.generic.html

BovHmask = np.zeros_like(BovH, dtype=np.uint8) # https://numpy.org/doc/stable/user/basics.types.html unassigned integer datatype instead of boolean true/false for mask
random_pixel = rng.choice(BovHmask.size, size=7, replace=False)
BovHmask.flat[random_pixel] = 1

# PART 4
# Create a new, single FITS file that contains BovH, BovHerr, and BovH-
# mask as SEPARATE images and a header which contains the values of
# α and β (with appropriately chosen keywords). Add your name to the
# header in a final keyword.

from astropy.io import fits

# https://stackoverflow.com/questions/59270533/writing-a-new-fits-file-from-old-data-with-astropy
primary_hdr = fits.Header()
primary_hdr['ALPHA'] = (float(alpha), 'Scaling factor on Halpha')
primary_hdr['BETA'] = (float(beta),  'Multiplier for sqrt(BovH) to form BovHerr')
primary_hdr['AUTHOR'] = ('Robin Ellis Brown')

# https://docs.astropy.org/en/stable/io/fits/index.html
primary_hdu = fits.PrimaryHDU(data=BovH, header=primary_hdr)
bovherr_hdu = fits.ImageHDU(data=BovHerr, name='BOVHERR')
mask_hdu = fits.ImageHDU(data=BovHmask, name='BOVHMASK')

hdul = fits.HDUList([primary_hdu, bovherr_hdu, mask_hdu])
hdul.writeto('BovH_products.fits', overwrite=True)

print("BovH_products.fits successfully saved.")

# PART 5
# Give your newly created FITS file to another member of the class, and obtain 
# a FITS file from another member of the class (this can be the same person if you wish). 
# Import the FITS file you were given, and determine the values of α and β, and 
# the locations of the elements in the mask array that are equal to 1.

from astropy.io import fits
import numpy as np

hdul = fits.open("BovH_file_test.fits")
# https://docs.astropy.org/en/stable/io/fits/api/headers.html
# https://docs.astropy.org/en/latest/io/fits/api/hdulists.htm
alpha = float(hdul[0].header["ALPHA"])
beta = float(hdul[0].header["BETA"])
# # https://numpy.org/doc/stable/reference/generated/numpy.asarray.html same data type as my own FITS file
BovHmask_in = np.asarray(hdul["BOVHMASK"].data, dtype=np.uint8) # converting to array
# # tried np.where first, epic fail # https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html
# 12 year old thread still cooks https://stackoverflow.com/questions/15976697/difference-between-nonzeroa-wherea-and-argwherea-when-to-use-which
ones_row_column = np.argwhere(BovHmask_in == 1)

print(f"alpha = {alpha:.4f}")
print(f"beta  = {beta:.4f}")
print("mask = 1 (row, col):\n", ones_row_column)

# Output:
# alpha = 10.7829
# beta  = 1.3258
# mask = 1 (row, col):
#  [[  30  328]
#  [  41 1746]
#  [ 884  548]
#  [ 888   50]
#  [1168  909]
#  [1273 1456]
#  [1480  256]]