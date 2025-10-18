# 14.1 Imaging: A little warmup
'''
1. Create a two-dimensional, 10 × 10 element array, in which each element is equal
to zero. Then, manually set the appropriate elements in the array to equal
100.0 such that, when the array is displayed as an image it shows the digit ”4”
(it does not have to be pretty, just recognisably a 4). Print the resulting array
to the screen and then display the array to the screen as an image, and compare
the two.'''
import numpy as np
import matplotlib.pyplot as plt

print("14.1 Imaging: A Little Warmup")
print("PROBLEM 1: Two dimensional array that looks like the number 4.")

image_four = np.zeros((10,10), dtype=float)

image_four [0,3] = 100.0
image_four [1,3] = 100.0
image_four [2,3] = 100.0
image_four [3,3] = 100.0
image_four [4,3] = 100.0
image_four [5,3] = 100.0
image_four [5,4] = 100.0
image_four [5,5] = 100.0
image_four [5,6] = 100.0
image_four [0,6] = 100.0
image_four [1,6] = 100.0
image_four [2,6] = 100.0
image_four [3,6] = 100.0
image_four [4,6] = 100.0
image_four [5,6] = 100.0
image_four [6,6] = 100.0
image_four [7,6] = 100.0
image_four [8,6] = 100.0
image_four [9,6] = 100.0

print("10x10 array:")
print(image_four)
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
plt.imshow(image_four)
plt.show()

# Output:
# 14.1 Imaging: A Little Warmup
# PROBLEM 1: Create a two dimensional 10x10 array where each element = 100.0, such that when it is displayed it looks like the number 4. Also display the image.
# 10x10 array:
# [[  0.   0.   0. 100.   0.   0. 100.   0.   0.   0.]
#  [  0.   0.   0. 100.   0.   0. 100.   0.   0.   0.]
#  [  0.   0.   0. 100.   0.   0. 100.   0.   0.   0.]
#  [  0.   0.   0. 100.   0.   0. 100.   0.   0.   0.]
#  [  0.   0.   0. 100.   0.   0. 100.   0.   0.   0.]
#  [  0.   0.   0. 100. 100. 100. 100.   0.   0.   0.]
#  [  0.   0.   0.   0.   0.   0. 100.   0.   0.   0.]
#  [  0.   0.   0.   0.   0.   0. 100.   0.   0.   0.]
#  [  0.   0.   0.   0.   0.   0. 100.   0.   0.   0.]
#  [  0.   0.   0.   0.   0.   0. 100.   0.   0.   0.]]

'''
2. Same as above, but instead display the digit ’9’, and when displaying the image,
use the ’inferno’ color map where the digit itself is displayed using a color near
the center of the color map. All other ”pixels” should be a single color, from
an extreme end of the color map (your choice which).
'''
import numpy as py
import matplotlib.pyplot as plt

print("\nPROBLEM 2: Two dimensional array that looks like the number 9 (inferno color map).")

image_nine = np.zeros((10,10), dtype=float)

image_nine [0,4] = 0.5
image_nine [0,5] = 0.5
image_nine [1,3] = 0.5
image_nine [2,3] = 0.5
image_nine [3,3] = 0.5
image_nine [4,3] = 0.5
image_nine [5,4] = 0.5
image_nine [5,5] = 0.5
image_nine [5,6] = 0.5
image_nine [0,6] = 0.5
image_nine [1,6] = 0.5
image_nine [2,6] = 0.5
image_nine [3,6] = 0.5
image_nine [4,6] = 0.5
image_nine [5,6] = 0.5
image_nine [6,6] = 0.5
image_nine [7,6] = 0.5
image_nine [8,6] = 0.5
image_nine [9,6] = 0.5

print("10x10 array:")
print(image_nine)
plt.imshow(image_nine, cmap='inferno', vmin=0.0, vmax=1.0)
plt.show()

# Output:
# PROBLEM 2: Two dimensional array that looks like the number 9 (inferno color map).
# 10x10 array:
# [[0.  0.  0.  0.  0.5 0.5 0.5 0.  0.  0. ]
#  [0.  0.  0.  0.5 0.  0.  0.5 0.  0.  0. ]
#  [0.  0.  0.  0.5 0.  0.  0.5 0.  0.  0. ]
#  [0.  0.  0.  0.5 0.  0.  0.5 0.  0.  0. ]
#  [0.  0.  0.  0.5 0.  0.  0.5 0.  0.  0. ]
#  [0.  0.  0.  0.  0.5 0.5 0.5 0.  0.  0. ]
#  [0.  0.  0.  0.  0.  0.  0.5 0.  0.  0. ]
#  [0.  0.  0.  0.  0.  0.  0.5 0.  0.  0. ]
#  [0.  0.  0.  0.  0.  0.  0.5 0.  0.  0. ]
#  [0.  0.  0.  0.  0.  0.  0.5 0.  0.  0. ]]

'''
3. Generate an N × N numpy array of all zeros, where 50 < N < 200. Then,
pick a random point in this array that is at least ten picels away from the
origin in both x and y directions. After that, write a funcion that calculates
the Euclidean distance between any two coordinate positions, and returns that
distance (which can be a float). Use this function, and a color map of your
choice that is nevertheless suited to the problem, to color code every pixel in
your image by its Euclidean distance from the random;y chosen point.
'''
import numpy as np
import matplotlib.pyplot as plt

print("\nPROBLEM 3: Euclidean distance between points")

# https://stackoverflow.com/questions/74343474/reproduce-numpy-random-numbers-with-numpy-rng
rng = np.random.default_rng() # reminder https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
N = int(rng.integers(51, 200)) # setting range, 200 excludeed

yy, xx = np.indices((N, N)) # coord grids of row/col indices.  https://numpy.org/doc/stable/reference/generated/numpy.indices.html

x0 = int(rng.integers(10, N)) # random point 10 "picels" away :-) 10 inclusive
y0 = int(rng.integers(10, N)) # same ^

dist = np.hypot(xx - x0, yy - y0)  # same as sqrt((x-x0)^2 + (y-y0)^2) https://numpy.org/doc/stable/reference/generated/numpy.hypot.html

image = plt.imshow(dist)
plt.title(f"Euclidean distance from ({x0}, {y0}) in a {N}×{N} image")
plt.colorbar(image, label="Distance (pixels)") # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
plt.axis("off")
plt.show()

# Output:
# PROBLEM 3: Euclidean distance between points

'''
4. Same as above, but this time, instead of the Euclidean distance, determine the
values of each pixel via the function:
v(x, y) = asin 2πx/b + ccos 2πy/d
Experiment with the values of a,b,c,d to see what they do. Pick an appropriate
color map and values of these parameters so that the displayed result looks like
”rippling water”.
'''
# couldn't figure out how to get it to look like a single water drop, but it looks like a bunch of uniform drops :)
import numpy as np
import matplotlib.pyplot as plt

print("\nPROBLEM 4: Water ripples via v(x, y) = asin 2πx/b + ccos 2πy/d")

# grid
N = 150
y, x = np.indices((N, N))

a, b = 1, 50
c, d = 1, 50

v = a*np.sin(2*np.pi*x/b) + c*np.cos(2*np.pi*y/d)

image = plt.imshow(v, cmap="Blues")
plt.title("v(x,y) = a sin(2πx/b) + c cos(2πy/d), a & c = 1, b & d = 40")
plt.axis("off")
plt.colorbar(image)
plt.show()

# Output:
# PROBLEM 4: Water ripples via v(x, y) = asin 2πx/b + ccos 2πy/d

'''
5. Something more interesting.
'''
# Let’s end with something a bit more interesting. For this last question, increase N in your initial 2D array to around 500. 
# Then, suppose that each pixel in the array corresponds to a complex number c, such that: c = x + iy (7) (don’t try and color-code your image by c). 
# Then, consider the function: qn+1 = q2n + c (8) [that is q subscript n+1 and q squared subscript n] where n runs over the integers, and q0 = 0. 
# To choose the colors of each pixel in your image, do the following: • Set c for that pixel using Equation 7 • With that c, and starting at n = 0, iterate over Equation 8. 
# • At each iteration of Equation 8, evaluate D = √real(q)^2 + imaginary(q)^2 (9) (this is of course just the magnitude of q). 
# • If D > 2 then stop iterating, and set the value of the pixel that defined c based on the value of n at which D exceeded 2. 
# Pick a suitable color map for these pixels that has a decent range of colors and does not include black. 
# • If n reaches nmax = 1000 and you still have D < 2 then stop the iteration and set the pixxel color to black. 
# • Display the resulting image. Such a ”simple” algorithm that is not all that far away from a simple Euclidean distance should not give an image that looks too complicated16. 
# • Play with your chosen values of N, and the threshods on D and nmax. What does each do?

# pretty useful for my personal understanding of the problem https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set
# also good https://complex-analysis.com/content/mandelbrot_set.html
# even better https://realpython.com/mandelbrot-set-python/
import numpy as np
import matplotlib.pyplot as plt

print("\nProblem 5: Mandelbrot Escape Time")

N, nmax, R = 500, 1000, 2.0
x = np.linspace(-1.75, 0.75, N) # using linspace for evenly spaced numbers over n - ni https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
y = np.linspace(-1.25, 1.25, N)
# https://www.youtube.com/watch?v=7K_a1mmraHU
X, Y = np.meshgrid(x, y) # creates two dimensional 'grid' from 1d arrays x & y https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
c = X + 1j*Y # 1j is the imaginary number

# iterate qn+1 = qn^2 + c from q0 = 0
q = np.zeros_like(c) # array same shape as c set c for that pixel using Equation 7: c = x+iy
iterations = np.zeros(c.shape, dtype=int) # .shape gives size of c  as array in (x, y)

# https://www.youtube.com/watch?v=BmpsWs-kNTM
# getting a runtime warning with this loop, can't figure out how to fix it but plot still works so gg I guess?
for n in range(1, nmax + 1):
    q = q*q + c
    # mark pixels that escape for the first time at step n (D = |q| > R)
    escaped_now = (iterations == 0) & ((q.real*q.real + q.imag*q.imag) > R*R)
    iterations[escaped_now] = n

cmap = plt.cm.turbo.copy() # turbo colormap copy, otherwise changes the global colormap (oops) # https://stackoverflow.com/questions/42843669/reset-default-matplotlib-colormap-values-after-using-set-under-or-set-over?
cmap.set_bad('black') # found this set_bad method, values < vmin are black https://stackoverflow.com/questions/65322133/understanding-matplotlib-set-bad-colormap

plt.imshow(iterations, origin='lower', cmap=cmap)
plt.xlabel("Real(c)"); plt.ylabel("Imaginary(c)")
plt.colorbar(label="Escape iteration n")
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.tight_layout.html
plt.tight_layout()
plt.show() # looks good
# good thread https://stackoverflow.com/questions/79598228/how-could-i-zoom-in-on-a-generated-mandelbrot-set-without-consuming-too-many-res

# Output:
# Problem 5: Mandelbrot Escape Time
# /Users/robinbrown/repos/ASTR300L/python_modules/part3/part3_exercises.py:213: RuntimeWarning: overflow encountered in multiply
#   escaped_now = (iterations == 0) & ((q.real*q.real + q.imag*q.imag) > R*R)
# /Users/robinbrown/repos/ASTR300L/python_modules/part3/part3_exercises.py:213: RuntimeWarning: overflow encountered in add
#   escaped_now = (iterations == 0) & ((q.real*q.real + q.imag*q.imag) > R*R)
# /Users/robinbrown/repos/ASTR300L/python_modules/part3/part3_exercises.py:211: RuntimeWarning: overflow encountered in multiply
#   q = q*q + c
# /Users/robinbrown/repos/ASTR300L/python_modules/part3/part3_exercises.py:211: RuntimeWarning: invalid value encountered in multiply
#   q = q*q + c

# 14.2 An Everyday Image

from scipy import datasets
face = datasets.face(gray=True)

'''
1. Default display
'''
# 1. Display the image to the screen using all defalult parameters.

import matplotlib.pyplot as plt

plt.imshow(face)
plt.show()

'''
2. Minimum/maximum display
'''
# 2. Find the minimum and maximum pixel values in the image. Then, make two new images. Ine with the colormap scaled to (min, max/2.0). The other scaled to (min*2.0, max). 
# https://numpy.org/doc/stable/reference/generated/numpy.min.html

import numpy as np
import matplotlib.pyplot as plt

#find min and max using numpy min/max function
min_val = np.min(face)
max_val = np.max(face)

print("Min pixel value:", min_val)
print("Max pixel value:", max_val)

# display image scaled between (min, max/2)
plt.imshow(face, vmin=min_val, vmax=max_val/2.0)
plt.title("Colormap scaled to (min, max/2.0)")
plt.colorbar()
plt.show()

# display image scaled between (min*2, max)
plt.imshow(face, vmin=min_val*2.0, vmax=max_val)
plt.title("Colormap scaled to (min*2.0, max)")
plt.colorbar()
plt.show()

'''
3. 30x30 element section
'''
# 3. Print the numerical values of any 30 ×30 element section of the image to the screen.

section = face[0:30, 0:30]
print(section)

'''
4. 100 pixel border removed
'''
# 4. Make new image consisting of original image with 100 pixel border round the edges removed. Plot new image beside original image to check result.

import matplotlib.pyplot as plt

# remove 100-pixel border from all sides (top, bottom , l, r)
cropped = face[100:-100, 100:-100]

#plot original and cropped images next to each other
plt.subplot(1, 2, 1)
plt.imshow(face)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cropped)
plt.title("Cropped")
plt.axis("off")

plt.show()

'''
5. Flow control row sort
'''
# 5. Using flow control, step down through each row of image and sort each in ascenfing order of the element values. Display resulting image beside the original.

import numpy as np
import matplotlib.pyplot as plt

# create copy of image so we dont modify the original
sorted_face = face.copy()

# loop through each row and sort pixel vals in ascending order
for i in range(sorted_face.shape[0]):
    sorted_face[i] = np.sort(sorted_face[i])

# display oroginal and row-sorted images next to eachother
plt.subplot(1,2,1)
plt.imshow(face)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(sorted_face)
plt.title("Row-sorted")
plt.axis("off")

plt.show()

'''
6. Unsummarizable intructions
'''
# 6. Last problem, lots of instrucitons
# https://stackoverflow.com/questions/17901218/numpy-argsort-what-is-it-doing
# https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
# https://numpy.org/doc/stable/user/basics.indexing.html

import numpy as np
import matplotlib.pyplot as plt

# make a copy of original iamge
odd_sort_face = face.copy()

order = np.argsort(odd_sort_face[0])  # get index order that would sort the first row
odd_sort_face[0] = odd_sort_face[0][order] #apply that order to the first row itself

# for each row after, sort it using the previous's row order
for i in range(1, odd_sort_face.shape[0]):
    odd_sort_face[i] = odd_sort_face[i][order]
    order = np.argsort(odd_sort_face[i]) #update the order to the one that sorts the current row

plt.subplot(1,2,1)
plt.imshow(face)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(odd_sort_face)
plt.title("Odd-sorted")
plt.axis("off")

plt.show()

# 14.3 Imaging: Increasingly real stars
'''
1. Attempt 1 - Utter Nothingness:
'''
# Generate a two dimensional, 1024 × 1024 element array, in which each element is equal to zero. 
# Display this ”image” to the screen. Then, in a separate command, output the image in pdf format
# (hint: fig.savefig(’bunny.pdf’)). This might be the night sky if the universe was totally empty.

import numpy as np
import matplotlib.pyplot as plt

img = np.zeros((1024, 1024), dtype=float)
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
fig, ax = plt.subplots()
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html
ax.imshow(img, cmap="gray", interpolation="nearest") #using 'nearest' universally so pixels don't smudge. not sure why we're saving these as PDFs because the interpolation doesn't seem to affect the saved file which is odd
ax.set_title("Utter Nothingness")
ax.axis("off")

print("\n14.3 Imaging: Increasingly real stars\nAttempt 1: Utter Nothingness -\nThe file will be saved as utter_nothingness.pdf and displayed as an image.")
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
fig.savefig("utter_nothingness.pdf", bbox_inches="tight") # using 'tight' universally

plt.show()

# Output:
# 14.3 Imaging: Increasingly real stars
# Attempt 1: Utter Nothingness -
# The file will be saved as utter_nothingness.pdf and displayed as an image.

'''
2. Attempt 2 - Realistic Nothingness
'''
# Generate a two dimensional, 1024 × 1024 element array, in which each element contains a normally
# distributed random number between 0.0 and 30.0. Display this ”image” to the screen, and
# output it in pdf format. This image should look like noise (which is what we made the array to be). 
# It might be an image of a completely empty night sky, but takem with a more realistic detector.

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

# mean 15, std 5
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.normal.html
noise = rng.normal(loc=15.0, scale=5.0, size=(1024, 1024))
# https://numpy.org/doc/2.3/reference/generated/numpy.clip.html
# clipping to keep only values 0-30
noise = np.clip(noise, 0.0, 30.0)

fig, ax = plt.subplots()
ax.imshow(noise, cmap="gray", interpolation="nearest")
ax.set_title("Realistic Nothingness (with normally distributed noise, 0–30)")
ax.axis("off")

print("\nAttempt 2: Realistic nothingness -\nThe file will be saved as realistic_nothingness.pdf and displayed as an image.")
fig.savefig("realistic_nothingness.pdf", bbox_inches="tight")

plt.show()

# Output:
# Attempt 2: Realistic nothingness -
# The file will be saved as realistic_nothingness.pdf and displayed as an image.

'''
3. Attempt 2 - Part 2
'''
# A huge factor in how an image looks is how the values in the array translate to the colormap. 
# With the defaults, the min and max of the array are set to the min and max of the colormap. 
# Let’s change that. Display the ”noise image” from just above so that 0.0 is black and 255.0 
# (rather than 50.0) is white. Since the max value in the array is 30, this should make the brightest 
# displayed color a dark grey. Make a pdf of the result. The image should now look like ”dark noise”.

fig, ax = plt.subplots()
ax.imshow(noise, cmap="gray", vmin=0.0, vmax=255.0, interpolation="nearest") # 0-255
ax.set_title("Dark Noise (display scaled 0, 255)")
ax.axis("off")

print("\nAttempt 2 Part 2: Dark noise -\nThe file will be saved as dark_noise.pdf and displayed as an image.")
fig.savefig("dark_noise.pdf", bbox_inches="tight")
plt.show()

# Output:

# Attempt 2 Part 2: Dark noise -
# The file will be saved as dark_noise.pdf and displayed as an image.

'''
4. Attempt 3 - Unrealistic Star
'''
# Take the ”noise” image from the previous question. Pick a position at random near the center of the image. 
# Manually set all the elements of the array within a 5 × 5 region centered on that point to be all equal to 255.0. 
# Redisplay this image using a greyscale colormap with black equal to 0.0 and white equal to 255.0. 
# Congratulations on making a star!

# fyi this problem and the rest of this section took me an insanely long time :D
# https://www.youtube.com/watch?v=PbKOrSottRQ "slicing"
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

noise = rng.normal(loc=15.0, scale=5.0, size=(1024, 1024))
noise = np.clip(noise, 0.0, 30.0)

# https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html
height, width = noise.shape # 1024h x 1024w
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html
r_center = rng.integers(height//2 - 20, height//2 + 21) # +/- 20 near center, could be lowered
c_center = rng.integers(width//2 - 20, width//2 + 21) # same ^

# 5x5 block = -2,-1,0,1,2, defining 2 row/column for numpy slicing,
half = 2 # radiuss for 5x5 block
r1, r2 = r_center - half, r_center + half + 1 # +1 to include all rows in 5x5 since the numpy slice exlucdes the stop
c1, c2 = c_center - half, c_center + half + 1 # same ^ for columns

star_addition = noise.copy() # tried doing star = noise first but realised it was changing the original array #cringe
# https://numpy.org/doc/stable/user/basics.indexing.html
star_addition[r1:r2, c1:c2] = 255.0 # star color 255=white

fig, ax = plt.subplots()
ax.imshow(star_addition, cmap="gray", vmin=0.0, vmax=255.0, interpolation="nearest")
ax.set_title(f"Attempt 3: Unrealistic Star ({r_center},{c_center})") # star coords
ax.axis("off")

print("\nAttempt 3: Unrealistic star -\nThe file will be saved as unrealistic_star.pdf and displayed as an image.")
fig.savefig("unrealistic_star.pdf", bbox_inches="tight")
plt.show()

# Output:
# Attempt 3: Unrealistic star -
# The file will be saved as unrealistic_star.pdf and displayed as an image.

'''
5. Attempt 4 - Slightly better star
'''
# The above is not a very realistic looking star, so let’s do better. Revert back to the ”noise” image with no star. and
# pick a random position near the center. Hint: doing the following step as a function, where you send the function the
# image and the appropriatae details of the ”star”, and the function returns a new image with the star inserted, will save you 
# time in the long run. For each element in a 5 × 5 region centered on that point, randomly generate a number between 50 and 150 
# and ADD it to the existing number in that element. So for example if the element contains 18 and you generate 107 for that element, 
# the result should be 125. Display the new image, and congratulate yourself on making a realistic(ish) looking star.

# import numpy as np
# import matplotlib.pyplot as plt

# rng = np.random.default_rng()
# noise = rng.normal(loc=15.0, scale=5.0, size=(1024, 1024))
# noise = np.clip(noise, 0.0, 30.0)

# height, width = noise.shape
# r_center = rng.integers(height//2 - 20, height//2 + 21)
# c_center = rng.integers(width//2 - 20, width//2 + 21)


# half = 2  # since size=5
# r1, r2 = r_center - half, r_center + half + 1
# c1, c2 = c_center - half, c_center + half + 1

# # random value 50-150 (151 excluded)
# patch = rng.integers(50, 151, size=(5, 5))
# star = noise.copy()
# star[r1:r2, c1:c2] = star[r1:r2, c1:c2] + patch # slicing new 5x5 50-150 patch

# np.clip(star, 0.0, 255.0, out=star)

# fig, ax = plt.subplots()
# ax.imshow(star, cmap="gray", vmin=0.0, vmax=255.0, interpolation="nearest")
# ax.set_title(f"Attempt 4 – Slightly Better Star ({r_center},{c_center})")
# ax.axis("off")

# print("\nAttempt 4: Slightly better star -\nThe file will be saved as slightly_better_star.pdf and displayed as an image.")
# fig.savefig("slightly_better_star.pdf", bbox_inches="tight")
# plt.show()

# rewritten using function 
import numpy as np
import matplotlib.pyplot as plt

def insert_star(image, size=5, add_low=50, add_high=150, center_window=20): #
    """
    INPUTS:
    image (array), size (default=5), add_low (default=50), add_high (default=150), center_window (default=20))
    OUTPUTS:
    new_image, (r_center, c_center)
    DESCRIPTION:
    Adds a 5x5 “star” by adding random values in [add_low, add_high] to a patch centered near middle of the image. Returns a copy.
    """
    rng = np.random.default_rng()
    new_image = image.copy()
    height, width = new_image.shape

    # centering, same as before
    r_center = rng.integers(height//2 - center_window, height//2 + center_window + 1)
    c_center = rng.integers(width//2 - center_window, width//2 + center_window + 1)

    # patch slice build, same as before
    half = size // 2
    r1, r2 = r_center - half, r_center + half + 1
    c1, c2 = c_center - half, c_center + half + 1

    # per-pixel additions, [add_low, add_high]
    add_patch = rng.integers(add_low, add_high + 1, size=(size, size))
    # adds new patch to image
    new_image[r1:r2, c1:c2] = new_image[r1:r2, c1:c2] + add_patch

    return new_image, (r_center, c_center)

# base noise image (no star)
rng = np.random.default_rng()
noise = rng.normal(15.0, 5.0, (1024, 1024))
noise = np.clip(noise, 0.0, 30.0)

# adding star
star, (r_center, c_center) = insert_star(noise, size=5, add_low=50, add_high=150, center_window=20)

fig, ax = plt.subplots()
ax.imshow(star, cmap="gray", vmin=0.0, vmax=255.0, interpolation="nearest")
ax.set_title(f"Attempt 4 – Slightly Better Star ({r_center},{c_center})")
ax.axis("off")

print("\nAttempt 4: Slightly better star -\nThe file will be saved as slightly_better_star.pdf and displayed as an image.")
fig.savefig("better_star.pdf", bbox_inches="tight")
plt.show()

# Output:
# Attempt 4: Slightly better star -
# The file will be saved as slightly_better_star.pdf and displayed as an image.

'''
6. Attempt 5 - A field of stars
'''
# In this attempt we will add multiple stars of different sizes to the image and make the bigger ones more likely to be brighter.
# We will do this by automatically generating them. Use numpy to randomly pick twenty positions (so, x, y coordinate) in the ”image”, 
# each at least 5 pixels from the edge of the array. Let’s call these positions α. For each α, generate a random ODD number between 3 and 15.
# Let’s call these numbers β. Centered on each α, consider a β × β region. For each eleent in this region, generate a random number 
# between 10 and 60, nultiply it by β/2 (being sure to use the appropriate β for that α) and add the result to the
# number already in that element. Display the final ”image”, using a greyscale colormap with black equal to 0.0 and white equal to 255.0. 
# Congratulations on making a whole field of stars!

# +3 hours
import numpy as np
import matplotlib.pyplot as plt

def insert_star_field(image, n_stars=20, edge=5):
    """
    INPUTS:
        image (array), n_stars (default=20), edge (default=5)
    OUTPUTS:
        new_image
    DESCRIPTION:
        Adds n_stars stars. Each star uses an odd β in 3,5,7,9,11,13,15; a β×β patch gets + (uniform random[10,60] * β/2) per pixel.
    """
    rng = np.random.default_rng()
    new_image = image.copy()
    height, width = new_image.shape

    for _ in range(n_stars): # '_' since function isn't using the loop variable
        r_center = rng.integers(edge, height - edge) # no closer than 5 pixels from border (edge default)
        c_center = rng.integers(edge, width  - edge) # same as above
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html
        beta = rng.choice([3, 5, 7, 9, 11, 13, 15])

        half = beta // 2
        # https://numpy.org/doc/stable/reference/generated/numpy.max.html
        m = max(edge, half) # choosing the larger of edge and half to guarantee the whole patch stays inside the image
        r_center = rng.integers(m, height - m)
        c_center = rng.integers(m, width  - m)

        r1, r2 = r_center - half, r_center + half + 1
        c1, c2 = c_center - half, c_center + half + 1

        add_patch = rng.integers(10, 61, size=(beta, beta)) * (beta / 2.0) # 10-60 random int (61 excluded), size= array of betaxbeta (patch): each pixel gets a random 10-60value, multiplying by beta/2 because the problem says to
        new_image[r1:r2, c1:c2] += add_patch # both floats

    return new_image

# base noise image (no star)
rng = np.random.default_rng()
noise = rng.normal(15.0, 5.0, (1024, 1024))
noise = np.clip(noise, 0.0, 30.0)

# adding a field of stars
stars = insert_star_field(noise, n_stars=20, edge=5)

fig, ax = plt.subplots()
ax.imshow(stars, cmap="gray", vmin=0.0, vmax=255.0, interpolation="nearest")
ax.set_title("Attempt 5 – Field of Stars")
ax.axis("off")

print("\nAttempt 5: Field of Stars -\nThe file will be saved as field_of_stars.pdf and displayed as an image.")
fig.savefig("field_of_stars.pdf", bbox_inches="tight")
plt.show()

# Output:
# Attempt 5: Field of Stars -
# The file will be saved as field_of_stars.pdf and displayed as an image.

'''
7. Attempt 6 - A realistic starfield
'''
# This attempt is very similar to the last, except this time we will replace the function that makes the stars with one that
# improves their realism a bit further. Write a function that takes as inputs two numbers, say γ and ε. The function
# should check if γ is odd and make it odd if it isn’t. The function should then return a γ × γ numpy array containing a 
# two-dimensional gaussianm with the peak at the venter, a FWHM of two pixels, and a peak value equal to ε. 
# Use this function, instead of the approach used in the previous question, to add stars to your image. 
# Make a pdf of your starfield, with appropriate choice of colormap and scaling.

import numpy as np
import matplotlib.pyplot as plt

def make_star_template(gamma, epsilon):
    """
    INPUTS:
        gamma (integer), epsilon (float)
    OUTPUTS:
        template (array)
    DESCRIPTION:
        Returns a gammaxgamma centered 2D Gaussian (FWHM=2) with peak=epsilon. If gamma is even, it is incremented to the next odd.
    """
    if gamma % 2 == 0: # making gamma odd
        gamma += 1
    half = gamma // 2
    # https://www.youtube.com/watch?v=gdeV4UeljUY
    # https://numpy.org/doc/stable/reference/generated/numpy.mgrid.html
    # need +1 to make half inclusive like before
    yy, xx = np.mgrid[-half:half+1, -half:half+1] # coord grid (center= 0,0). [-half:half+1] gives gamma points
    sigma = 2.0 / (2.0 * np.sqrt(2.0 * np.log(2.0))) # from FWHM=2. np.log gives natural log. https://statproofbook.github.io/P/norm-fwhm.html # basic but refresher
    # https://numpy.org/doc/stable/reference/generated/numpy.exp.html
    template = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2)) #2d gaussian #https://en.wikipedia.org/wiki/Gaussian_function
    template *= (epsilon / template[half, half]) #center pixel = epsilon, template[half,half] = peak
    
    return template

def insert_star_field(image, n_stars=20, edge=7):
    """
    INPUTS:
        image (array), n_stars (default=20), edge (default=7)
    OUTPUTS:
        new_image
    DESCRIPTION:
        Adds n_stars Gaussian stars. Each star uses odd β in {3,5,7,9,11,13,15}, center >= edge from border, peak ε = [10,60]*(β/2).
    """
    rng = np.random.default_rng()
    new_image = image.copy()
    height, width = new_image.shape

    for _ in range(n_stars):
        beta = int(rng.choice([3, 5, 7, 9, 11, 13, 15])) # odd beta
        half = beta // 2 # radius for slicing
        r_center = rng.integers(edge, height - edge) # center >= edge from borders so full betaxbeta fits
        c_center = rng.integers(edge, width  - edge) # same ^
        epsilon = rng.integers(10, 61) * (beta / 2.0) # basically same as before. epsilon = peak brightness with same beta/2 multiplication
        template = make_star_template(beta, epsilon) # gaussian stamp with beta size and epislon peak
        r1, r2 = r_center - half, r_center + half + 1 # slice boundaries same as before
        c1, c2 = c_center - half, c_center + half + 1 # same ^
        new_image[r1:r2, c1:c2] += template # same as before

    return new_image

# base noise image
rng = np.random.default_rng()
noise = rng.normal(15.0, 5.0, (1024, 1024))
noise = np.clip(noise, 0.0, 30.0)

# adding starfield
stars = insert_star_field(noise, n_stars=20, edge=7)

fig, ax = plt.subplots()
ax.imshow(stars, cmap="gray", vmin=0.0, vmax=255.0)
ax.set_title("Attempt 6 – Realistic Gaussian Starfield")
ax.axis("off")

print("\nAttempt 6: Realistic starfield -\nThe file will be saved as realistic_starfield.pdf and displayed as an image.")
fig.savefig("realistic_starfield.pdf", bbox_inches="tight")
plt.show()

# Output:
# Attempt 6: Realistic starfield -
# The file will be saved as realistic_starfield.pdf and displayed as an image.

'''INTERMEDIARY'''
print("\nThe following sections prints very long lists of whatever.\nPrepare yourself.")

yes = input("Enter 'yes' to continue: ")

if yes == "yes":
    print("Continuing.")
else:
    print("Continuing anyways!")

# 14.4 File I/O Exercises
'''
1. Test1.txt
'''
# Read the three columns of Test1.txt into three numpy arrays

import numpy as np
# https://numpy.org/doc/2.3/reference/generated/numpy.loadtxt.html
column1_test1, column2_test1, column3_test1 = np.loadtxt("Test1.txt", unpack=True)
print("\n14.4 File I/O Exercises\nProblem 1: Test1.txt")
print("Test1 – column 1:\n", column1_test1)
print("Test1 – column 2:\n", column2_test1)
print("Test1 – column 3:\n", column3_test1)

# Output:
# 14.4 File I/O Exercises
# Problem 1: Test1.txt
# Test1 – column 1:
#  [ 4.  5.  6.  7.  8.  9. 10.]
# Test1 – column 2:
#  [0.180901 0.237141 0.303833 0.3164   0.320766 0.323301 0.327548]
# Test1 – column 3:
#  [3.5042 3.5122 3.5208 3.5223 3.5228 3.5231 3.5234]

'''
2. Test2.txt
'''
# Read all the columns of Test2.txt into individual numpy arrays. Make sure to skip the first two rows.

import numpy as np

columns_test2 = np.loadtxt("Test2.txt", skiprows=2, unpack=True)
print("\nProblem 2: Test2.txt")
print("Test2.txt columns:\n", columns_test2)

# Output:
# Problem 2: Test2.txt
# Test2.txt columns:
#  [[ 4.000000e+00  5.000000e+00  6.000000e+00 ...  2.720000e+02
#    2.730000e+02  2.740000e+02]
#  [ 1.809010e-01  2.371410e-01  3.038330e-01 ...  2.100189e+00
#    2.100299e+00  2.100463e+00]
#  [ 3.504200e+00  3.512200e+00  3.520800e+00 ...  3.639400e+00
#    3.638600e+00  3.637500e+00]
#  ...
#  [ 1.036520e+01  9.848300e+00  9.371900e+00 ... -1.329600e+00
#   -1.355400e+00 -1.392200e+00]
#  [ 1.008440e+01  9.571300e+00  9.102900e+00 ... -1.409700e+00
#   -1.436400e+00 -1.474400e+00]
#  [ 1.186570e+01  1.130770e+01  1.078550e+01 ... -7.391000e-01
#   -7.615000e-01 -7.930000e-01]]

'''
3. Test 3.txt
'''
# Read all the columns of Test3.txt into individual numpy arrays. Make sure to skip the header and footer rows.

import numpy as np

columns_test3 = np.loadtxt("Test3.txt", dtype=str, skiprows=34, max_rows=42, unpack=True)
print("\nProblem 3: Test3.txt")
print("Test3.txt columns:\n", columns_test3)

# Output:
# Problem 3: Test3.txt
# Test3.txt columns:
#  [['IRAS00188-0856' 'IRAS00397-1312' 'IRAS01003-2238' 'IRAS03158+4227'
#   'IRAS03521+0028' 'IRAS05189-2524' 'IRAS06035-7102' 'IRAS06206-6315'
#   'IRAS07598+6508' 'IRAS08311-2459' 'IRAS08572+3915' 'IRAS09022-3615'
#   'IRAS10378+1109' 'IRAS10565+2448' 'IRAS11095-0238' 'IRAS12071-0444'
#   'IRAS13120-5453' 'IRAS13451+1232' 'IRAS14348-1447' 'IRAS14378-3651'
#   'IRAS15250+3609' 'IRAS15462-0450' 'IRAS16090-0139' 'IRAS17208-0014'
#   'IRAS19254-7245' 'IRAS19297-0406' 'IRAS20087-0308' 'IRAS20100-4156'
#   'IRAS20414-1651' 'IRAS20551-4250' 'IRAS22491-1808' 'IRAS23128-5919'
#   'IRAS23230-6926' 'IRAS23253-5415' 'IRAS23365+3604' 'IRAS09320+6134'
#   'IRAS12540+5708' 'IRAS13428+5608' 'IRAS13536+1836' 'IRAS15327+2340'
#   'IRAS16504+0228' 'IRAS01572+0009']
#  ['00188' '00397' '01003' '03158' '03521' '05189' '06035' '06206' '07598'
#   '08311' '08572' '09022' '10378' '10565' '11095' '12071' '13120'
#   '4C12.50' '14348' '14378' '15250' '15462' '16090' '17208' '19254'
#   '19297' '20087' '20100' '20414' '20551' '22491' '23128' '23230' '23253'
#   '23365' 'UGC5101' 'Mrk231' 'Mrk273' 'Mrk463' 'Arp220' 'NGC6240'
#   'Mrk1014']
#  ['5.360507' '10.564704' '15.708365' '49.801667' '58.675800' '80.255800'
#   '90.725042' '95.255042' '121.137833' '128.335833' '135.105792'
#   '136.052961' '160.121539' '164.825493' '168.014071' '182.438049'
#   '198.776494' '206.889007' '219.410000' '220.245867' '231.747517'
#   '237.236721' '242.918469' '260.841481' '292.839167' '293.088544'
#   '302.849458' '303.373083' '311.075888' '314.611589' '342.955267'
#   '348.944790' '351.515083' '352.025417' '354.755305' '143.964981'
#   '194.059308' '206.175463' '209.011963' '233.738563' '253.245295'
#   '29.959214']
#  ['-8.657217' '-12.934122' '-22.365895' '42.641111' '0.617611'
#   '-25.362600' '-71.052833' '-63.289861' '64.996833' '-25.159361'
#   '39.065111' '-36.450274' '10.888415' '24.542905' '-2.906219'
#   '-5.020490' '-55.156452' '12.290067' '-15.005556' '-37.075538'
#   '35.977092' '-4.992669' '-1.785156' '-0.283594' '-72.655000'
#   '-3.998962' '-2.997417' '-41.793028' '-16.671172' '-42.650056'
#   '-17.873183' '-59.054320' '-69.171889' '-53.975278' '36.352308'
#   '61.353182' '56.873677' '55.886847' '18.372078' '23.503139' '2.400926'
#   '0.394615']
#  ['3' '5' '5' '2' '3' '5' '2' '2' '4' '4' '3' '4' '5' '2' '3' '3' '4' '3'
#   '3' '5' '4' '4' '4' '4' '3' '4' '5' '2' '5' '4' '3' '3' '4' '5' '5' '4'
#   '5' '3' '3' '4' '4' '4']
#  ['5' '5' '5' '2' '3' '5' '2' '2' '4' '4' '3' '3' '5' '2' '3' '3' '5' '3'
#   '3' '5' '4' '4' '4' '4' '3' '3' '4' '2' '5' '4' '3' '3' '3' '5' '5' '4'
#   '4' '4' '3' '4' '4' '4']
#  ['2' '2' '3' '4' '5' '1' '6' '7' '2' '8' '1' '9' '5' '1' '2' '2' '10'
#   '3' '1' '10' '1' '2' '5' '10' '11' '7' '7' '7' '7' '10' '1' '7' '7' '6'
#   '10' '1' '1' '1' '3' '1' '10' '3']
#  ['0.00' '0.00' '0.00' '42.8' '3.88' '0.19' '10.4' '4.80' '0.00' '0.00'
#   '6.62' '4.03' '0.00' '24.75' '1.29' '2.80' '0.00' '3.20' '5.47' '0.00'
#   '1.27' '0.00' '0.00' '0.00' '10.2' '1.30' '0.00' '7.40' '0.00' '0.00'
#   '2.68' '3.94' '1.17' '0.00' '0.00' '0.40' '0.64' '0.77' '4.40' '0.72'
#   '0.89' '0.00']
#  ['0.12842' '0.26172' '0.11783' '0.13443' '0.15191' '0.04256' '0.07946'
#   '0.09244' '0.14830' '0.10045' '0.05835' '0.05964' '0.13627' '0.04310'
#   '0.10663' '0.12835' '0.03076' '0.12174' '0.08273' '0.06764' '0.05521'
#   '0.09979' '0.13358' '0.04281' '0.06171' '0.08573' '0.10567' '0.12958'
#   '0.08687' '0.04300' '0.07776' '0.04464' '0.10659' '0.12978' '0.06441'
#   '0.03937' '0.04217' '0.03654' '0.04924' '0.01840' '0.02448' '0.16311']
#  ['2' '1' '1' '3' '2' '3' '1' '3' '4' '4' '3' '1' '2' '1' '2' '3' '3' '3'
#   '2' '3' '1' '4' '2' '1' '3' '1' '2' '1' '1' '1' '1' '1' '2' '2' '2' '2'
#   '4' '3' '3' '3' '2' '4']
#  ['-999' '1.05e7' '2.54e7' '-999' '-999' '2.95e7' '2.04e7' '-999'
#   '1.02e8' '-999' '-999' '-999' '1.02e7' '2.04e7' '3.92e7' '3.50e7'
#   '-999' '6.54e7' '4.25e7' '4.60e7' '4.25e7' '6.86e7' '-999' '2.33e8'
#   '7.89e7' '-999' '1.94e8' '-999' '1.03e8' '3.22e7' '3.81e7' '4.36e7'
#   '3.50e7' '-999' '3.71e7' '5.50e8' '1.73e7' '5.61e8' '5.50e7' '6.08e7'
#   '2.33e8' '1.35e8']
#  ['-999' '2.63e6' '6.35e6' '-999' '-999' '7.38e6' '5.10e6' '-999'
#   '4.88e7' '-999' '-999' '-999' '2.55e6' '5.10e6' '7.38e6' '8.75e6'
#   '-999' '1.64e7' '1.76e7' '1.15e7' '1.06e7' '1.72e7' '-999' '5.83e7'
#   '1.97e7' '-999' '4.85e7' '-999' '2.58e7' '8.05e6' '9.53e6' '1.09e7'
#   '8.75e6' '-999' '9.28e6' '2.75e8' '4.33e6' '1.40e8' '1.85e7' '1.52e7'
#   '5.83e7' '3.38e7']
#  ['-999' '2' '2' '-999' '-999' '2' '1' '-999' '4' '-999' '-999' '-999'
#   '5' '1' '1' '1' '-999' '1' '2' '2' '2' '2' '-999' '2' '1' '-999' '2'
#   '-999' '2' '2' '1' '1' '2' '-999' '2' '3' '2' '2' '6' '2' '2' '2']
#  ['266.0e6' '315.0e6' '76.1e6' '-999' '289.0e6' '208.0e6' '-999' '-999'
#   '390.0e6' '-999' '90.0e6' '-999' '197.0e6' '-999' '-999' '-999' '-999'
#   '-999' '-999' '-999' '-999' '114.0e6' '370.0e6' '-999' '-999' '-999'
#   '-999' '-999' '107.0e6' '-999' '-999' '-999' '-999' '-999' '-999'
#   '-999' '378.0e6' '199.0e6' '-999' '151.0e6' '-999' '1258.0e6']
#  ['-999' '10.5e6' '25.4e6' '-999' '-999' '29.5e6' '20.4e6' '-999' '-999'
#   '-999' '-999' '-999' '-999' '20.4e6' '39.2e6' '35.0e6' '-999' '65.4e6'
#   '42.5e6' '46.0e6' '42.5e6' '68.6e6' '-999' '233.0e6' '78.9e6' '-999'
#   '194.0e6' '-999' '103.0e6' '32.2e6' '38.1e6' '43.6e6' '35.0e6' '-999'
#   '37.1e6' '5.50e8' '17.3e6' '561.0e6' '5.50e7' '60.8e6' '233.0e6'
#   '135.0e6']
#  ['-999' '2.63e6' '6.35e6' '-999' '-999' '7.38e6' '5.10e6' '-999' '-999'
#   '-999' '-999' '-999' '-999' '5.10e6' '7.38e6' '8.75e6' '-999' '1.64e7'
#   '1.76e7' '1.15e7' '1.06e7' '1.72e7' '-999' '5.83e7' '1.97e7' '-999'
#   '4.85e7' '-999' '2.58e7' '8.05e6' '9.53e6' '1.09e7' '8.75e6' '-999'
#   '9.28e6' '2.75e8' '4.33e6' '1.40e8' '1.85e7' '1.52e7' '5.83e7' '3.38e7']
#  ['-999' '-999' '-999' '8.70' '-999' '8.87' '-999' '-999' '-999' '-999'
#   '8.61' '-999' '-999' '8.37' '-999' '-999' '8.55' '7.72' '8.62' '8.07'
#   '-999' '-999' '-999' '7.66' '-999' '-999' '-999' '9.31' '-999' '8.00'
#   '8.73' '-999' '-999' '-999' '8.17' '-999' '8.47' '8.24' '-999' '7.93'
#   '8.61' '8.39']
#  ['-999' '-999' '-999' '1000' '-999' '491' '-999' '-999' '-999' '-999'
#   '800' '-999' '-999' '450' '-999' '-999' '549' '640' '450' '425' '-999'
#   '-999' '-999' '600' '-999' '-999' '-999' '456' '-999' '450' '241'
#   '-999' '-999' '-999' '450' '-999' '700' '620' '-999' '800' '400' '268']
#  ['-999' '-999' '-999' '350' '-999' '219' '-999' '-999' '-999' '-999'
#   '403' '-999' '-999' '100' '-999' '-999' '1115' '227' '420' '180' '-999'
#   '-999' '-999' '176' '-999' '-999' '-999' '672' '-999' '200' '654'
#   '-999' '-999' '-999' '57' '-999' '350' '200' '-999' '117' '267' '93']]

# 14.5 FITS File Exercises
'''
1. Loading BBand.fits
'''
# 1. Load the Bband.fits file from Data/images into python. What sort of data does this file contain? How many separate arrays and headers?
# https://docs.astropy.org/en/stable/io/fits/

from astropy.io import fits

#load fits file & access data array and its header
hdulist = fits.open("Data/Images/ImagesForAnalysis/Bband.fits")
data = hdulist[0].data
header = hdulist[0].header

print("This file has", len(hdulist), "HDU.")
print("Data shape:", data.shape)
print("Data type:", data.dtype)

# THere is 1 HDU, which means there is ONE image array and ONE header describing that array.

# Output:
# This file has 1 HDU.
# Data shape: (2009, 2009)
# Data type: >f4

'''
2. Print header information
'''
# 2. Print the header information to the screen (hint: hdulist.info). What does it tell you? Summarize the salient information from the header.

from astropy.io import fits

hdulist = fits.open("Data/Images/ImagesForAnalysis/Bband.fits")
hdulist.info()

# view entire header content 
#print(hdulist[0].header)

# This FITS file contains one HDU, which is the primary image. The image data are stored as 32-bit
# floating point numbers and have dimesions of 2009x2009 pixels. The header has 914 entries 
# describing details abt the observation and instrument.

# Output:
# Filename: Data/Images/ImagesForAnalysis/Bband.fits
# No.    Name      Ver    Type      Cards   Dimensions   Format
#   0  PRIMARY       1 PrimaryHDU     914   (2009, 2009)   float32  

'''
3. Displaying BBand.fits
'''
# 3. You should have discovered that the file contains an image. Display the image, using default parameters. What do you see?
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html

import matplotlib.pyplot as plt

plt.imshow(data)
plt.show()

# I see several bright point sources (stars lol) that are scattered across a dark background. Some of the rly faint stars
# are barely visible bc the default color scaling is dominated by the brightest pixels.

'''
4. Statistics on the image array
'''
# 4. Calculate the mean, median, and standard deviation of elements in array. Calculate the vase 10 logarithm of the elements in the array. Make a histogram of the (original) array values and output this histogram as a pdf.

import numpy as np
import matplotlib.pyplot as plt

# use numpy functions to calcualte image stats
print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Std:", np.std(data))

# calculate base10 og of the data (offset by +1 to avoid log(0))
log_data = np.log10(data + 1)
print("Log10 of data:", log_data.flatten()[:5])

# make a histogram of original image values
plt.hist(data.flatten(), bins=100, color='purple', edgecolor='black')
plt.yscale('log')
plt.xlabel("Pixel value")
plt.ylabel("Count (log scale)")
plt.title("Histo of Bband Image Pixel Values")

# save histogram as PDF
plt.savefig("bband_hisogram.pdf")
plt.show

# Output:
# Mean: 368.32596
# Median: 352.17007
# Std: 669.2939
# Log10 of data: [2.5350478 2.5102117 2.4934535 2.4984717 2.5262303]

'''
5. Estimating min/max values
'''
# 5. Use the histogram to estimate the min and max values to translate the elements in the array to a colormap, and redisplay the image with these values. What do you see?
# https://numpy.org/devdocs/reference/generated/numpy.percentile.html
# https://docs.astropy.org/en/stable/visualization/normalization.html

import numpy as np
import matplotlib.pyplot as plt

vmin = np.percentile(data,5) #near background peak (dark noise)
vmax = np.percentile(data, 99.5) #ignore the brightest outlieers

print("Histogram-based vmin/vmax:", vmin, vmax)

#display w/ high contrast perceptual colormap
plt.imshow(data, vmin=vmin, vmax=vmax, cmap='magma')
plt.colorbar(label="pixel value")
plt.title("B-band image with histogram-chosen scaling")
plt.show()

# I see a star-field.

# Output:
# Histogram-based vmin/vmax: 330.5201416015625 560.6754760742189

'''
6. Scaling/stretching
'''
# 6. Experiment with different python commands to achieve a pretty scaling of the image. What astronomical object is this an image of?
#https://stackoverflow.com/questions/49538185/purpose-of-numpy-log1p
# https://numpy.org/doc/2.3/reference/generated/numpy.log1p.html

import numpy as np
import matplotlib.pyplot as plt

#shift so the minimum pixel becomes 0 (avoids negatives before non linear stretching)
img0 = data - np.min(data)

# log stretch (log1p to avoid log(0))
disp = np.log1p(img0)

vmin = np.percentile(disp, 5)
vmax = np.percentile(disp, 99.5)

plt.imshow(disp, vmin=vmin, vmax=vmax, cmap='magma')
plt.colorbar(label='stretched pixel value')
plt.title('B-band image, log stretch, percentile scaling')
plt.show()

# This is a star field (jk, now I know its the pillars of creation)

'''
7. Load/display Halpha
'''
# 7. From the same directory, load the Halpha image into python as a separate image. Display this image with an appropriate colormap scaling. Then, divide the B band image by the Halpha image to make a new image. Display this new image with an appropriate scaling.

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

#load both images
bband = fits.getdata("Data/Images/ImagesForAnalysis/Bband.fits")
halpha = fits.getdata("Data/Images/ImagesForAnalysis/Halpha.fits")

#display halpha with the usual percentile scaling
vmin_h = np.percentile(halpha, 5)
vmax_h = np.percentile(halpha, 99.5)

plt.imshow(halpha, vmin=vmin_h, vmax=vmax_h, cmap='magma')
plt.colorbar(label='Pixel value (Halpha)')
plt.title('Halpha image w/ percentile scaling')
plt.show()

###

#create ratio image (add 1e-6 to prevent division by 0)
ratio = bband / (halpha + 1e-6)

#percentile scaling for ratio image
vmin_r = np.percentile(ratio, 5)
vmax_r = np.percentile(ratio, 99.5)

plt.imshow(ratio, vmin=vmin_r, vmax=vmax_r, cmap='magma')
plt.colorbar(label='B / Hα ratio')
plt.title('B-band divided by Hα image')
plt.show()

'''
8. The Final FITS
'''
# 8. For this final question you will need to work with at least one other person.
# https://docs.astropy.org/en/stable/io/fits/usage/headers.html
# https://docs.astropy.org/en/latest/io/fits/index.html
# https://docs.python.org/3/library/random.html
# https://numpy.org/doc/2.1/reference/random/generator.html
# https://numpy.org/devdocs/reference/generated/numpy.zeros_like.html

import numpy as np
from astropy.io import fits

#load images as numpy arrays
b_band = fits.getdata("Data/Images/ImagesForAnalysis/Bband.fits")
h_alpha = fits.getdata("Data/Images/ImagesForAnalysis/Halpha.fits")

#take image data to make it float so we can actually do math with it
b_img = b_band.astype(float)
h_img = h_alpha.astype(float)

#pick random float in [2.7, 27.9] using random number generator
rng = np.random.default_rng()
alpha = rng.uniform(2.1, 27.9)

# scale halpha and make BovH (avoid dividing by 0)
eps = 1e-8
BovH = b_img / (alpha * h_img + eps)

print("alpha =", alpha, " and BovH shape:", BovH.shape)

#####

# pick random beta in [1.057, 1.553]
beta = rng.uniform(1.057, 1.553)

#uncertainity image
BovHerr = beta * np.sqrt(np.clip(BovH, 0, None))

print("beta =", beta, " and BovHerr shape:", BovHerr.shape)

#####

# create array of 0s that has same shape as BovH, uses unsigned 8bit ints as data type
BovHmask = np.zeros_like(BovH, dtype=np.uint8)

h, w = BovHmask.shape                              # get num of rows & cols of mask
flat_idx = rng.choice(h*w, size=7, replace=False)  #randomly pick 7 unique flat pixel indices
rows, cols = np.unravel_index(flat_idx, (h, w))    # convert flat indices to (row,col) coords
BovHmask[rows, cols] = 1                           # set those 7 random pixel locations to 1 in the mask

print("Mask ones at (row, col):", list(zip(rows, cols)))

#####

#store alpha, beta, author to header data
primary_hdu = fits.PrimaryHDU()
primary_hdu.header["ALPHA"] = (float(alpha), "Scale factor for Halpha")
primary_hdu.header["BETA"] = (float(beta), "Scale factor for Bband")
primary_hdu.header["AUTHOR"] = "Silvia Arjona Garcia"

#create image HDUs for all bovh, bovherr, bovhmask
hdu_bovh = fits.ImageHDU(BovH, name="BOVH")
hdu_bovherr = fits.ImageHDU(BovHerr, name="BOVHERR")
hdu_mask = fits.ImageHDU(BovHmask, name="BOVHMASK")

hdul_out = fits.HDUList([primary_hdu, hdu_bovh, hdu_bovherr, hdu_mask]) # combine all HDUs into a single FITS structure
hdul_out.writeto("BovH_package.fits", overwrite=True)

print("Wrote BovH_package.fits with extensions: PRIMARY, BOVH, BOVHERR, BOVHMASK")

# Output:
# alpha = 6.984394895994777  and BovH shape: (2009, 2009)
# beta = 1.4139290777847564  and BovHerr shape: (2009, 2009)
# Mask ones at (row, col): [(1750, 818), (1885, 1092), (222, 847), (102, 1), (948, 1691), (592, 744), (1866, 454)]
# Wrote BovH_package.fits with extensions: PRIMARY, BOVH, BOVHERR, BOVHMASK

'''
8. Part 2
'''
# 8. continued
# https://docs.astropy.org/en/stable/io/fits/index.html
# https://stackoverflow.com/questions/19482970/get-a-list-from-pandas-dataframe-column-headers

from astropy.io import fits
import numpy as np

jyxzel_file = "FITSAstr300L-3.fits"
hdul = fits.open(jyxzel_file)

#read metadata
hdr = hdul[0].header                 #access primary header of FITS file
alpha_jyx = hdr.get("ALPHA", None)   # extract alpha and beta vals
beta_jyx = hdr.get("BETA", None)

print("From Jyxzel's file:")
print("  ALPHA :", alpha_jyx)
print("  BETA  :", beta_jyx)

mask = hdul["BOVHMASK"].data.astype(int)          # read mask and convert to int
rows, cols = np.where(mask == 1)                  # find row, col where mask = 1
coords = list(zip(rows.tolist(), cols.tolist()))  # pair row & col indices into tuples
print("  Mask 1-locations (row, col):", coords)   # print coords of pixels where mask = 1

# Output:
# From Jyxzel's file:
#   ALPHA : 27
#   BETA  : 1.155
#   Mask 1-locations (row, col): [(197, 831), (537, 1295), (554, 453), (1001, 541), (1307, 246), (1598, 1305), (1900, 1142)]

# 14.6 Science with Astronomical Images
'''
1. Loading the image
'''
# 1.  Load the image TheStar/coj1m011-kb05-20140607-0113-e90.fits into python.

from astropy.io import fits

#load the FITS file
img_data = fits.open("Data/StellarPhotometry/coj1m011-kb05-20140607-0113-e90.fits")

#print header info
header_info = img_data[0].header
print(header_info)

# This FITS image is an LCOGT 1-m exposure of J1614-1906. It was observed on 2014-06-07 from 14:51:40.337 to 14:52:01.773 UTC.

# Output:
# very very long single line

'''
2. Copying data and header into variable
'''
# 2. Copy the data and header into their own variable 

# I basically did half of this in question 1.

header_info = img_data[0].header
img_array = img_data[0].data

'''
3. Displaying and scaling the image
'''
# 3. Produce a display image of the data, adjusting the image scaling appropriately

import matplotlib.pyplot as plt
import  numpy as np

img_array = img_data[0].data

# percentile scaling
vmin = np.percentile(img_array, 5)
vmax = np.percentile(img_array, 99.5)

plt.imshow(img_array, cmap='magma', origin='lower', vmax=vmax, vmin=vmin)
plt.colorbar(label='Pixel value')
plt.title('J1614-1906 (2014-06-07) Scaled Display')
plt.show()

'''
4. Identifying the star
'''
# 4. Identify the star in the image 
# https://docs.astropy.org/en/latest/wcs/index.html
#https://docs.astropy.org/en/stable/api/astropy.wcs.utils.skycoord_to_pixel.html

from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np

# build WCS from the header
w = WCS(header_info)

# coordinates of J1614-190617
target = SkyCoord('16h14m20.3s', '-19d06m48.1s', frame='icrs')

# convert sky position to pixel coords using skycoord
x, y = skycoord_to_pixel(target, w, origin=0)
print(f"J1614-190617 at pixel coordinates ({x:.1f}, {y:.1f})")

# show image with marker
plt.imshow(img_array, origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
plt.plot(x, y, 'o', mfc='none', ms=14, mew=2, label='J1614-190617') #plot open circle at target pixel
plt.title("J1614-190617")
plt.show()

# Output:
# WARNING: FITSFixedWarning: 'obsfix' made the change 'Set OBSGEO-L to   149.070877 from OBSGEO-[XYZ].
# Set OBSGEO-B to   -31.272798 from OBSGEO-[XYZ].
# Set OBSGEO-H to     1161.994 from OBSGEO-[XYZ]'. [astropy.wcs.wcs]
# J1614-190617 at pixel coordinates (967.1, 1047.9)

'''
5. Selecting a star-less region
'''
# 5. From the whole image, select a region that has no stars to estimate this background signal.
# https://stackoverflow.com/questions/50067977/plt-add-patch-causes-an-error-how-do-i-add-a-rectangle-over-a-set-of-points

import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats

y0, y1 = 100, 250
x0, x1 = 900, 1100

# small region with no stars
sky_box = img_array[y0:y1, x0:x1]

sky_mean = float(np.mean(sky_box))
sky_median = float(np.median(sky_box))
sky_std = float(np.std(sky_box))
print(f"Sky mean={sky_mean:.3f}, median={sky_median:.3f}, std={sky_std:.3f}")

# draw sky box on graph to visualize

#prcentage scaling
vmin = np.percentile(img_array, 5)
vmax = np.percentile(img_array, 99.5)

plt.imshow(img_array, origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
plt.gca().add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, lw=2))
plt.title("Sky box")
plt.show()

# Output:
# Sky mean=57.834, median=57.783, std=16.493

'''
6. Defining new minimal region containing the target star
'''
# 6. Define a separate region containing the target star and as little else as possible. Compute the total number of counts in the region.
#https://stackoverflow.com/questions/32271331/can-anybody-explain-me-the-numpy-indices

import numpy as np
import matplotlib.pyplot as plt

#radius of aperture
r = 22

yy, xx = np.indices(img_array.shape)               # creates two 2D arrays representing row & col for every pixel
ap_mask = (xx - x)**2 + (yy - y)**2 <= r**2        # distance from star center; inside circle = true, outside circle = false

n_pix = int(ap_mask.sum())                             #num of pixels in circle
total_counts_region = float(img_array[ap_mask].sum())  #total light from pixels inside circle

print(f"Aperture radius r = {r} px")
print(f"Pixels in region   = {n_pix}")
print(f"TOTAL counts in region (star + sky) = {total_counts_region:.1f}")

# visualize region (used same patch thing as number 5 but for a circle)
plt.imshow(img_array, origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
plt.gca().add_patch(plt.Circle((x, y), r, fill=False, color='teal', lw=2))  # draw circle patch at (x, y)
plt.title("Chosen star region")
plt.show()

# Output:
# Aperture radius r = 22 px
# Pixels in region   = 1517
# TOTAL counts in region (star + sky) = 193747.7

'''
7. Using GAIN to convert counts to electrons
'''
# 7. Using the GAIN header keyword, convert from counts to the number of electrons.

gain = header_info['GAIN']  #reads gain value from header info

e_from_counts = total_counts_region * gain

print(f"GAIN = {gain:.3f} e⁻/ADU")
print(f"Total electrons = {e_from_counts:.1f} e⁻")

# Output:
# GAIN = 1.400 e⁻/ADU
# Total electrons = 271246.7 e⁻

'''
8. Computing per-pixel mean of sky background minus target region
'''
# 8. Compute the mean sky background per pixel from the background region, and them subtract this (in the appropriate way) from the target region.

# background signal in circle region
background_counts = sky_mean * n_pix

# star signal after subtracting background
star_counts = total_counts_region - background_counts

# per-pixel values
total_per_pix = total_counts_region / n_pix
background_per_pix = sky_mean
star_per_pix = star_counts / n_pix

print(f"Total counts (star + sky): {total_counts_region:.1f} ADU")
print(f"Background counts:         {background_counts:.1f} ADU")
print(f"Net star counts:           {star_counts:.1f} ADU\n")

print(f"Per-pixel totals:")
print(f"  Total per pixel:      {total_per_pix:.2f} ADU/pix")
print(f"  Background per pixel: {background_per_pix:.2f} ADU/pix")
print(f"  Net star per pixel:   {star_per_pix:.2f} ADU/pix")

# Output:
# Total counts (star + sky): 193747.7 ADU
# Background counts:         87733.5 ADU
# Net star counts:           106014.2 ADU

# Per-pixel totals:
#   Total per pixel:      127.72 ADU/pix
#   Background per pixel: 57.83 ADU/pix
#   Net star per pixel:   69.88 ADU/pix

'''
9. Uncertainties
'''
# 9. Use these hints to propagate uncertainties appropriately.

# star uncertainty: one measurement -> poisson (sqrt of photons)
star_electrons = star_counts * gain
star_uncertainty = np.sqrt(star_electrons)

# background uncertainty: use empirical standard deviation of sky
sky_std_electrons = np.std(sky_box) * gain  # per pixel
background_uncertainty_per_pixel = sky_std_electrons
background_uncertainty_in_aperture = np.sqrt(n_pix) * sky_std_electrons  # scaled to the aperture

# print with two decimals
print(f"Uncertainty in star photons:            {star_uncertainty:.2f} photons")
print(f"Uncertainty in background (per pixel):  {background_uncertainty_per_pixel:.2f} photons")
print(f"Uncertainty in background (aperture):   {background_uncertainty_in_aperture:.2f} photons")

# Output:
# Uncertainty in star photons:            385.25 photons
# Uncertainty in background (per pixel):  23.09 photons
# Uncertainty in background (aperture):   899.32 photons

'''
10. 10
'''
# 10 

from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np

# build WCS from the header
w = WCS(header_info)

# coordinates of J1614-190617
target = SkyCoord('16h14m20.912s', '-19d06m04.7s', frame='icrs')

# convert sky position to pixel coords using skycoord
xs, ys = skycoord_to_pixel(target, w, origin=0)
print(f"J1614-190617 at pixel coordinates ({x:.1f}, {y:.1f})")

#radius of aperture
r = 22

yys, xxs = np.indices(img_array.shape)               # creates two 2D arrays representing row & col for every pixel
ap_mask_s = (xxs - xs)**2 + (yys - ys)**2 <= r**2        # distance from star center; inside circle = true, outside circle = false

n_pix_s = int(ap_mask.sum())                             #num of pixels in circle
total_counts_region_s = float(img_array[ap_mask_s].sum())  #total light from pixels inside circle

print(f"Aperture radius r = {r} px")
print(f"Pixels in region   = {n_pix}")
print(f"TOTAL counts in region (star + sky) = {total_counts_region:.1f}")

# visualize region (used same patch thing as number 5 but for a circle)
plt.imshow(img_array, origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
plt.gca().add_patch(plt.Circle((x, y), r, fill=False, color='teal', lw=2))  # draw circle patch at (x, y)
plt.title("Chosen star region")
plt.show()

gain = header_info['GAIN']  #reads gain value from header info

e_from_counts = total_counts_region_s * gain

# star signal after subtracting background
star_counts_s = total_counts_region_s - background_counts

# per-pixel values
total_per_pix_s = total_counts_region_s / n_pix
star_per_pix_s = star_counts_s / n_pix

# star uncertainty: one measurement -> poisson (sqrt of photons)
star_electrons_s = star_counts_s * gain
star_uncertainty_s = np.sqrt(star_electrons_s)

###

m_standard = 13.50

flux_ratio = star_counts / star_counts_s
m_target = m_standard - 2.5 * np.log10(flux_ratio)

error_m = (2.5 / np.log(10)) * np.sqrt(
    (star_uncertainty / star_counts) ** 2 + (star_uncertainty_s / star_counts_s) ** 2
)

print(f"Estimated r-band magnitude of target star: {m_target: .3}")
print(f"Estimated r-band magnitude of target star uncertainty: {error_m: .1}")

# Output:
# WARNING: FITSFixedWarning: 'obsfix' made the change 'Set OBSGEO-L to   149.070877 from OBSGEO-[XYZ].
# Set OBSGEO-B to   -31.272798 from OBSGEO-[XYZ].
# Set OBSGEO-H to     1161.994 from OBSGEO-[XYZ]'. [astropy.wcs.wcs]
# J1614-190617 at pixel coordinates (948.1, 955.7)
# Aperture radius r = 22 px
# Pixels in region   = 1517
# TOTAL counts in region (star + sky) = 193747.7

'''
11.
'''
# Hi Mitchell. Unfortunately, it’s 6:45am and I wasn’t able to get a working code going. 
# I tried every possible option and couldn’t get it to work, so I give up. 
# If possible, it would be great to meet at a later date to work through this together. Sorry. 

# Silvia

'''
12.
'''
# See number 11

'''
13.
'''
# To tell if the star’s brightness is actually changing, you can use a statistical test instead of just looking at the plot. 
# A simple way is a chi-squared test, which compares how much the measured magnitudes vary to how much variation you’d expect 
# just from measurement errors. If the real scatter is much larger than what random noise can explain, the star is likely variable. 
# This test is good for spotting slow or irregular changes in brightness but might miss very short-term variations that happen 
# between the times the images were taken.

# 14.7 Capstone Questions:
'''
7. Capstone Questions
'''
# The brightness of the disk of the Sun looks the sane from Earth as it does from
# Venus. But for faraway stars (that is, stars for which you cannot see the disk,
# they are unresolved points), they look fainter the further away they are. Why
# these two contrasting behaviours? What is going on?

print("\n14.7 - Capstone Questions\nProblem 1: Contrasting brightness behaviors")
# The brightness of the disk of the Sun looks the sane from Earth as it does from
# Venus. But for faraway stars (that is, stars for which you cannot see the disk,
# they are unresolved points), they look fainter the further away they are. Why
# these two contrasting behaviours? What is going on?

# https://www.youtube.com/watch?v=HxAMN4mZKNM
# cool demonstration https://www.youtube.com/watch?v=FhTPAuK7LQo
# useful https://www.astro.rug.nl/~ahelmi/teaching/gal2010/ellipt.pdf
# using unicode for symbols because why not
# unicode printing https://stackoverflow.com/questions/8651361/how-do-you-print-superscript
print("The brightness of the disk of the Sun looks the same from Earth as it does from Venus because " \
"as the apparent area of the sun shrinks by 1/d\u00b2, the total flux also shrinks by the same factor " \
"(inverse square of distance). Using F = L/(4\u03C0d\u00b2) and S = F/(\u03C0\u03B8\u00b2), a disk keeps the same surface " \
"brightness S with distance. For a disk of radius R at distance d, \u03B8 \u2248 R/d, so the apparent area scales like " \
"\u03C0 (R/d)\u00b2. Total flux falls as 1/d\u00b2 while S stays constant. Distant stars, however, have disks that are smaller than " \
"the observing instrument (in this case, a human eye) can resolve, so they appear as tiny spots. The flux still drops " \
"as 1/d\u00b2, but the image can't shrink any further so they look fainter with distance."
)

# Output:
# 14.7 - Capstone Questions
# Problem 1: Contrasting brightness behaviors
# The brightness of the disk of the Sun looks the same from Earth as it does from Venus because as the apparent area of the sun shrinks by 1/d², 
# the total flux also shrinks by the same factor (inverse square of distance). Using F = L/(4πd²) and S = F/(πθ²), a disk keeps the same surface 
# brightness S with distance. For a disk of radius R at distance d, θ ≈ R/d, so the apparent area scales like π (R/d)². 
# Total flux falls as 1/d² while S stays constant. Distant stars, however, have disks that are smaller than the observing instrument (in this case, a human eye) 
# can resolve, so they appear as tiny spots. The flux still drops as 1/d², but the image can't shrink any further so they look fainter with distance.

'''
Problem 2: Smartphone vs Telescope camera RGB
'''
# With only a few exceptions, the cameras in smartphones or consumer cameras
# do not form a colour image in the way that a camera on a telescope does. A
# camera on a telescope does it by taking three separate images in ”red”, ”green”,
# and ”blue” filters, which are then combined later. How does a smartphone or
# consumer camera do it? Why would it be a REALLY bad idea for telescope
# cameras to take colour images in the same way (conceptually) as a smartphone camera?

# https://www.youtube.com/watch?v=LWxu4rkZBLw
# https://www.arrow.com/en/research-and-events/articles/introduction-to-bayer-filters
# https://www.sdss3.org/instruments/camera.php
# https://optcorp.com/blogs/astrophotography-101/color-and-monochrome-sensors?srsltid=AfmBOoq2uQyYKIHbIWShsdLhbjp01z39LjSFhbTPz9Dcn4R-jDmBQ1rz
# interesting example of demosaicing https://www.researchgate.net/figure/Effect-of-demosaicing-on-low-light-noise-characteristics-a-RGB-image-with-spatially_fig1_221362689
# more on demosaicing https://photo.stackexchange.com/questions/22596/what-are-the-pros-and-cons-of-different-bayer-demosaicing-algorithms
# there seem to be many different techniques for demosaicing CFA exposures, I can't really tell if there is a 'best'
print("\nProblem 2: Smartphone vs Telescope camera RGB")
print("Smartphones make color images by placing a color filter array over a single sensor. Each pixel measures only " \
"one color, and the two missing colors are recovered through demosaicing, which is a post-shot processing method. " \
"This would be a bad idea on a telescope because a color filter array gets rid of photons at every pixel (only 1/3 of " \
"the incoming light is used for a pixel's measured color in a standard Bayer pattern), that is to say that the detected " \
"counts N drop and the signal-to-noise ratio with it. The demosaicing process blurs fine detail and (obviously) mixes colors, " \
"so the real resolution is reduced. A telescope, on the other hand, uses a monochrome sensor and separate RGB " \
"filters so nearly every photon is captured."
)

# Output:
# Problem 2: Smartphone vs Telescope camera RGB
# Smartphones make color images by placing a color filter array over a single sensor. Each pixel measures 
# only one color, and the two missing colors are recovered through demosaicing, which is a post-shot processing method. 
# This would be a bad idea on a telescope because a color filter array gets rid of photons at every pixel (only 1/3 of the incoming 
# light is used for a pixel's measured color in a standard Bayer pattern), that is to say that the detected counts N drop and the 
# signal-to-noise ratio with it. The demosaicing process blurs fine detail and (obviously) mixes colors, so the real resolution 
# is reduced. A telescope, on the other hand, uses a monochrome sensor and separate RGB filters so nearly every photon is captured.

'''
Problem 3: Cosmic ray lookalike
'''
# In the question ”Attempt 3 - An unrealistic star”, the result actually looks more like the result of a 
# cosmic ray hitting the detector. Why? What about this image characterizes the ”star” as a cosmic ray hit?

# useful https://www.sciencedirect.com/science/article/abs/pii/S0026271416302827
# excellent demonstration tool https://astro.unl.edu/naap/vsp/ccds.html#:~:text=The%20light%20from%20a%20star,Plot%20to%20verify%20your%20estimate.
print("\nProblem 3: Cosmic ray lookalike pourquoi?")
print("The star blotch from 'Attempt 3' looks like a cosmic ray hit because it has hard edges and is " \
"perfectly uniform in color (all white). Cosmic rays do not interfere with the optics, and instead strike the silicon " \
"of a CCD directly. A real star would have a central peak and/or radial fall-off pattern, and " \
"would spread smoothly over multiple pixels. Even a very bright star would not have the hard edges and perfectly uniform " \
"intensity that is shown in 'Attempt 3'."
)

# Output:
# Problem 3: Cosmic ray lookalike pourquoi?
# The star blotch from 'Attempt 3' looks like a cosmic ray hit because it has hard edges and is perfectly uniform 
# in color (all white). Cosmic rays do not interfere with the optics, and instead strike the silicon of a CCD directly. 
# A real star would have a central peak and/or radial fall-off pattern, and would spread smoothly over multiple pixels. 
# Even a very bright star would not have the hard edges and perfectly uniform intensity that is shown in 'Attempt 3'.

'''
Problem 4: Hubble Space Telescope WFPC/2 Undersampling PSF
'''
# The Hubble Space Telescope used to contain an instrument called WFPC/2. This instrument ”undersampled 
# the telescope point spread function”. Explain, with the aid of a diagram, what this means.

# not particularly relevant to the question but informative nonetheless https://www.stsci.edu/hst/instrumentation/legacy/wfpc2
# https://www.cambridge.org/core/journals/publications-of-the-astronomical-society-of-australia/article/detector-sampling-of-opticalir-spectra-how-many-pixels-per-fwhm/530D2419763BAA8576733C007A865BBD
# https://www.highpointscientific.com/astronomy-hub/post/astro-photography-guides/undersampling-and-oversampling-in-astrophotography

print("\nProblem 4: Wide Field and Planetary Camera 2 undersampling")
print("In simple terms, the WFPC2 instrument's pixels were too big compared to the width of the HST's blurred star image, " \
"so each stellar object fell on too few pixels. For proper sampling, you want \u2265 2 pixels across the star's image in " \
"order for it to be recorded accurately. Each pixel is a sample of the scene, and Nyquist's rule says you " \
"need \u2265 2 samples across the smallest feature you want to capture. For example if a star's FWHM is 0.1'', " \
"your pixel scale should be \u2264 0.05''. If the pixels were larger, the star would fall in a single pixel " \
"across it's core, so the shape can't be reconstructed. The star would look (sorta) like a 1x1 pixel downsampling of " \
"the 'Attempt 3' star blotch." \
)
# Diagram: https://hst-docs.stsci.edu/drizzpac/chapter-3-description-of-the-drizzle-algorithm/3-1-imagereconstructionandrestorationtechnique
# Not too sure what was expected for the diagram but this depiction of undersampling should do