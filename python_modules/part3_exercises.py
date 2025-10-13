# '''
# 1. Create a two-dimensional, 10 × 10 element array, in which each element is equal
# to zero. Then, manually set the appropriate elements in the array to equal
# 100.0 such that, when the array is displayed as an image it shows the digit ”4”
# (it does not have to be pretty, just recognisably a 4). Print the resulting array
# to the screen and then display the array to the screen as an image, and compare
# the two.'''
# import numpy as np
# import matplotlib.pyplot as plt

# print("14.1 Imaging: A Little Warmup")
# print("PROBLEM 1: Create a two dimensional 10x10 array where each element = 100.0, such that when it is displayed it looks like the number 4. Also display the image.")

# image_four = np.zeros((10,10), dtype=float)

# image_four [0,3] = 100.0
# image_four [1,3] = 100.0
# image_four [2,3] = 100.0
# image_four [3,3] = 100.0
# image_four [4,3] = 100.0
# image_four [5,3] = 100.0
# image_four [5,4] = 100.0
# image_four [5,5] = 100.0
# image_four [5,6] = 100.0
# image_four [0,6] = 100.0
# image_four [1,6] = 100.0
# image_four [2,6] = 100.0
# image_four [3,6] = 100.0
# image_four [4,6] = 100.0
# image_four [5,6] = 100.0
# image_four [6,6] = 100.0
# image_four [7,6] = 100.0
# image_four [8,6] = 100.0
# image_four [9,6] = 100.0

# print("10x10 array:")
# print(image_four)
# plt.imshow(image_four)
# plt.show()

# # Output:
# # 14.1 Imaging: A Little Warmup
# # PROBLEM 1: Create a two dimensional 10x10 array where each element = 100.0, such that when it is displayed it looks like the number 4. Also display the image.
# # 10x10 array:
# # [[  0.   0.   0. 100.   0.   0. 100.   0.   0.   0.]
# #  [  0.   0.   0. 100.   0.   0. 100.   0.   0.   0.]
# #  [  0.   0.   0. 100.   0.   0. 100.   0.   0.   0.]
# #  [  0.   0.   0. 100.   0.   0. 100.   0.   0.   0.]
# #  [  0.   0.   0. 100.   0.   0. 100.   0.   0.   0.]
# #  [  0.   0.   0. 100. 100. 100. 100.   0.   0.   0.]
# #  [  0.   0.   0.   0.   0.   0. 100.   0.   0.   0.]
# #  [  0.   0.   0.   0.   0.   0. 100.   0.   0.   0.]
# #  [  0.   0.   0.   0.   0.   0. 100.   0.   0.   0.]
# #  [  0.   0.   0.   0.   0.   0. 100.   0.   0.   0.]]

# '''
# 2. Same as above, but instead display the digit ’9’, and when displaying the image,
# use the ’inferno’ color map where the digit itself is displayed using a color near
# the center of the color map. All other ”pixels” should be a single color, from
# an extreme end of the color map (your choice which).
# '''
# import numpy as py
# import matplotlib.pyplot as plt

# image_nine = np.zeros((10,10), dtype=float)

# image_nine [0,4] = 100.0
# image_nine [0,5] = 100.0
# image_nine [1,3] = 100.0
# image_nine [2,3] = 100.0
# image_nine [3,3] = 100.0
# image_nine [4,3] = 100.0
# image_nine [5,4] = 100.0
# image_nine [5,5] = 100.0
# image_nine [5,6] = 100.0
# image_nine [0,6] = 100.0
# image_nine [1,6] = 100.0
# image_nine [2,6] = 100.0
# image_nine [3,6] = 100.0
# image_nine [4,6] = 100.0
# image_nine [5,6] = 100.0
# image_nine [6,6] = 100.0
# image_nine [7,6] = 100.0
# image_nine [8,6] = 100.0
# image_nine [9,6] = 100.0

# print("10x10 array:")
# print(image_nine)
# plt.imshow(image_nine, cmap='inferno')
# plt.show()

# '''
# 3. Generate an N × N numpy array of all zeros, where 50 < N < 200. Then,
# pick a random point in this array that is at least ten picels away from the
# origin in both x and y directions. After that, write a funcion that calculates
# the Euclidean distance between any two coordinate positions, and returns that
# distance (which can be a float). Use this function, and a color map of your
# choice that is nevertheless suited to the problem, to color code every pixel in
# your image by its Euclidean distance from the random;y chosen point.
# '''
# import numpy as np
# import matplotlib.pyplot as plt

# # https://stackoverflow.com/questions/74343474/reproduce-numpy-random-numbers-with-numpy-rng
# rng = np.random.default_rng()
# N = int(rng.integers(51, 200))

# yy, xx = np.indices((N, N))

# x0 = int(rng.integers(10, N))
# y0 = int(rng.integers(10, N))

# dist = np.hypot(xx - x0, yy - y0)  # same as sqrt((x-x0)^2 + (y-y0)^2)

# plt.figure(figsize=(6,6))
# im = plt.imshow(dist, origin="upper")
# plt.title(f"Euclidean distance from ({x0}, {y0}) in a {N}×{N} image")
# plt.colorbar(im, fraction=0.046, pad=0.04, label="Distance (pixels)")
# plt.axis("off")
# plt.show()


# '''
# 4. Same as above, but this time, instead of the Euclidean distance, determine the
# values of each pixel via the function:
# v(x, y) = asin 2πx/b + ccos 2πy/d
# Experiment with the values of a,b,c,d to see what they do. Pick an appropriate
# color map and values of these parameters so that the displayed result looks like
# ”rippling water”.
# '''
# import numpy as np
# import matplotlib.pyplot as plt

# # grid
# N = 150
# yy, xx = np.indices((N, N))

# a, b = 1.0, 18.0
# c, d = 1.0, 24.0

# v = a*np.sin(2*np.pi*xx/b) + c*np.cos(2*np.pi*yy/d)

# plt.figure(figsize=(6,6))
# im = plt.imshow(v, origin="upper")
# plt.title("v(x,y) = a sin(2πx/b) + c cos(2πy/d)")
# plt.axis("off")
# plt.colorbar(im, fraction=0.046, pad=0.04)
# plt.show()

# '''
# 5. Something more interesting.
# '''
# import numpy as np
# import matplotlib.pyplot as plt

# N, nmax, R = 500, 1000, 2.0
# x = np.linspace(-1.75, 0.75, N) # using linspace for evenly spaced numbers over n - ni
# y = np.linspace(-1.25, 1.25, N)
# # https://www.youtube.com/watch?v=7K_a1mmraHU
# X, Y = np.meshgrid(x, y) # creates two dimensional 'grid' from one-dimensional arrays x & y
# c = X + 1j*Y # 1j is the imaginary number, where j represents 'i'

# q = np.zeros_like(c) # Set c for that pixel using Equation 7: c = x+iy
# iterations = np.zeros(c.shape, dtype=float) # .shape gives the dimension size of 'c' array in (x, y)

# for n in range(1, nmax + 1): # stopping iteration and setting color to black at nmax (1000)
#     q = q*q + c
#     m = (iterations == 0) & (np.abs(q) > R)
#     iterations[m] = n
#     q[iterations > 0] = 0

# cmap = plt.cm.turbo.copy()
# cmap.set_bad('black')
# img = np.ma.masked_equal(iterations, 0) 

# plt.imshow(img, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap)
# plt.xlabel("Real(c)"); plt.ylabel("Imaginary(c)")
# plt.colorbar(label="Escape iteration n")
# plt.tight_layout(); plt.show()
# # good enough

# 14.2 An Everyday Image

# from scipy import datasets
# import numpy as np
# import matplotlib.pyplot as plt

# face = datasets.face(gray=True)

# '''
# 1. Default display:
# '''
# plt.figure()
# plt.imshow(face, cmap="gray")
# plt.title("1. Default display")
# plt.axis("off")

# '''
# 2. Min/max and two rescaled displays
# '''
# vmin = float(face.min())
# vmax = float(face.max())
# print(f"2. minimum pixel = {vmin}, maximum pixel = {vmax}")

# # (min, max/2.0)
# plt.figure()
# plt.imshow(face, cmap="gray", vmin=vmin, vmax=vmax/2.0)
# plt.title("2. Scaled to (min, max/2)")
# plt.axis("off")

# # (min*2.0, max)
# plt.figure()
# plt.imshow(face, cmap="gray", vmin=vmin*2.0, vmax=vmax)
# plt.title("2. Scaled to (min*2, max)")
# plt.axis("off")

# plt.show()

# '''
# 3. Print any 30x30 element section to the screen
# '''
# r0, c0 = 200, 300
# block = face[r0:r0+30, c0:c0+30]
# npp.set_printoptions(linewidth=200, suppress=True)
# print(f"3. 30x30 block starting at (row {r0}, col {c0}):")
# print(block)

# '''
# 4) Remove 100px border and compare
# '''
# cropped = face[100:-100, 100:-100]

# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.imshow(face, cmap="gray")
# plt.title("4. Original")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(cropped, cmap="gray")
# plt.title("4. Cropped (100 pixel border removed)")
# plt.axis("off")

# plt.tight_layout()
# plt.show()

# '''
# 5 Row-wise sort (in ascending order) and compare to original
# '''
# sorted_rows = face.copy()
# for i in range(sorted_rows.shape[0]):
#     sorted_rows[i, :] = np.sort(sorted_rows[i, :])

# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.imshow(face, cmap="gray")
# plt.title("5. Original")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(sorted_rows, cmap="gray")
# plt.title("5. Row-wise sorted")
# plt.axis("off")

# plt.tight_layout()
# plt.show()

# '''
# 6. Index sorted rows
# '''
# # https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html
# original = face  # use the original image
# H, W = original.shape
# weird = np.empty_like(original)

# # Sort ascending
# first_order = np.argsort(original[0])
# weird[0] = original[0][first_order]

# # For each next row i, apply the permutation that would sort row i-1 (of the ORIGINAL)
# for i in range(1, H):
#     order = np.argsort(original[i-1])
#     weird[i] = original[i][order]

# # Show original vs. index-sorted
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.imshow(original, cmap="gray")
# plt.title("Original")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(weird, cmap="gray")
# plt.title("Sorted by previous row's order")
# plt.axis("off")

# plt.tight_layout()
# plt.show()

# 14.3. Imaging: Increasingly real stars
'''
1. Attempt 1 - Utter Nothingness:
'''
import numpy as np
import matplotlib.pyplot as plt

img = np.zeros((1024, 1024), dtype=float)
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
fig, ax = plt.subplots(figsize=(6, 6))
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html
ax.imshow(img, cmap="gray", origin="upper", interpolation="nearest")
ax.axis("off")

print("14.3 Imaging: Increasingly real stars\nProblem 1: Utter Nothingness -\nThe file will be saved as bunny.pdf and displayed as an image.")
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
fig.savefig("bunny.pdf", bbox_inches="tight")

plt.show()

'''
2. Attempt 2 - Realistic Nothingness
'''
import numpy as np
import matplotlib.pyplot as plt

range = np.random.default_rng()

# mean 15, std 5
noise = range.normal(loc=15.0, scale=5.0, size=(1024, 1024))
# https://numpy.org/doc/2.3/reference/generated/numpy.clip.html
noise = np.clip(noise, 0.0, 30.0)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(noise, cmap="gray", origin="upper", interpolation="nearest")
ax.set_title("Realistic Nothingness (with normally distributed noise, 0–30)")
ax.axis("off")

print("\nProblem 2: Realistic Nothingness -\nThe file will be saved as realistic_nothingness.pdf and displayed as an image.")
fig.savefig("realistic_nothingness.pdf", bbox_inches="tight")

plt.show()

# Part 2 (Problem 4.)

fig3, ax3 = plt.subplots(figsize=(6, 6))
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
ax3.imshow(noise, cmap="gray", vmin=0.0, vmax=255.0, origin="upper", interpolation="nearest")
ax3.set_title("Dark Noise (display scaled 0, 255)")
ax3.axis("off")
fig3.savefig("dark_noise.pdf", bbox_inches="tight")
plt.show()

'''
3. Attempt 3 - Unrealistic Star
'''
import numpy as np
import matplotlib.pyplot as plt

range = np.random.default_rng()

# mean 15, std 5
noise = range.normal(loc=15.0, scale=5.0, size=(1024, 1024))
noise = np.clip(noise, 0.0, 30.0)

height, width = noise.shape
r_center = int(range.integers(height//2 - 20, width//2 + 21))
c_center = int(range.integers(height//2 - 20, width//2 + 21))






