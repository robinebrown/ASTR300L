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
# # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
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
# im = plt.imshow(dist, )
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
# im = plt.imshow(v, )
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

# https://www.youtube.com/watch?v=BmpsWs-kNTM
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

# # 14.2 An Everyday Image

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
# np.set_printoptions(linewidth=200, suppress=True)
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

# # 14.3. Imaging: Increasingly real stars
# '''
# 1. Attempt 1 - Utter Nothingness:
# '''
# # Generate a two dimensional, 1024 × 1024 element array, in which each element is equal to zero. 
# # Display this ”image” to the screen. Then, in a separate command, output the image in pdf format
# # (hint: fig.savefig(’bunny.pdf’)). This might be the night sky if the universe was totally empty.

# import numpy as np
# import matplotlib.pyplot as plt

# img = np.zeros((1024, 1024), dtype=float)
# # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
# fig, ax = plt.subplots()
# # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html
# ax.imshow(img, cmap="gray", interpolation="nearest")
# ax.set_title("Utter Nothingness")
# ax.axis("off")

# print("14.3 Imaging: Increasingly real stars\nAttempt 1: Utter Nothingness -\nThe file will be saved as utter_nothingness.pdf and displayed as an image.")
# # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
# fig.savefig("bunny.pdf", bbox_inches="tight")

# plt.show()

# '''
# 2. Attempt 2 - Realistic Nothingness
# '''
# # Generate a two dimensional, 1024 × 1024 element array, in which each element contains a normally
# # distributed random number between 0.0 and 30.0. Display this ”image” to the screen, and
# # output it in pdf format. This image should look like noise (which is what we made the array to be). 
# # It might be an image of a completely empty night sky, but takem with a more realistic detector.

# import numpy as np
# import matplotlib.pyplot as plt

# rng = np.random.default_rng()

# # mean 15, std 5
# # https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.normal.html
# noise = rng.normal(loc=15.0, scale=5.0, size=(1024, 1024))
# # https://numpy.org/doc/2.3/reference/generated/numpy.clip.html
# # clipping to keep only values 0-30
# noise = np.clip(noise, 0.0, 30.0)

# fig, ax = plt.subplots()
# ax.imshow(noise, cmap="gray", interpolation="nearest")
# ax.set_title("Realistic Nothingness (with normally distributed noise, 0–30)")
# ax.axis("off")

# print("\nAttempt 2: Realistic nothingness -\nThe file will be saved as realistic_nothingness.pdf and displayed as an image.")
# fig.savefig("realistic_nothingness.pdf", bbox_inches="tight")

# plt.show()

# '''
# 3. Attempt 2 - Part 2
# '''
# # A huge factor in how an image looks is how the values in the array translate to the colormap. 
# # With the defaults, the min and max of the array are set to the min and max of the colormap. 
# # Let’s change that. Display the ”noise image” from just above so that 0.0 is black and 255.0 
# # (rather than 50.0) is white. Since the max value in the array is 30, this should make the brightest 
# # displayed color a dark grey. Make a pdf of the result. The image should now look like ”dark noise”.

# fig, ax = plt.subplots()
# ax.imshow(noise, cmap="gray", vmin=0.0, vmax=255.0, interpolation="nearest")
# ax.set_title("Dark Noise (display scaled 0, 255)")
# ax.axis("off")

# print("\nAttempt 2 Part 2: Dark noise -\nThe file will be saved as dark_noise.pdf and displayed as an image.")
# fig.savefig("dark_noise.pdf", bbox_inches="tight")
# plt.show()

# '''
# 4. Attempt 3 - Unrealistic Star
# '''
# # Take the ”noise” image from the previous question. Pick a position at random near the center of the image. 
# # Manually set all the elements of the array within a 5 × 5 region centered on that point to be all equal to 255.0. 
# # Redisplay this image using a greyscale colormap with black equal to 0.0 and white equal to 255.0. 
# # Congratulations on making a star!

# import numpy as np
# import matplotlib.pyplot as plt

# rng = np.random.default_rng()

# noise = rng.normal(loc=15.0, scale=5.0, size=(1024, 1024))
# noise = np.clip(noise, 0.0, 30.0)

# height, width = noise.shape
# r_center = rng.integers(height//2 - 20, height//2 + 21)
# c_center = rng.integers(width//2 - 20, width//2 + 21)

# half = 2
# r1, r2 = r_center - half, r_center + half + 1
# c1, c2 = c_center - half, c_center + half + 1

# star = noise.copy() # tried doing star = noise first but realised it was changing the original array #cringe
# star[r1:r2, c1:c2] = 255.0

# # black=0, white=255
# fig, ax = plt.subplots()
# ax.imshow(star, cmap="gray", vmin=0.0, vmax=255.0, interpolation="nearest")
# ax.set_title(f"Attempt 3: Unrealistic Star ({r_center},{c_center})")
# ax.axis("off")

# print("\nAttempt 3: Unrealistic star -\nThe file will be saved as unrealistic_star.pdf and displayed as an image.")
# fig.savefig("unrealistic_star.pdf", bbox_inches="tight")
# plt.show()

# '''
# 5. Attempt 4 - Slightly better star
# '''
# # The above is not a very realistic looking star, so let’s do better. Revert back to the ”noise” image with no star. and
# # pick a random position near the center. Hint: doing the following step as a function, where you send the function the
# # image and the appropriatae details of the ”star”, and the function returns a new image with the star inserted, will save you 
# # time in the long run. For each element in a 5 × 5 region centered on that point, randomly generate a number between 50 and 150 
# # and ADD it to the existing number in that element. So for example if the element contains 18 and you generate 107 for that element, 
# # the result should be 125. Display the new image, and congratulate yourself on making a realistic(ish) looking star.

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
# star[r1:r2, c1:c2] = star[r1:r2, c1:c2] + patch

# np.clip(star, 0.0, 255.0, out=star)

# fig, ax = plt.subplots()
# ax.imshow(star, cmap="gray", vmin=0.0, vmax=255.0, interpolation="nearest")
# ax.set_title(f"Attempt 4 – Slightly Better Star ({r_center},{c_center})")
# ax.axis("off")

# print("\nAttempt 4: Slightly better star -\nThe file will be saved as slightly_better_star.pdf and displayed as an image.")
# fig.savefig("slightly_better_star.pdf", bbox_inches="tight")
# plt.show()

# # rewritten using function as suggested 
# import numpy as np
# import matplotlib.pyplot as plt

# def insert_star(image, size=5, add_low=50, add_high=150, center_window=20):
#     """
#     INPUTS:
#     image (array), size (default=5), add_low (default=50), add_high (default=150), center_window (default=20))
#     OUTPUTS:
#     new_image, (r_center, c_center)
#     DESCRIPTION:
#     Adds a 5×5 “star” by adding random values in [add_low, add_high] to a patch centered near middle of the image. Returns a copy.
#     """
#     rng = np.random.default_rng()
#     new_image = image.copy()
#     height, width = new_image.shape

#     # centering
#     r_center = rng.integers(height//2 - center_window, height//2 + center_window + 1)
#     c_center = rng.integers(width//2 - center_window, width//2 + center_window + 1)

#     # patch
#     half = size // 2
#     r1, r2 = r_center - half, r_center + half + 1
#     c1, c2 = c_center - half, c_center + half + 1

#     # per-pixel additions, [add_low, add_high]
#     add_patch = rng.integers(add_low, add_high + 1, size=(size, size))
#     new_image[r1:r2, c1:c2] = new_image[r1:r2, c1:c2] + add_patch

#     return new_image, (r_center, c_center)

# # base noise image (no star)
# rng = np.random.default_rng()
# noise = rng.normal(15.0, 5.0, (1024, 1024))
# noise = np.clip(noise, 0.0, 30.0)

# # adding star to noise image
# star, (r_center, c_center) = insert_star(noise, size=5, add_low=50, add_high=150, center_window=20)

# fig, ax = plt.subplots()
# ax.imshow(star, cmap="gray", vmin=0.0, vmax=255.0, interpolation="nearest")
# ax.set_title(f"Attempt 4 – Slightly Better Star ({r_center},{c_center})")
# ax.axis("off")

# print("\nAttempt 4: Slightly better star -\nThe file will be saved as slightly_better_star.pdf and displayed as an image.")
# fig.savefig("better_star.pdf", bbox_inches="tight")
# plt.show()

# '''
# 6. Attempt 5 - A field of stars
# '''
# # In this attempt we will add multiple stars of different sizes to the image and make the bigger ones more likely to be brighter.
# # We will do this by automatically generating them. Use numpy to randomly pick twenty positions (so, x, y coordinate) in the ”image”, 
# # each at least 5 pixels from the edge of the array. Let’s call these positions α. For each α, generate a random ODD number between 3 and 15.
# # Let’s call these numbers β. Centered on each α, consider a β × β region. For each eleent in this region, generate a random number 
# # between 10 and 60, nultiply it by β/2 (being sure to use the appropriate β for that α) and add the result to the
# # number already in that element. Display the final ”image”, using a greyscale colormap with black equal to 0.0 and white equal to 255.0. 
# # Congratulations on making a whole field of stars!

# import numpy as np
# import matplotlib.pyplot as plt

# def insert_star_field(image, n_stars=20, edge=5):
#     """
#     INPUTS:
#         image (array), n_stars (default=20), edge (default=5)
#     OUTPUTS:
#         new_image
#     DESCRIPTION:
#         Adds n_stars stars. Each star uses an odd β in {3,5,7,9,11,13,15}; a β×β patch gets + (uniform[10,60] * β/2) per pixel.
#     """
#     rng = np.random.default_rng()
#     new_image = image.copy()
#     height, width = new_image.shape

#     for _ in range(n_stars): # '_' because function isn't using the loop variable
#         r_center = rng.integers(edge, height - edge) # no closer than 5 pixels from border (edge default)
#         c_center = rng.integers(edge, width  - edge) # same as above
#         # https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html
#         beta = rng.choice([3, 5, 7, 9, 11, 13, 15])

#         half = beta // 2
#         r1, r2 = r_center - half, r_center + half + 1
#         c1, c2 = c_center - half, c_center + half + 1

#         add_patch = rng.integers(10, 61, size=(beta, beta)) * (beta / 2.0)
#         new_image[r1:r2, c1:c2] += add_patch

#     return new_image

# # base noise image (no star)
# rng = np.random.default_rng()
# noise = rng.normal(15.0, 5.0, (1024, 1024))
# noise = np.clip(noise, 0.0, 30.0)

# # adding a field of stars
# stars = insert_star_field(noise, n_stars=20, edge=5)

# fig, ax = plt.subplots()
# ax.imshow(stars, cmap="gray", vmin=0.0, vmax=255.0, interpolation="nearest")
# ax.set_title("Attempt 5 – Field of Stars")
# ax.axis("off")

# print("\nAttempt 5: Field of Stars -\nThe file will be saved as field_of_stars.pdf and displayed as an image.")
# fig.savefig("field_of_stars.pdf", bbox_inches="tight")
# plt.show()

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
        Returns a gamma×gamma centered 2D Gaussian (FWHM=2) with peak=epsilon. If gamma is even, it is incremented to the next odd.
    """
    if gamma % 2 == 0: # making gamma odd
        gamma += 1
    half = gamma // 2
    # https://www.youtube.com/watch?v=gdeV4UeljUY
    # https://numpy.org/doc/stable/reference/generated/numpy.mgrid.html
    # need +1 to make half inclusive
    yy, xx = np.mgrid[-half:half+1, -half:half+1]
    sigma = 2.0 / (2.0 * np.sqrt(2.0 * np.log(2.0))) # from FWHM=2; helpful https://statproofbook.github.io/P/norm-fwhm.html
    template = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    template *= (epsilon / template[half, half])
    
    return template

def insert_star_field(image, n_stars=20, edge=7):
    """
    INPUTS:
        image (array), n_stars (default=20), edge (default=7)
    OUTPUTS:
        new_image
    DESCRIPTION:
        Adds n_stars Gaussian stars. Each star uses odd β in {3,5,7,9,11,13,15}, center ≥ edge from border, peak ε = [10,60]*(β/2).
    """
    rng = np.random.default_rng()
    new_image = image.copy()
    height, width = new_image.shape

    for _ in range(n_stars):
        beta = int(rng.choice([3, 5, 7, 9, 11, 13, 15]))
        half = beta // 2
        r_center = rng.integers(edge, height - edge)
        c_center = rng.integers(edge, width  - edge)
        epsilon = rng.integers(10, 61) * (beta / 2.0)
        template = make_star_template(beta, epsilon)
        r1, r2 = r_center - half, r_center + half + 1
        c1, c2 = c_center - half, c_center + half + 1
        new_image[r1:r2, c1:c2] += template

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

# 14.4 File I/O Exercises





