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

'''
5. Something more interesting.
'''
import numpy as np
import matplotlib.pyplot as plt

N, nmax, R = 500, 1000, 2.0
x = np.linspace(-1.75, 0.75, N) # using linspace for evenly spaced numbers over n - ni
y = np.linspace(-1.25, 1.25, N)
# https://www.youtube.com/watch?v=7K_a1mmraHU
X, Y = np.meshgrid(x, y) # creates two dimensional 'grid' from one-dimensional arrays x & y
c = X + 1j*Y # 1j is the imaginary number, where j represents 'i'

q = np.zeros_like(c) # Set c for that pixel using Equation 7: c = x+iy
iterations = np.zeros(c.shape, dtype=float) # .shape gives the dimension size of 'c' array in (x, y)

for n in range(1, nmax + 1): # stopping iteration and setting color to black at nmax (1000)
    q = q*q + c
    m = (iterations == 0) & (np.abs(q) > R)
    iterations[m] = n
    q[iterations > 0] = 0

cmap = plt.cm.turbo.copy()
cmap.set_bad('black')
img = np.ma.masked_equal(iterations, 0) 

plt.imshow(img, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap)
plt.xlabel("Real(c)"); plt.ylabel("Imaginary(c)")
plt.colorbar(label="Escape iteration n")
plt.tight_layout(); plt.show()
# good enough