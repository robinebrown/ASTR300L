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
print("PROBLEM 1: Create a two dimensional 10x10 array where each element = 100.0, such that when it is displayed it looks like the number 4. Also display the image.")

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

image_nine = np.zeros((10,10), dtype=float)

image_nine [0,4] = 100.0
image_nine [0,5] = 100.0
image_nine [1,3] = 100.0
image_nine [2,3] = 100.0
image_nine [3,3] = 100.0
image_nine [4,3] = 100.0
image_nine [5,4] = 100.0
image_nine [5,5] = 100.0
image_nine [5,6] = 100.0
image_nine [0,6] = 100.0
image_nine [1,6] = 100.0
image_nine [2,6] = 100.0
image_nine [3,6] = 100.0
image_nine [4,6] = 100.0
image_nine [5,6] = 100.0
image_nine [6,6] = 100.0
image_nine [7,6] = 100.0
image_nine [8,6] = 100.0
image_nine [9,6] = 100.0

print("10x10 array:")
print(image_nine)
plt.imshow(image_nine, cmap='inferno')
plt.show()