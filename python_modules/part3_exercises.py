'''
1. Create a two-dimensional, 10 × 10 element array, in which each element is equal
to zero. Then, manually set the appropriate elements in the array to equal
100.0 such that, when the array is displayed as an image it shows the digit ”4”
(it does not have to be pretty, just recognisably a 4). Print the resulting array
to the screen and then display the array to the screen as an image, and compare
the two.'''
import numpy as np
import matplotlib.pyplot as plt

test_image = np.zeros((10,10), dtype=float)
test_image [2,5] = 214.0
test_image [7,3] = 214.0
print(test_image)
plt.imshow(test_image)