# # 18.1 18.2 18.7
# '''
# 1. Linear model plot for Y = αX + β
# '''
# # 1. Write a Python function that returns a one-dimensional numpy array Y, given an input 
# # array X, using the following expression: Y = αX + β (17) Generate a 100 element numpy 
# # array running from zero to unity to serve as the X values. Choose appropriate values for 
# # each model parameter (they can be anything you like, but you might want to read the rest 
# # of this question before choosing). Make a publication-quality plot of the X and Y data, plotted 
# # as a solid black line. Axes should be labelled, and there should be a legend.
# import numpy as np
# import matplotlib.pyplot as plt

# def linear_model(X, alpha, beta):
#     return alpha * X + beta

# # data
# X = np.linspace(0.0, 1.0, 100)
# Y = linear_model(X, alpha=2.0, beta=0.5)

# print("18.1 Introductory Exercises\nProblem 1: Linear model plot for Y = αX + β")

# # No idea what a "publication-quality" plot is supposed to look like but here's this
# # could specify "color" and "linestyle" but I found that matplotlib has easy single-string modifiers
# # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
# plt.plot(X, Y, '-k', label='Y = 2.0X + 0.5') # 'k-' here because k is color (black) and - is line type (solid)
# plt.title("Linear Model")
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend() # assuming default is ok
# plt.tight_layout()
# plt.savefig('xy_line.pdf', bbox_inches='tight')
# plt.show()

# # Output:
# # 18.1 Introductory Exercises
# # Problem 1: Linear model plot for Y = αX + β

# '''
# 2. Cosine model plot for Y = α cos(βX + γ)
# '''
# # Same as the above, but for the expression: Y = α cos(βX + γ)

# import numpy as np
# import matplotlib.pyplot as plt

# def cos_model(X, alpha, beta, gamma):
#     return alpha * np.cos(beta * X + gamma)

# # data
# X = np.linspace(0.0, 1.0, 100)
# Y = cos_model(X, alpha=1.5, beta=4*np.pi, gamma=0.3)

# print("\nProblem 2: Cosine model plot for Y = α cos(βX + γ)")

# plt.plot(X, Y, '-k', label='Y = 1.5·cos(4πX + 0.3)')
# plt.title("Cosine Model")
# plt.xlabel('Horizontal Axis')
# plt.ylabel('Vertical Axis')
# plt.legend()
# plt.tight_layout()
# plt.savefig('xy_cosine.pdf', bbox_inches='tight')
# plt.show()

# # Output:
# # Problem 2: Cosine model plot for Y = α cos(βX + γ)

# '''
# 3. Quadratic model plot for Y = αX^2 + βX + γ
# '''
# # Same as question 1, but for the expression: Y = αX2 + βX + γ

# import numpy as np
# import matplotlib.pyplot as plt

# def quadratic_model(X, alpha, beta, gamma):
#     return alpha * X**2 + beta * X + gamma

# # data
# X = np.linspace(0.0, 1.0, 100)
# Y = quadratic_model(X, alpha=1, beta=-1, gamma=1)

# print("\nProblem 3: Quadratic model plot for Y = αX^2 + βX + γ")

# plt.plot(X, Y, '-k', label='Y = 1X² − 1X + 1')
# plt.title("Quadratic Model")
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.tight_layout()
# plt.savefig('xy_quadratic.pdf', bbox_inches='tight')
# plt.show()

# # Output:
# # Problem 3: Quadratic model plot for Y = αX^2 + βX + γ

# '''
# 4. Exponential model plot for Y = α e^X + β
# '''
# # Same as question 1, but for the expression: Y = αeX + β

# import numpy as np
# import matplotlib.pyplot as plt

# def exponential_model(X, alpha, beta):
#     return alpha * np.exp(X) + beta

# # data
# X = np.linspace(0.0, 1.0, 100)
# Y = exponential_model(X, alpha=0.8, beta=0.2)

# print("\nProblem 4: Exponential model plot for Y = α e^X + β")

# plt.plot(X, Y, '-k', label='Y = 0.8·e^X + 0.2')
# plt.title("Exponential Model")
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.tight_layout()
# plt.savefig('xy_exponential.pdf', bbox_inches='tight')
# plt.show()

# # Output:
# # Problem 4: Exponential model plot for Y = α e^X + β

# '''
# 5. Special-function model plot for Y = α·J₀(βX + γ); cylindrical Bessel function
# '''
# # Same as question 1, but for any function (your choice) within scipy.special.

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import special

# # found this 'faster' Bessel function on the scipy docs
# def j0_model(X, alpha, beta, gamma): # https://docs.scipy.org/doc/scipy/reference/special.html
#     return alpha * special.j0(beta * X + gamma)

# X = np.linspace(0.0, 1.0, 100)
# Y = j0_model(X, alpha=1.0, beta=20.0*np.pi, gamma=0.0)

# print("\nProblem 5b: Special-function model plot for Y = α·J₀(βX + γ); cylindrical Bessel function")

# plt.plot(X, Y, '-k', label='Y = 1.0·J₀(20πX)')
# plt.title("Cylindrical Bessel Function")
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.tight_layout()
# plt.savefig('xy_special_besselj0.pdf', bbox_inches='tight')
# plt.show()

# # Output:
# Problem 5b: Special-function model plot for Y = α·J₀(βX + γ) [cylindrical Bessel function]

'''
6. Combined plot: linear, cosine, quadratic, exponential
'''
# Make a single plot in which all four of the above expressions (non-special) are plotted. Adjust the 
# parameters of each model so they all are displayed ’as informatively as possible’ on the same plot, given 
# the same X values in all cases. Label each model in the legend, and make each line a different color 
# (hint: the tableau colors are designed to differentiate data of this type - categorical).

import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0.0, 3.0, 100)

linear  = 1.5 * X + 0.0
cosine  = 1.0 * np.cos(4*np.pi * X)
quadratic = 1.0 * X**2 - 1.0 * X
exponential  = 1.0 * np.exp(X) + 0.0

print("\nProblem 6: Combined plot for linear, cosine, quadratic, exponential")

plt.plot(X, linear,  color='tab:blue',  label='Linear: Y = 1.5X + 0.0')
plt.plot(X, cosine,  color='tab:orange',label='Cosine: Y = 1.0·cos(4πX)')
plt.plot(X, quadratic, color='tab:green', label='Quadratic: Y = X² − X')
plt.plot(X, exponential,  color='tab:red',   label='Exponential: Y = 1.0·e^X')
plt.title("Combined plot of linear, cosine, quadratic, and exponential models")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.tight_layout()
plt.savefig('xy_all_models.pdf', bbox_inches='tight')
plt.show()

# Output:
# Problem 6: Combined plot for linear, cosine, quadratic, exponential

# '''
# 7. Noisy quadratic: Y = αX^2 + βX + γ with X sampled from N(X_true, σ)
# '''
# # Write a new function that returns a set of y values, given an input set of x values, for the relation in 
# # Equation 19, taking in the same free parameters as before. This time however, the function should return 
# # a set of y values that also include an amount of normally distributed noise. In other words, the function 
# # should take the input x values, and then generate a new set of x values, each one drawn from a normal distribution 
# # centered on the ’true’ input x value. The FWHM of the normal distribution can be the same for each x value, but 
# # should be a new input parameter to the function.

# # import numpy as np
# # import matplotlib.pyplot as plt

# # def quadratic_model(X, alpha, beta, gamma):
# #     return alpha * X**2 + beta * X + gamma

# # def quadratic_noisy_matrix(X, alpha, beta, gamma, fwhm, rng, n=100):
# #     sigma = fwhm / 2.3548200450309493
# #     Xn = rng.normal(loc=X, scale=sigma, size=(n, X.size))
# #     return alpha * Xn**2 + beta * Xn + gamma  # shape: (n, len(X))

# # rng = np.random.default_rng()
# # X = np.linspace(0.0, 1.0, 100)
# # alpha, beta, gamma = 1.2, -0.8, 0.2
# # fwhm = 0.05

# # Y_true = quadratic_model(X, alpha, beta, gamma)
# # Y_noisy = quadratic_noisy_matrix(X, alpha, beta, gamma, fwhm, rng, n=100)

# # print("\nProblem 7: Noisy quadratic Y from X sampled with given FWHM")

# # plt.plot(X, Y_noisy.T, linewidth=0.8, alpha=0.15, color='tab:blue')
# # plt.plot(X, Y_true, linewidth=2.0, color='k', label='True quadratic')
# # plt.xlabel('X')
# # plt.ylabel('Y')
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig('xy_quadratic_true_vs_noisy_vectorized.pdf', bbox_inches='tight')
# # plt.show()

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()
X = np.linspace(0.0, 1.0, 100)
a, b, gamma = 1, -1, 1
sigma = 0.05 / 2.3548200450309493
Xn = rng.normal(X, sigma, size=(100, X.size)) # Gaussian
Y_noisy = a*Xn**2 + b*Xn + gamma
Y_true  = a*X**2  + b*X  + gamma

print("\nProblem 7: Noisy quadratic Y from X sampled with given FWHM")

plt.plot(X, Y_noisy.T, linewidth = 1, color = 'blue', alpha = 0.25)
plt.plot(X, Y_true, 'k', linewidth=2, label='True quadratic')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.tight_layout()
plt.savefig('xy_quadratic_true_vs_noisy_min.pdf', bbox_inches='tight')
plt.show()


# Output:
# Problem 7: Noisy quadratic Y from X sampled with given FWHM

# '''
# 8. 100 element array from 0 to 10 α = 2, β = 3, γ = 5
# '''
# # 8. Finally: Use your new function and a 100 element x array running from 0 to
# # 10 to answer the following question: For α = 2, β = 3, γ = 5, approximately
# # what value of the normal distribution width is needed for the plotted relation
# # to no longer resemble (as judged qualitatively) a polynomial? Hint, you do not
# # need to do more than generate a fair number of plots while experimentig with
# # the value of the normal distribution width.

# import numpy as np
# import matplotlib.pyplot as plt

# def quadratic_model(X, alpha, beta, gamma):
#     return alpha * X**2 + beta * X + gamma

# def quadratic_noisy_matrix(X, alpha, beta, gamma, fwhm, rng, n=100):
#     sigma = fwhm / 2.3548200450309493
#     Xn = rng.normal(loc=X, scale=sigma, size=(n, X.size))
#     return alpha * Xn**2 + beta * Xn + gamma  # shape: (n, len(X))

# rng = np.random.default_rng()
# X = np.linspace(0.0, 10.0, 100)
# a = 2.0
# b = 3.0
# gamma = 5.0
# FWHM = [0.1, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5]

# print("\nProblem 8: Visual sweep over FWHM to judge departure from polynomial")

# fig, axes = plt.subplots(2, 4, sharex=True, sharey=True)

# for ax, f in zip(axes.ravel(), FWHM):
#     Y_true = quadratic_model(X, a, b, g)
#     Y_noisy = quadratic_noisy_matrix(X, a, b, g, f, rng, n = 100)
#     ax.plot(X, Y_noisy.T, linewidth = 1.0, zorder = 1)
#     ax.plot(X, Y_true, 'k', linewidth = 1.0, zorder = 2)

# fig.tight_layout()
# plt.plot()
















# #     ax.plot(X, Y_noisy.T, lw=0.6, alpha=0.12, zorder=1)
# #     ax.plot(X, Y_true, 'k', lw=2.0, zorder=2, label='True')
# #     ax.set_title(f'FWHM = {f}')
# # for ax in axes[-1]:
# #     ax.set_xlabel('X')
# # for ax in axes[:,0]:
# #     ax.set_ylabel('Y')
# # fig.tight_layout()
# # fig.savefig('q8_fwhm_grid.pdf', bbox_inches='tight')
# # plt.show()
