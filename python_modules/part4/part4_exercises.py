'''
1. Linear model plot for Y = αX + β
'''
# 1. Write a Python function that returns a one-dimensional numpy array Y, given an input 
# array X, using the following expression: Y = αX + β (17) Generate a 100 element numpy 
# array running from zero to unity to serve as the X values. Choose appropriate values for 
# each model parameter (they can be anything you like, but you might want to read the rest 
# of this question before choosing). Make a publication-quality plot of the X and Y data, plotted 
# as a solid black line. Axes should be labelled, and there should be a legend.

import numpy as np
import matplotlib.pyplot as plt

def linear_model(X, alpha, beta):
    """
    INPUTS: X (100 element numpy array), alpha (float; slope value), beta (float; intercept)
    OUTPUTS: Y (100 element numpy array)
    DESCRIPTION: Computes linear relation, y = ax + b, for X
    """
    return alpha * X + beta

# data
X = np.linspace(0.0, 1.0, 100)
Y = linear_model(X, alpha=2.0, beta=0.5)

print("18.1 Introductory Exercises\nProblem 1: Linear model plot for Y = αX + β")

# No idea what a "publication-quality" plot is supposed to look like but here's this
# could specify "color" and "linestyle" but I found that matplotlib has easy single-string modifiers
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
plt.plot(X, Y, '-k', label='Y = 2.0X + 0.5') # '-k' here because k is black and - is line type (solid)
plt.title("Linear Model")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend() # I'm assuming default is ok
plt.tight_layout()
plt.savefig('xy_line.pdf', bbox_inches='tight')
plt.show()

# Output:
# 18.1 Introductory Exercises
# Problem 1: Linear model plot for Y = αX + β

'''
2. Cosine model plot for Y = α cos(βX + γ)
'''
# Same as the above, but for the expression: Y = α cos(βX + γ)

import numpy as np
import matplotlib.pyplot as plt

def cos_model(X, alpha, beta, gamma):
    """
    INPUTS: X (100 element numpy array), alpha (float; amplitude), beta (float; scale), gamma (float: phase)
    OUTPUTS: Y (100 element numpy array)
    DESCRIPTION: Computes cosine relation, y = a * cos(b * x + c), for X
    """
    return alpha * np.cos(beta * X + gamma)

# data
X = np.linspace(0.0, 1.0, 100)
Y = cos_model(X, alpha=1.5, beta=4*np.pi, gamma=0.3)

print("\nProblem 2: Cosine model plot for Y = α cos(βX + γ)")

plt.plot(X, Y, '-k', label='Y = 1.5·cos(4πX + 0.3)')
plt.title("Cosine Model")
plt.xlabel('Horizontal Axis')
plt.ylabel('Vertical Axis')
plt.legend()
plt.tight_layout()
plt.savefig('xy_cosine.pdf', bbox_inches='tight')
plt.show()

# Output:
# Problem 2: Cosine model plot for Y = α cos(βX + γ)

'''
3. Quadratic model plot for Y = αX^2 + βX + γ
'''
# Same as question 1, but for the expression: Y = αX2 + βX + γ

import numpy as np
import matplotlib.pyplot as plt

def quadratic_model(X, alpha, beta, gamma):
    """
    INPUTS: X (100 element numpy array), alpha (float; quadratic), beta (float; linear), gamma (float; constant)
    OUTPUTS: Y (100 element numpy array)
    DESCRIPTION: Computes the quadratic relation, y = a * x**2 + b * x + γ, for X
    """
    return alpha * X**2 + beta * X + gamma

X = np.linspace(0.0, 1.0, 100)
Y = quadratic_model(X, alpha=1, beta=-1, gamma=1)

print("\nProblem 3: Quadratic model plot for Y = αX^2 + βX + γ")

plt.plot(X, Y, '-k', label='Y = 1X² − 1X + 1')
plt.title("Quadratic Model")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.tight_layout()
plt.savefig('xy_quadratic.pdf', bbox_inches='tight')
plt.show()

# Output:
# Problem 3: Quadratic model plot for Y = αX^2 + βX + γ

'''
4. Exponential model plot for Y = α e^X + β
'''
# Same as question 1, but for the expression: Y = αeX + β

import numpy as np
import matplotlib.pyplot as plt

def exponential_model(X, alpha, beta):
    """
    INPUTS: X (100 element numpy array), alpha (float; scale), beta (float, offset)
    OUTPUTS: Y (100 element numpy array)
    DESCRIPTION: Computes the exponential relation, y = a * exp(x) + b, for X
    """
    return alpha * np.exp(X) + beta

# data
X = np.linspace(0.0, 1.0, 100)
Y = exponential_model(X, alpha=0.8, beta=0.2)

print("\nProblem 4: Exponential model plot for Y = α e^X + β")

plt.plot(X, Y, '-k', label='Y = 0.8·e^X + 0.2')
plt.title("Exponential Model")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.tight_layout()
plt.savefig('xy_exponential.pdf', bbox_inches='tight')
plt.show()

# Output:
# Problem 4: Exponential model plot for Y = α e^X + β

'''
5. Special-function model plot for Y = α·J₀(βX + γ); cylindrical Bessel function
'''
# Same as question 1, but for any function (your choice) within scipy.special.

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

# found this 'faster' Bessel function on the scipy docs
def j0_model(X, alpha, beta, gamma): # https://docs.scipy.org/doc/scipy/reference/special.html
    """
    INPUTS: X (100 element numpy array), alpha (float; amplitude), beta (float; scale for J₀), gamma (float; phase offset)
    OUTPUTS: Y (100 element numpy array)
    DESCRIPTION: Computes the special-function model Y = alpha * J₀(beta * X + gamma), where J₀ is the cylindrical Bessel function of the first kind of order zero
    """
    return alpha * special.j0(beta * X + gamma)

X = np.linspace(0.0, 1.0, 100)
Y = j0_model(X, alpha=1.0, beta=20.0*np.pi, gamma=0.0)

print("\nProblem 5: Special-function model plot for Y = α·J₀(βX + γ); cylindrical Bessel function")

plt.plot(X, Y, '-k', label='Y = 1.0·J₀(20πX)')
plt.title("Cylindrical Bessel Function")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.tight_layout()
plt.savefig('xy_special_besselj0.pdf', bbox_inches='tight')
plt.show()

# Output:
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

# The question doesn't explicitly say to use the functions so I'm just redefining them as variables for simplicity. Don't flame me.
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

'''
7. Noisy quadratic: Y = αX^2 + βX + γ with X sampled from N(X_true, σ)
'''
# Write a new function that returns a set of y values, given an input set of x values, for the relation in 
# Equation 19, taking in the same free parameters as before. This time however, the function should return 
# a set of y values that also include an amount of normally distributed noise. In other words, the function 
# should take the input x values, and then generate a new set of x values, each one drawn from a normal distribution 
# centered on the ’true’ input x value. The FWHM of the normal distribution can be the same for each x value, but 
# should be a new input parameter to the function.

import numpy as np
import matplotlib.pyplot as plt

def quadratic_model(X, alpha, beta, gamma):
    """
    INPUTS: X (100 elementy numpy array, alpha (float; quadratic), beta (float; linear), gamma (float; constant)
    OUTPUTS: Y (100 element numpy array)
    DESCRIPTION: Computes the quadratic relation, y = a * x**2 + b * x + c, for X
    """
    return alpha * X**2 + beta * X + gamma

def quadratic_noisy_matrix(X, alpha, beta, gamma, fwhm, rng, n=100):
    """
    INPUTS: X (float), alpha (float; quadratic), beta (float; linear), gamma (float; constant), fwhm (float; width of normal noise on X), rng (numpy random generator), n (integer; number of noisy curves)
    OUTPUTS: Y_noisy (2d numpy array)
    DESCRIPTION: Makes n noisy versions of the quadratic relation y = alpha * x**2 + beta * x + gamma by adding normal (Gaussian) noise with the given FWHM to X and then computing y for each noisy X.
    """
    sigma = fwhm / 2.3548200450309493
    Xn = rng.normal(loc=X, scale=sigma, size=(n, X.size))
    return alpha * Xn**2 + beta * Xn + gamma  # shape: (n, len(X))

rng = np.random.default_rng()
X = np.linspace(0.0, 1.0, 100)
alpha, beta, gamma = 1.2, -0.8, 0.2
fwhm = 0.05

Y_true = quadratic_model(X, alpha, beta, gamma)
Y_noisy = quadratic_noisy_matrix(X, alpha, beta, gamma, fwhm, rng, n=100)

print("\nProblem 7: Noisy quadratic Y from X sampled with given FWHM")

plt.plot(X, Y_noisy.T, linewidth=0.8, alpha=0.15, color='tab:blue')
plt.plot(X, Y_true, linewidth=2.0, color='k', label='True quadratic')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.tight_layout()
plt.savefig('xy_quadratic_true_vs_noisy_vectorized.pdf', bbox_inches='tight')
plt.show()

# Output:
# Problem 7: Noisy quadratic Y from X sampled with given FWHM

'''
8. 100 element array from 0 to 10 α = 2, β = 3, γ = 5
'''
# 8. Finally: Use your new function and a 100 element x array running from 0 to
# 10 to answer the following question: For α = 2, β = 3, γ = 5, approximately
# what value of the normal distribution width is needed for the plotted relation
# to no longer resemble (as judged qualitatively) a polynomial? Hint, you do not
# need to do more than generate a fair number of plots while experimentig with
# the value of the normal distribution width.

import numpy as np
import matplotlib.pyplot as plt

def quadratic_model(X, alpha, beta, gamma):
    """
    INPUTS: X (100 elementy numpy array, alpha (float; quadratic), beta (float; linear), gamma (float; constant)
    OUTPUTS: Y (100 element numpy array)
    DESCRIPTION: Computes the quadratic relation, y = a * x**2 + b * x + c, for X
    """
    return alpha * X**2 + beta * X + gamma

def quadratic_noisy_matrix(X, alpha, beta, gamma, fwhm, rng, n=100):
    """
    INPUTS: X (float), alpha (float; quadratic), beta (float; linear), gamma (float; constant), fwhm (float; width of normal noise on X), rng (numpy random generator), n (integer; number of noisy curves)
    OUTPUTS: Y_noisy (2d numpy array)
    DESCRIPTION: Makes n noisy versions of the quadratic relation y = alpha * x**2 + beta * x + gamma by adding normal (Gaussian) noise with the given FWHM to X and then computing y for each noisy X.
    """
    sigma = fwhm / 2.3548200450309493
    Xn = rng.normal(loc=X, scale=sigma, size=(n, X.size))
    return alpha * Xn**2 + beta * Xn + gamma  # shape (n, len(X))

rng = np.random.default_rng()

X = np.linspace(0.0, 10.0, 100)
alpha, beta, gamma = 2.0, 3.0, 5.0

f1, f2, f3, f4, f5 = 0.5, 1.0, 2.0, 5.0, 20.0

Y_true = quadratic_model(X, alpha, beta, gamma)

Y_noisy_1 = quadratic_noisy_matrix(X, alpha, beta, gamma, f1, rng, n=100)
Y_noisy_2 = quadratic_noisy_matrix(X, alpha, beta, gamma, f2, rng, n=100)
Y_noisy_3 = quadratic_noisy_matrix(X, alpha, beta, gamma, f3, rng, n=100)
Y_noisy_4 = quadratic_noisy_matrix(X, alpha, beta, gamma, f4, rng, n=100)
Y_noisy_5 = quadratic_noisy_matrix(X, alpha, beta, gamma, f5, rng, n=100)

print("\nProblem 8: Adjusted normal distrubution")
# loooooooooots of plotting
fig, axes = plt.subplots(2, 3)
axes = axes.ravel()

axes[0].plot(X, Y_noisy_1.T, linewidth=0.8, alpha=0.2)
axes[0].plot(X, Y_true, 'k', linewidth=2)
axes[0].set_title(f'FWHM={f1}')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')

axes[1].plot(X, Y_noisy_2.T, linewidth=0.8, alpha=0.2)
axes[1].plot(X, Y_true, 'k', linewidth=2)
axes[1].set_title(f'FWHM={f2}')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')

axes[2].plot(X, Y_noisy_3.T, linewidth=0.8, alpha=0.2)
axes[2].plot(X, Y_true, 'k', linewidth=2)
axes[2].set_title(f'FWHM={f3}')
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')

axes[3].plot(X, Y_noisy_4.T, linewidth=0.8, alpha=0.2)
axes[3].plot(X, Y_true, 'k', linewidth=2)
axes[3].set_title(f'FWHM={f4}')
axes[3].set_xlabel('X')
axes[3].set_ylabel('Y')

axes[4].plot(X, Y_noisy_5.T, linewidth=0.8, alpha=0.2)
axes[4].plot(X, Y_true, 'k', linewidth=2)
axes[4].set_title(f'FWHM={f5}')
axes[4].set_xlabel('X')
axes[4].set_ylabel('Y')

axes[5].axis('off')

fig.suptitle('α=2, β=3, γ=5: Noisy quadratic for different FWHM', y=0.98)
plt.tight_layout()
plt.show()

# Output:
# Problem 8: Visual sweep over FWHM to judge departure from polynomial

'''
9. Predict CO SLEDs from IR components (Farrah+ 2025)
'''
# 9. This one is a ilttle different, but includes a LOT of things that astronomers,
# physicists, and data scientists do, conceptually, on a daily vasis. The file ”Some-
# Quasars.txt” contains a list of 30 or so high redshift quasars - their names,
# redshifts, a CO transition name (you will not need this), their total infrared
# luminosities, and the fraction of this total infrared luminosity that arises from
# AGN activity (the reaminder is produced by star formation).
# A brief interlude is now needed. The Carbon Monoxide molecule is quite com-
# mon in interstellar space. As you may discover in higher level courses, CO,
# when heated by starlight, has emission a set of emission lines characterized by
# two numbers, effectively an upper level and a lower level. So, the lowest fre-
# quency line is called ”1-0” as the transition is from a first excited state to a
# ground state, the second transition is ”2-1” as you are going from the second
# to the first excited states, and so on. It is common to measure a bunch of these
# lines and then make a plot called a ”SLED”, which plots the upper level for
# a line on the x axis, and the flux of that line on the y axis. The shape of the
# resulting plot can tell you all sorts of fun things about how CO is heated by
# stars in a galaxy. This, in turn, can tell you all sorts of fun things about what
# that galaxy may have done in its past, and may do in the future, but that is
# for another course.
# Back to 300L In a recent paper by Farrah et al 2022, Universe, 9, 122, which
# you can find on NASA ADS, there are a set of equations for turning a star-
# burst luminosity or an AGN luminosity into a predicted CO line luminosity, for
# transitions up to 13-12.
# • Use the Total infrared luminosities and AGN fractions in the text file to
# calculate starburst and AGN luminosities, for each quasar.
# • Using these luminosities, calculate total predicted CO line luminosities,
# for all lines for which there is an equation, for each quasar.
# • Make a single figure which shows the resulting SLEDs of every single one
# of the quasars in the text file. The figure can have multiple panels if you
# so wish...

import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('ModelFittingPart4/SomeQuasars.txt', names=True)

Lir = data['Lir']
fagn = data['fagn']

Starburst  = (1.0 - fagn) * Lir
agn_luminosity = fagn * Lir

Jup = np.array([1, 5, 6, 7, 8, 9, 10, 11, 12, 13]) # transitions

# in the order of Jup
a_arr = np.array([0.99, 0.99, 1.10, 0.98, 1.03, 1.16, 0.97, 0.65, 0.61, 0.14])
b_arr = np.array([-6.35, -4.59, -5.85, -4.36, -4.99, -6.66, -4.36, -0.58, -0.09, 5.91])
# reminder https://www.w3schools.com/python/numpy/numpy_array_slicing.asp
logLSb = np.log10(Starburst)[:, None]
all_L = 10**(logLSb * a_arr + b_arr)   # shape (Nquasars, 10)

# overwrite high-J with linear fits (J = 10,11,12,13 and columns 6,7,8,9)
s_coeff = np.array([1.16/1e5,  2.53/1e6,  2.37/1e6, -2.53/1e6])
a_coeff = np.array([9.91/1e5,  6.08/1e5,  7.66/1e5,  2.12/1e5])
c3 = np.array([-7.20e6,   3.16e6,   -3.00e6,   9.03e6])

L_high = Starburst[:, None] * s_coeff + agn_luminosity[:, None] * a_coeff + c3
all_L[:, 6:10] = L_high

print("\nProblem 9: CO SLEDs for quasars")

plt.figure()
plt.plot(Jup, all_L.T)  # one call, many SLEDs
plt.xlabel('J_up')
plt.ylabel('L_CO')
plt.yscale('log')
plt.title('Predicted CO SLEDs for all quasars')
plt.tight_layout()
plt.show()

# Output:
# Problem 9: CO SLEDs for quasars

# 18.2 Model Estimation
'''
1. ExampleData1 x & y correlation
'''
# Read the data from ”ExampleData/ExampleData1.dat” into python. Make a nicely 
# formateed scatter plot of the data. Qualitatively describe what you see - does 
# it look like the x and y data are related? If so, how?

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("ModelFittingPart4/ExampleData/ExampleData1.txt", skiprows=1)

X = data[:, 0]
Y = data[:, 1]
Y_error = data[:, 2]

print("\n18.2 - Model Estimation")
print("\nProblem 1: ExampleData1 plot; x & y correlation")
print("The scatter plot shows a very clear positive linear trend. As X increases, Y " \
      "also increases in an almost straight-line way. The points cluster tightly around a line, with " \
      "scatter that is similar in size to the error bars, so X and Y appear to be strongly correlated and " \
      "approximately follow a linear relation.")

plt.errorbar(X, Y, yerr=Y_error, fmt='o', color='k', ecolor='k', label='Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of ExampleData1')
plt.legend()
plt.savefig('ExampleData1_scatter.pdf')
plt.show()

# Output:
# Problem 1: ExampleData1 plot; x & y correlation
# The scatter plot shows a very clear positive linear trend. As X increases, Y also increases in an almost straight-line way. 
# The points cluster tightly around a line, with scatter that is similar in size to the error bars, so X and Y appear to be 
# strongly correlated and approximately follow a linear relation.

'''
2. Self-made functions
'''
# For this question you must use functions you wrote yourself. Basic numpy commands are fine, but no statistics commands, no fitting commands.
# • Using the ”x-values” array from above, generate a set of y-values using the relation y = αx + β (21) with α = 1.40 and β = 6.0. 
# Overplot these x and y values on your scatter plot, as a dashed line. Make sure the line is thick enough to easily see. 
# • Calculate both the individual residuals between the ”model” line and the data (as measured in the vertical direction), and the sum of the 
# squares of the residuals between the model line and the data. 
# • In a new, single figure, plot the residuals as a function of x. What do the sum of squares, and the individual residuals, tell you 
# about the model to data comparison? What does one tell you that the other does not? Why sum of the squares, and not just the sum of the residuals?

import numpy as np
import matplotlib.pyplot as plt

def linear_model(x, alpha, beta):
    """
    INPUTS: x (numpy array), alpha (float; slope), beta (float; intercept) 
    OUTPUTS: y_model (numpy array, same length as x)
    DESCRIPTION: Computes the linear relation, y = alpha * x + beta, for x
    """
    return alpha * x + beta

def residuals(y_data, y_model):
    """
    INPUTS: y_data (numpy array), y_model (numpy array)
    OUTPUTS: r (numpy array of residuals)
    DESCRIPTION: Computes vertical residuals, r = y_data - y_model
    """
    return y_data - y_model

def sum_of_squares(residuals):
    """
    INPUTS: residuals (numpy array of residuals)
    OUTPUTS: s (float; sum of squared residuals)
    DESCRIPTION: Returns the sum of r**2, a single number that measures
                 how far the data are from the model overall.
    """
    return np.sum(residuals**2)

data = np.loadtxt("ModelFittingPart4/ExampleData/ExampleData1.txt", skiprows=1)

X = data[:, 0]
Y = data[:, 1]
Y_error = data[:, 2]

alpha = 1.40
beta = 6.0

Y_model = linear_model(X, alpha, beta)
R = residuals(Y, Y_model)
S = sum_of_squares(R)

print("\nProblem 2: Linear model with alpha = 1.40, beta = 6.0")
print(f"Sum of squared residuals: {S:.3f}")
print("The individual residuals show, point by point, where the model is too high or too " \
"low and by how much at each x value. The sum of squared residuals gives a single number that " \
"measures the overall mismatch between model and data. We square the residuals so positive and " \
"negative values don’t cancel out, and so large errors count more than small ones.")

# first plot: data + model line
# sorting X for nice smooth line, without sorting it turns into a weird zigzag mess
# https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html
indices = np.argsort(X)
X_sorted = X[indices]
Y_model_sorted = Y_model[indices] # need to sort Y values too

plt.errorbar(X, Y, yerr=Y_error, fmt='o', color='tab:blue', ecolor='k', label='Data') # error bar handling https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html
plt.plot(X_sorted, Y_model_sorted, 'k--', linewidth=2.5, label='Model: y = 1.40x + 6.0')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('ExampleData1: data with linear model')
plt.legend()
plt.savefig('ExampleData1_with_model.pdf')
plt.show()

# second plot: residuals vs x
plt.scatter(X, R)
plt.axhline(0.0, color='k', linestyle='--') # cheeky little reminder https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axhline.html
plt.xlabel('X')
plt.ylabel('Residual (Y_data - Y_model)')
plt.title('Residuals vs X for y = 1.40x + 6.0')
plt.savefig('ExampleData1_residuals.pdf')
plt.show()

# Output:
# Problem 2: Linear model with alpha = 1.40, beta = 6.0
# Sum of squared residuals: 612.774
# The individual residuals show, point by point, where the model is too high or too low and by how much at each x value. 
# The sum of squared residuals gives a single number that measures the overall mismatch between model and data. We square 
# the residuals so positive and negative values don’t cancel out, and so large errors count more than small ones.

'''
3a. ExampleData2 x & y correlation
'''
# Repeat the same analysis as for ExampleData1, but now using ExampleData2.

import numpy as np
import matplotlib.pyplot as plt

# load data: 3 columns (X, Y, Yerr)
data2 = np.loadtxt("ModelFittingPart4/ExampleData/ExampleData2.txt", skiprows=1)

X2 = data2[:, 0]
Y2 = data2[:, 1]
Y2_error = data2[:, 2]

print("\nProblem 3a: ExampleData2 plot; x & y correlation")
print("It is evident from the plots that the X and Y datasets are not correlated, at " \
      "least not in any easily discernible way. The points fill the plot almost like random noise " \
      "and there is no clear trend.")

plt.errorbar(X2, Y2, yerr=Y2_error, fmt='o', color='k', ecolor='k', label='Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of ExampleData2')
plt.legend()
plt.savefig('ExampleData2_scatter.pdf')
plt.show()

# Output:
# Problem 3a: ExampleData2 plot; x & y correlation
# It is evident from the plots that the X and Y datasets are not correlated, at least not in any easily discernible way. 
# The points fill the plot almost like random noise and there is no clear trend.

'''
3b. Self-made functions on ExampleData2
'''
# Re-use the same functions from Problem 2

import numpy as np
import matplotlib.pyplot as plt

def linear_model(x, alpha, beta):
    """
    INPUTS: x (numpy array), alpha (float; slope), beta (float; intercept) 
    OUTPUTS: y_model (numpy array, same length as x)
    DESCRIPTION: Computes the linear relation, y = alpha * x + beta, for x
    """
    return alpha * x + beta

def residuals(y_data, y_model):
    """
    INPUTS: y_data (numpy array), y_model (numpy array)
    OUTPUTS: r (numpy array of residuals)
    DESCRIPTION: Computes vertical residuals, r = y_data - y_model
    """
    return y_data - y_model

def sum_of_squares(residuals_array):
    """
    INPUTS: residuals_array (numpy array of residuals)
    OUTPUTS: s (float; sum of squared residuals)
    DESCRIPTION: Returns the sum of r**2, a single number that measures
                 how far the data are from the model overall.
    """
    return np.sum(residuals_array**2)

data2 = np.loadtxt("ModelFittingPart4/ExampleData/ExampleData2.txt", skiprows=1)

X2 = data2[:, 0]
Y2 = data2[:, 1]
Y2_error = data2[:, 2]

alpha = 1.40
beta = 6.0

Y2_model = linear_model(X2, alpha, beta)
R2 = residuals(Y2, Y2_model)
S2 = sum_of_squares(R2)

print("\nProblem 3b: Linear model on ExampleData2 with alpha = 1.40, beta = 6.0")
print(f"Sum of squared residuals (ExampleData2): {S2:.3f}")
print("Compared to ExampleData1, a larger sum of squared residuals means the model " \
      "does not follow the data as closely")

# first plot: data + model line
indices2 = np.argsort(X2)
X2_sorted = X2[indices2]
Y2_model_sorted = Y2_model[indices2]

plt.errorbar(X2, Y2, yerr=Y2_error, fmt='o', color='tab:blue', ecolor='k', label='Data')
plt.plot(X2_sorted, Y2_model_sorted, 'k--', linewidth=2.5, label='Model: y = 1.40x + 6.0')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('ExampleData2: data with linear model')
plt.legend()
plt.savefig('ExampleData2_with_model.pdf')
plt.show()

# second plot: residuals vs X
plt.scatter(X2, R2)
plt.axhline(0.0, color='k', linestyle='--')
plt.xlabel('X')
plt.ylabel('Residual (Y_data - Y_model)')
plt.title('Residuals vs X for ExampleData2, y = 1.40x + 6.0')
plt.savefig('ExampleData2_residuals.pdf')
plt.show()

# Output:
# Problem 3b: Linear model on ExampleData2 with alpha = 1.40, beta = 6.0
# Sum of squared residuals (ExampleData2): 36456372.641
# Compared to ExampleData1, a larger sum of squared residuals means the model does not follow the 
# data as closely. The individual residuals show how the model over or under-predicts Y at each X, 
# and the sum of squares gives a single overall measure of the mismatch. We square the residuals so 
# positive and negative values do not cancel out.

# 18.3 - Model Fitting
'''
1.  Using any python approach of your choice that explicitly only performs linear regression
'''
import numpy as np
import matplotlib.pyplot as plt

print("\n18.3 - Model fitting")
print("\nProblem1 : Python approach to perform linear regression")

def linear_fit(x, y):
    """
    Description:
        Calculates best-fit slope and intercept for linear model y = a*x + b 
        using the unweighted least-squares method. Also computes R^2 value
        to quantify the "goodness" of fit.

    Inputs:
        x (1-D numpy array of x-values)
        y (1-D numpy arrayv of y-values corresponding to x)

    Outputs:
        a (Best-fit slope of the line)
        b (Best-fit intercept of the line)
        R2 (to measure goodness of fit)
    """
    xm = np.mean(x)        # mean of x values
    ym = np.mean(y)        # mean of y values

    # compute sums needed for least-squares formulas
    Sxx = np.sum((x - xm)**2)
    Sxy = np.sum((x - xm) * (y - ym))

    a = Sxy / Sxx          # best-fit slope
    b = ym - a * xm        # best-fit intercept

    # predicted y-values and R^2
    yfit = a * x + b
    ss_res = np.sum((y - yfit)**2)   # residual sum of squares
    ss_tot = np.sum((y - ym)**2)     # total sum of squares
    R2 = 1 - ss_res / ss_tot


    return a, b, R2

def go(filename):
    """
    Description:
        Loads x, y, and y-error data from txt file, performs linear
        least-squares fit using linear_fit() function, prints best-fit
        parameters, and generates plot showing data with error bars and best-fit line.

    Inputs:
        filename (name of the text file containing a header row followed by three numeric columns)

    Outputs:
        None (prints fit results to screen and produces a plot, but doesn't return any values.
    """
    # skip header row, unpack 3 columns (X, Y, YErrors)
    x, y, yerr = np.loadtxt(filename, skiprows=1, unpack=True)

    a, b, R2 = linear_fit(x, y)

    print(f"Results for {filename}:")
    print(f"  alpha (slope)     = {a}")
    print(f"  beta  (intercept) = {b}")
    print(f"  R^2               = {R2}")
    print()

    #plot data & fit
    xs = np.linspace(x.min(), x.max(), 200)
    ys = a * xs + b

    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt='o', label="data")  # uses YValuesErrors
    plt.plot(xs, ys, label="best-fit line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Linear fit for {filename}")
    plt.legend()
    plt.tight_layout()

go("ModelFittingPart4/ExampleData/ExampleData1.txt")
go("ModelFittingPart4/ExampleData/ExampleData2.txt")
plt.show()

# For my fitting approach, I performed linear regression using the basic least-squares method by calculating the slope
# and intercept directly from the data. I also calculated the R^2 value to basically measure how well the line fits the data
# and also plotted the data with their corresponding best-fit lines. 

'''
2. Comment on formal goodness of fit & whether or not you think the model is in fact a good predictor of the data.
'''
print("\nProblem 2: Commenting on the formal goodness of the fit")
print("For ExampleData1, the formal goodness of fit is very high: 0.994. This indicates that the linear model explains" \
"almost all of the variation in the data. Also, observing visually, the data fall super close to a straight line, which" \
"the best-fit line matches well. So, both the formalaR^2 value and the visual pattern agree that a linear model is a" \
"really good predictor of the data.")

print("\nFor ExampleData2, the formal goodness of fit is very low: 2.45*10^-7 (basically 0 lol). This means that the linear model " \
"explains none of the variation in the dataset. Visually, this is confirmed because there is no linear trend, and the best-fit" \
"line is almost flat. So, both the formal R^2 value and the visual pattern agree that a linear model is not a good predictor of the data.")

'''
3. Read in the file ”Basics/ExampleData3.dat”. Make a visually informative plot of these data
'''
import numpy as np
import matplotlib.pyplot as plt

print("\nProblem 3: Visually informative plot of ExampleData3")

# load data (skip header, unpack X, Y, YErrors)
x, y, yerr = np.loadtxt("ModelFittingPart4/ExampleData/ExampleData3.txt", skiprows=1, unpack=True)

# compute best-fit slope, intercept, and R^2 using previous linear_fit function
a, b, R2 = linear_fit(x, y)

# generate x-values and corresponding model y-values for plotting the fit line
xs = np.linspace(x.min(), x.max(), 300)
ys = a * xs + b

# plot
plt.figure(figsize=(7,5))
plt.errorbar(x, y, yerr=yerr, fmt='o', label="data")
plt.plot(xs, ys, color="orange", linewidth=2, label="linear fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear fit for ExampleData3.txt")
plt.legend()
plt.tight_layout()
plt.show()

print("Results for ExampleData3.txt:")
print("  alpha (slope)     =", a)
print("  beta  (intercept) =", b)
print("  R^2               =", R2)
print("The data in ExampleData3.txt show a nonlinear trend, even though they increase overall. The linear fits" \
"the general upward slope and gives a fairly high R^2 value, but the plot shows a clear curve that the best-fit" \
"line doesn't follow. So, the model gives an approximation but isn't a very accurate predictor.")

'''
4. This seems to be problematic, since the formalism of linear regression cannot be used to fit such a model. Briefly comment on why this is. If you take the logarithm of Equation 23, what relation emerges?
'''

print("\nProblem 4: Comments on the problematic nature of this thing")
print("A power-law model is not a straight-line equation, so it doesn't match the form that linear regression is designed to" \
"handle. Linear regression only works when the model can be written as a constant times a function plus another constant," \
"but a power-law has the variable raised to a power, which makes it nonlinear. Because of this, we can't directly plug it" \
"into the usual least-squares formulas. The equation needs to be transformed before linear regression can be used.")
print("Starting with y = αx^β + y, if y=0 (or just assume its small enough to ignore), we get: y = ax^B. Taking the log of" \
"both sides: lny = lnα + βlnx. This is linear, a straight line and can be fit using standard linear regression but on" \
"log-transformed data.")

'''
5. Take the logarithm of the x and y values, fit a linear relation to these data, calculate the best-fit values of α, β, γ. Make a plot of the data, the linear fit, and the log-linear fit.
'''

print("\nProblem 5: Logarithm of the x and y values with linear fit and best fit values")

# load data (3 columns: x, y, yerr)
x, y, yerr = np.loadtxt("ModelFittingPart4/ExampleData/ExampleData3.txt", skiprows=1, unpack=True)

# take logs (only valid if y > 0)
logx = np.log(x)
logy = np.log(y)

#fit log-linear model: ln(y) = ln(a) + B ln(x)
a_log, b_log, R2_log = linear_fit(logx, logy)   # slope = B, intercept = ln(a)

beta = a_log                     # slope
alpha = np.exp(b_log)            # intercept back-transformed
gamma = 0                        # must be assumed for log transform

print("Power-law fit parameters:")
print("   α =", alpha)
print("   β =", beta)
print("   γ =", gamma)
print("   R^2 (log-linear) =", R2_log)

# create prediction curves
xs = np.linspace(x.min(), x.max(), 300)

# linear straight-line fit
a_lin, b_lin, _ = linear_fit(x, y)
ys_lin = a_lin * xs + b_lin

# power-law curve y = a x^B
ys_power = alpha * xs**beta

# plot
plt.figure(figsize=(8,6))
plt.errorbar(x, y, yerr=yerr, fmt='o', label="data")

plt.plot(xs, ys_lin, label="linear fit")
plt.plot(xs, ys_power, label="power-law log-linear fit")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear vs Log-Linear (Power-Law) Fits for ExampleData3.txt")
plt.legend()
plt.tight_layout()
plt.show()

'''
6. Find one of these that is NOT scipy.odr, and fit the data in ExampleData3.dat with it. Compare your best fit parameters with the log-linear fit results from above. Plt this fitted relation on the same figure as the linear and log-linear fits.
'''
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
# https://en.wikipedia.org/wiki/Non-linear_least_squares
# https://hernandis.me/2020/04/05/three-examples-of-nonlinear-least-squares-fitting-in-python-with-scipy.html

print("\nProblem 6: New fit on ExampleData3")

from scipy.optimize import curve_fit

# model for the nonlinear fit (a full power-law w additive constant gamma)
def power_law_model(x, alpha, beta, gamma):
    """
    Description:
        Power-law model with an additive constant: y = α x^β + γ.

    Inputs:
        x (Independent variable)
        alpha (Amplitude of the power-law)
        beta (Exponent of the power-law)
        gamma (Additive constant offset)

    Outputs:
        y (Model values computed as α x^β + γ for the given x)
    """
    return alpha * x**beta + gamma

# load data from file
x, y, yerr = np.loadtxt("ModelFittingPart4/ExampleData/ExampleData3.txt", skiprows=1, unpack=True)

# perform a linear fit directly to x and y for comparison
a_lin, b_lin, R2_lin = linear_fit(x, y)
xs = np.linspace(x.min(), x.max(), 300)
ys_lin = a_lin * xs + b_lin

# transform x and y into log space and fit straight line to estimate alpha and beta
logx = np.log(x)
logy = np.log(y)
beta_log, ln_alpha_log, R2_log = linear_fit(logx, logy)
alpha_log = np.exp(ln_alpha_log)
gamma_log = 0.0
ys_power_log = alpha_log * xs**beta_log

# use nonlinear least-squares routine to fit full power-law model to original data
# initial parameter guesses come from the log-linear fit above
p0 = [alpha_log, beta_log, gamma_log]

popt, pcov = curve_fit(
    power_law_model,
    x,
    y,
    p0=p0,
    sigma=yerr,
    absolute_sigma=True
)

# extract nonlinear best-fit values and use to compute model curve
alpha_nl, beta_nl, gamma_nl = popt
ys_power_nl = power_law_model(xs, alpha_nl, beta_nl, gamma_nl)

print("Log-linear power-law (γ = 0):")
print("   α_log =", alpha_log)
print("   β_log =", beta_log)
print("   γ_log =", gamma_log)
print()
print("Nonlinear power-law fit:")
print("   α_nl =", alpha_nl)
print("   β_nl =", beta_nl)
print("   γ_nl =", gamma_nl)


# plot raw data and all three fitted curves
plt.figure(figsize=(8, 6))
plt.errorbar(x, y, yerr=yerr, fmt='o', label="data")
plt.plot(xs, ys_lin, label="linear fit")
plt.plot(xs, ys_power_log, label="log-linear power-law fit (γ = 0)")
plt.plot(xs, ys_power_nl, label="nonlinear power-law fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("ExampleData3: Linear vs Power-Law Fits")
plt.legend()
plt.tight_layout()
plt.show()

'''
7. Read in the file ”Basics/ExampleData4.dat”. Without plotting the data, fit a linear model to the data and inspect the goodness of fit it produces. Do the fit results alone suggest the linear model is a good one?
'''

print ("\nProblem 7: Linear fit on ExampleData4 without plotting")
# load data
x4, y4, yerr4 = np.loadtxt("ModelFittingPart4/ExampleData/ExampleData4.txt", skiprows=1, unpack=True)

# fit linear model
a4, b4, R2_4 = linear_fit(x4, y4)

print("Linear fit for ExampleData4.txt:")
print("  slope (a)      =", a4)
print("  intercept (b)  =", b4)
print("  R^2            =", R2_4)
print("Based on the results, the linear model seems to work very well. It results in an R^2 value of 0.998, which is extremly high. A slope" \
"of about 1.21 and a small intercept also show a stable fit. So yes, from the fit results alone, they suggest that the linear model"
"is a good one.")

'''
8. Now plot the data in ExampleData4.dat. By looking at the data, is a linear model appropriate? Give answers under the following two scenarios.
'''

print("\nProblem 8: ExampleData4 plot and analysis")
x, y, yerr = np.loadtxt("ModelFittingPart4/ExampleData/ExampleData4.txt", skiprows=1, unpack=True)

plt.figure(figsize=(7,5))
plt.scatter(x, y, label="data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("ExampleData4.txt")
plt.legend()
plt.tight_layout()
plt.show()

print("Scenario 1: It seems that my lab assistant chose really narrow ranges of x, which is what likely produced the two" \
"tight clusters in the far corners. And so because the data cover only two tiny regions of the x-range, the linear" \
"regression can't actually see what is going on in those regions. The high R^2 value is misleading since it" \
"essentially is just connecting the two clusters with a line. I think I could not reliably say that the underlying" \
"relationship is linear since the sampling is too limited. I would say to my assistant 'Next time, please sample x" \
"values across the full range, not just two small clusters. Otherwise we can't tell whether it behaves linearly or" \
"not'.'")
print("Scenario 2: Here the objects are naturally distributed, and both x and y are measured quantities. This makes the" \
"two-cluster pattern real. Still, the high R^2 value is misleading. I can easily conclude in this scenario that a" \
"linear model is not good for this data. A single straight loine connected two separate clouds of points does not" \
"represent a real physical rekationship. Instead, I'd probably suggest modeling them separately.")

'''
9. In one of the cases above, a linear model is not appropriate. What sort of model might be?
'''
# https://en.wikipedia.org/wiki/Model-based_clustering?

print("\nProblem 9: Determining an appropriate model")
print("A clustering model like a two-component model or Gaussian mixture. This treats the points as belonging to two " \
"different populations instead of trying to force one function to fit everything.")

# 18.4 A Cluster of stars
'''
1.  Color–magnitude diagram for one star cluster
'''
# Pick a star cluster. Read in the ”raw” data file for that cluster. Then, make a
# scatter plot, using all default parameters, where the x axis is a magnitude
# (of your choice), and the y axis is the difference between two magnitudes (again,
# two of your choice). This type of plot is called a color-magnitude diagram, and
# is widely used in astronomy as a diagnostic tool. Save your plot as a png file.

import numpy as np
import matplotlib.pyplot as plt

def colors(mag1, mag2):
    """
    INPUTS: mag1 (numpy array), mag2 (numpy array)
    OUTPUTS: color (numpy array, same length as mag1)
    DESCRIPTION: Computes a color index, (mag1 - mag2).
    """
    return mag1 - mag2

data = np.genfromtxt("ModelFittingPart4/StarClusters/M67/M67_raw.csv", delimiter=",", names=True, dtype=None, encoding=None) # csv help https://note.nkmk.me/en/python-numpy-loadtxt-genfromtxt-savetxt/

# magnitudes from header
g_mag = data['gMeanPSFMag']
r_mag = data['rMeanPSFMag']

# color g - r
g_mag_minus_r_mag = colors(g_mag, r_mag)

print("\n18.4 - A cluster of stars")
print("Problem 1: Color-magnitude diagram for M67")

plt.scatter(g_mag, g_mag_minus_r_mag, s=5)
plt.xlabel('gMeanPSFMag (mag)')
plt.ylabel('gMeanPSFMag - rMeanPSFMag (mag)')
plt.title('M67 Color-Magnitude Diagram')

plt.tight_layout()
plt.savefig('M67_color_magnitude_diagram.png', bbox_inches='tight')
plt.show()

# Output:
# 18.4 - A cluster of stars
# Problem 1: Color-magnitude diagram for M67

'''
2.  Log-scaled color–magnitude diagram with style tweaks and partner feedback
'''
# For this question you will, at the end, need to work with your partner directly.
# The above plot does not look particularly edifying. Let’s make it more infor-
# mative:
# • Make the scaling of both axes logarithmic
# • Pick a bright color from the tableau palette and plot the points in this
# color.
# • Play with the size and transparency (alpha) parameters on the scatter plot
# command so that the result looks more informative than the original; in
# particular, look for a ”dense” region of stars in your plot.
# • Plotted this way, it should be clear that many of the stars in the cluster fall
# along a very narrow ”track” in color-magnitude space, which is consistent
# with these stars all being of a comparable age.
# • Show your plot to your partner/look at your partners plot, and recieve/-
# give feedback on how to improve it. Implement the changes your partner
# suggests, if you agree with them.
# • Using the ”text” command in pyplot, add ”Feedback from yourpartners
# surname” to your plot, at a suitable location and using a tableau color
# different to the one used to plot the data. Save your final plot as a pdf file
# with a resolution of 600dpi (this is the minimum for a professional journal,
# but in most other cases is overkill).

import numpy as np
import matplotlib.pyplot as plt

def colors(mag1, mag2):
    """
    INPUTS: mag1 (numpy array), mag2 (numpy array)
    OUTPUTS: color (numpy array, same length as mag1)
    DESCRIPTION: Computes a color index, (mag1 - mag2).
    """
    return mag1 - mag2

data = np.genfromtxt("ModelFittingPart4/StarClusters/M67/M67_raw.csv", delimiter=",", names=True, dtype=None, encoding=None)

g_mag = data['gMeanPSFMag']
r_mag = data['rMeanPSFMag']
g_mag_minus_r_mag = colors(g_mag, r_mag)

print("\nProblem 2: Log-scaled color-magnitude diagram for M67")

plt.scatter(g_mag, g_mag_minus_r_mag, s=10, alpha=0.3, color='tab:orange')
plt.xlabel('gMeanPSFMag (mag)')
plt.ylabel('gMeanPSFMag - rMeanPSFMag (mag)')
plt.title('M67 Color-Magnitude Diagram (log axes)')

# adding text on plot bc noob https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html
ax = plt.gca()
ax.text(0.05, 0.05, "Feedback from Silvia:\n- brighter color\n- larger points\n- alpha=0.3", transform=ax.transAxes, color='tab:green')

plt.tight_layout()
plt.savefig('M67_color_magnitude_diagram_feedback.pdf', bbox_inches='tight')
plt.show()

# Output:
# Problem 2: Log-scaled color-magnitude diagram for M67

'''
3. Isochrone overlay and cluster age estimate
'''
# The associated isochrone files contain theoretical predictions for the magnitudes
# of a star at a specific age (given in the filename). Read in these isochrones and
# use them to qualitatively estimate the age of your cluster. You can do this
# by overplotting each isochrone in turn (using the columns appropriate to your
# choice of magnitude and color), and then scaling the ”magnitude” axis of your
# isochrone until it is as close as possible to the bulk of stars in the closter. Repeat
# this for each isochrone and visually judge which one comes closest.
# Add your best ”fit” isochrone to your plot. Use a tableau color different to
# those for the other plot elements, put it ”on top” of the data (using zorder)
# with a line thickness of around 5. Give the line some transparency too. Add
# its age as a legend to the plot

import numpy as np
import matplotlib.pyplot as plt

def colors(mag1, mag2):
    """
    INPUTS: mag1 (numpy array), mag2 (numpy array)
    OUTPUTS: color (numpy array, same length as mag1)
    DESCRIPTION: Computes a color index, (mag1 - mag2).
    """
    return mag1 - mag2

# cluster data (same as before)
data = np.genfromtxt("ModelFittingPart4/StarClusters/M67/M67_raw.csv", delimiter=",", names=True, dtype=None, encoding=None)

g_mag = data['gMeanPSFMag']
r_mag = data['rMeanPSFMag']
g_mag_minus_r_mag = colors(g_mag, r_mag)

isochrone = np.loadtxt("ModelFittingPart4/StarClusters/M67/M67_4GY.txt", comments="#")

# EEP M/Mo LogTeff LogG LogL/Lo open gp1 rp1 ip1 zp1 yp1 wp1
gp1 = isochrone[:, 6]
rp1 = isochrone[:, 7]
gp1_minus_rp1 = colors(gp1, rp1)

magnitude_offset = 9.7  # eyeballed it, not really sure

gp1_shifted = gp1 + magnitude_offset

print("\nProblem 3: Isochrone overlay for M67")
print(f"Using M67_4GY.txt with a magnitude shift of {magnitude_offset:.2f} mag.")
print("By eye, this ~4 Gyr isochrone matches the main sequence and turnoff best, so the cluster is a few Gyr.")

plt.scatter(g_mag, g_mag_minus_r_mag, s=10, alpha=0.3, color='tab:orange', label='M67 stars') # og cluster plot

plt.plot(gp1_shifted, gp1_minus_rp1, color='tab:purple', linewidth=5.0, alpha=0.5, zorder=5, label='Isochrone: 4 Gyr') # isochrone on top

plt.xlabel('gMeanPSFMag (mag)')
plt.ylabel('gMeanPSFMag - rMeanPSFMag (mag)')
plt.title('M67 Color-Magnitude Diagram with Isochrone')

plt.gca().invert_xaxis()  # keeping convention because why not
plt.legend()
plt.tight_layout()
plt.savefig('M67_CMD_best_isochrone.pdf', bbox_inches='tight')
plt.show()

# Output:
# Problem 3: Isochrone overlay for M67
# Using M67_4GY.txt with a magnitude shift of 9.70 mag.
# By eye, this ~4 Gyr isochrone matches the main sequence and turnoff best, so the cluster is a few Gyr.

# 18.5 Fitting a standard stellar spectrum
'''
1. Write a function that takes as input a single temperature and an array of wavelengths, and outputs a blackbody curve. Use SI units.
'''

print("\n18.5 Fitting a standard stellar spectrum")
print("Problem 1: Blackbody output of temperature and wavelengths\n(there's nothing to see here)")

# https://en.wikipedia.org/wiki/Planck%27s_law

import numpy as np

# physical constants in SI units
h = 6.62607015e-34      # planck's constant [J s]
c = 2.99792458e8        # speed of light [m/s]
k_B = 1.380649e-23        # Boltzmann constant [J/K]

def blackbody(T, lam):
    """
    Description:
        Calculates the blackbody curve for a given temperature.
        You give it one temperature and an array of wavelengths,
        and it returns the brightness of a perfect blackbody at
        each wavelength. All units are in SI.

    Inputs:
        T (temperature in Kelvin)
        lam (wavelengths in meters)

    Outputs:
        B_lambda (blackbody intensity at each wavelength (W·m⁻²·sr⁻¹·m⁻¹))
    """
    lam = np.asarray(lam)  # ensure array
    a = 2.0 * h * c**2 / lam**5  #planck's law in wv form
    b = h * c / (lam * k_B * T)  #^
    return a / (np.exp(b) - 1.0)


'''
2. Use this function to generate seven pure blackbody curves, for temperatures corresponding to the following stars: O2, B1, A4, F3, G0, K5, M6.
'''
# https://sites.uni.edu/morgans/astro/course/Notes/section2/spectraltemps.html

import numpy as np
import matplotlib.pyplot as plt

print("\nProblem 2: Generating pure blackbody curves")

spectral_types = {
    "O2": 45000,
    "B1": 23000,
    "A4": 8480,
    "F3": 6850,
    "G0": 6050,
    "K5": 4400,
    "M6": 3100,
}

# wv range
lam_m = np.logspace(-10, -2, 6000)
lam_um = lam_m * 1e6

plt.figure(figsize=(12, 8))

for sp, T in spectral_types.items():
    B = blackbody(T, lam_m)
    B_norm = B / np.max(B)
    plt.plot(lam_um, B_norm, label=f"{sp} ({T} K)", linewidth=2)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Wavelength [µm]")
plt.ylabel("Normalized $B_\\lambda$ (arbitrary units)")
plt.title("Blackbody Curves for Different Spectral Types")

# view window
plt.xlim(1e-4, 1e4)
plt.ylim(1e-30, 1e2)

plt.legend(loc="upper right", fontsize=10)
plt.grid(alpha=0.25, which="both")
plt.tight_layout()
plt.show()


# print table
print("Spectral Type   Temperature (K)")
print("-------------------------------")
for sp, T in spectral_types.items():
    print(f"{sp:10s}   {T:>8d}")


'''
3. Pick one of these spectra, read it in, and make a plot of the spectrum.
'''
#with open("uka2v.dat", "r") as f:
#    for i in range(20):
#        print(f.readline().rstrip())

import numpy as np
import matplotlib.pyplot as plt

print("\nProblem 3: Plotting uka2v.dat")

# load columns from file
data = np.loadtxt("ModelFittingPart4/StandardSpectra/uka2v.dat", comments="#")

# extract wavelength (col 0) and flux (col 1 = 'lk')
wave_A = data[:, 0]
flux_norm = data[:, 1]   # you can change to 2 or 3 if desired

# plot
plt.figure(figsize=(10, 6))
plt.plot(wave_A, flux_norm, color="black", linewidth=1)

plt.xlabel(r"Wavelength [$\AA$]")
plt.ylabel("Normalized Intensity")
plt.title("Standard Spectrum: uka2v")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


'''
4. Generate and overplot a blackbody curve with the same temperature (as in- ferred by spectral type) as your chosen star, adjusting the normalization as necessary.
'''

print("\nProblem 4: Overplotting blackbody curve")
T = 9000  # temp inferred from spectral type (A2)

# convert wavelength Å → m
lam_m = wave_A * 1e-10

# compute blackbody
bb = blackbody(T, lam_m)

# scale blackbody so its peak matches the observed spectrum (for shape comparison)
scale = flux_norm.max() / bb.max()
bb_scaled = bb * scale

#plot observed spectrum + blackbody
plt.figure(figsize=(10,6))
plt.plot(wave_A, flux_norm, label="uka2v spectrum", color="black")
plt.plot(wave_A, bb_scaled, label=f"Blackbody ({T} K)", color="red")

plt.xlabel("Wavelength [Å]")
plt.ylabel("Flux (normalized)")
plt.title("uka2v Spectrum with Blackbody Overplotted")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("The observed spectrum and the blackbody curve have the same overall shape, but the real spectrum shows a lot of " \
"absorption features that the blackbody does not. At longer wavelengths, where the spectrum has less lines, the" \
"continuum follows the blackbody a lot closer. Overall, a good shape match I'd say, though the peak is slightly off.")


'''
5. Using a python approach of your choice (which you should describe), fit a blackbody function to your chosen stellar spectrum, and derive the best fit temperature and uncertainty.
'''
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

print("\nProblem 5: Fitting blackbody function to chosen stellar spectrum")

# blackbody model for fitting
# F(λ) = A * B_λ(T)
# curve_fit will vary T and A to minimize squared residuals
def bb_model(lam_A, T, A):
    """
    Description:
        Takes wavelengths in Angstroms, converts them to meters,
        evaluates the blackbody function at temperature T,
        and scales the result by A.

    Inputs:
        lam_A (array of wavelengths)
        T (temp in Kelvin)
        A (amplitude scaling factor)

    Outputs:
        array of scaled blackbody flux values
    """
    lam_m = lam_A * 1e-10        # convert Å → meters for the blackbody function
    return A * blackbody(T, lam_m)

# choose continuum points
# avoid λ < 3500 Å and avoid λ > 9000 Å
mask = (wave_A > 3500) & (wave_A < 9000) & (flux_norm > 0.6)  # keep only flux > 0.6 to avoid deep absorption lines

# wavelength and flux arrays we will fit to
lam_cont  = wave_A[mask]
flux_cont = flux_norm[mask]

# fit the blackbody model
# curve_fit performs non-linear least squares, return best-fit parameters and covariance matrix
popt, pcov = curve_fit(bb_model, lam_cont, flux_cont, p0=[7200, 1.0])

T_fit, A_fit = popt                 # extract fitted temperature + scale
T_err = np.sqrt(np.diag(pcov))[0]    # 1σ uncertainty from covariance

print("Best-fit temperature:", T_fit, "K")
print("Uncertainty:", T_err, "K")

# compute best-fit curve
bb_fit_raw = bb_model(wave_A, T_fit, A_fit)

# rescale the fitted curve for plotting only
# the continuum envelope peak (like the 7200 K reference curve)
scale_fit = flux_norm.max() / bb_model(wave_A, T_fit, 1).max()
bb_fit = bb_model(wave_A, T_fit, scale_fit)

T_spectral = 9000  # A2 star
scale_spec = flux_norm.max() / bb_model(wave_A, T_spectral, 1).max()
bb_spec = bb_model(wave_A, T_spectral, scale_spec)

# plot data + best-fit + comparison curve
plt.figure(figsize=(10, 6))
plt.plot(wave_A, flux_norm, color="black", label="uka2v spectrum")

# best-fit temperature (red curve)
plt.plot(wave_A, bb_fit, color="red",
         label=f"Best-fit BB: {T_fit:.0f} K")

#spectral type blackbody curve
plt.plot(wave_A, bb_spec, color="orange", linestyle="--",
         label=f"BB from spectral type (~{T_spectral} K)")

plt.xlabel("Wavelength [Å]")
plt.ylabel("Normalized Flux")
plt.title("Blackbody Fit to uka2v Continuum")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


print("The spectral type blackbody (A2=9000K) peaks too far and kinda sits above the observed continuum," \
"while the fitted blackbody matches the continuum a bit better. This suggests that the quoted spectral type is roughly correct" \
"but too hot.")
print("I fit a model F(lambda) = AB(T) to continuum points using curve_fit. This does non-linear least-squares minimization. The" \
"algorithm adjusts T to A. This minimizes the squared difference between the model and the selected data. The best fit temp comes" \
"from the returned parameters and the uncertainty is taken from the square root of the covariance matrix.")


'''
6. Repeat the above analysis steps for five further standard spectra, of your choice.
'''
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

print("\nProblem 6: Repeating analysis for five further standard spectra")
def fit_and_plot_spectrum(filename, T_spectral):
    """
    Description:
        Loads a stellar spectrum, selects continuum regions,
        fits a blackbody curve to those regions, and overplots: the 
        observed spectrum, the best-fit blackbody, a blackbody using 
        the catalog (spectral-type) temperature

    Inputs:
        filename (name of the .dat file containing wavelength + flux)
        T_spectral (expected temperature based on spectral type)

    Outputs:
        T_fit (best-fit temperature from curve_fit)
        T_err (1-sigma uncertainty on T_fit)
    """

    # load wavelength (Å) and normalized flux
    data = np.loadtxt(filename, comments="#")
    wave_A = data[:, 0]
    flux_norm = data[:, 1]

    # continuum mask: avoid blue/red edges + avoid absorption dips
    mask = (wave_A > 3500) & (wave_A < 9000) & (flux_norm > 0.6)
    lam_cont  = wave_A[mask]        # continuum wavelengths
    flux_cont = flux_norm[mask]     # continuum flux values

    # fit F(λ) = A * B_λ(T) using nonlinear least-squares
    popt, pcov = curve_fit(
        bb_model,             # model function
        lam_cont,              # wavelengths used in fit
        flux_cont,             # corresponding fluxes
        p0=[T_spectral, 1.0]   # initial guess to help convergence
    )

    T_fit, A_fit = popt   # best-fit temp + amp
    T_err = np.sqrt(np.diag(pcov))[0]   # uncertainty on temp

    print(f"\nFile: {filename}")
    print("Best-fit temperature:", T_fit, "K")
    print("Uncertainty:", T_err, "K")

    # scale best-fit BB so its peak matches the observed peak (makes comparison clearer)
    scale_fit = flux_norm.max() / bb_model(wave_A, T_fit, 1).max()
    bb_fit = bb_model(wave_A, T_fit, scale_fit)

    # scale blackbody predicted by the spectral type
    scale_spec = flux_norm.max() / bb_model(wave_A, T_spectral, 1).max()
    bb_spec = bb_model(wave_A, T_spectral, scale_spec)

    # plotting
    plt.figure(figsize=(10, 6))
    plt.plot(wave_A, flux_norm, color="black", label=f"{filename} spectrum") # observed spectrum
    plt.plot(wave_A, bb_fit, color="red", label=f"Best-fit BB: {T_fit:.0f} K") # fitted curve
    plt.plot(wave_A, bb_spec, color="orange", linestyle="--", label=f"BB from spectral type (~{T_spectral} K)") # spectral-type curve

    plt.xlabel("Wavelength [Å]")
    plt.ylabel("Normalized Flux")
    plt.title(f"Blackbody Fit for {filename}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return T_fit, T_err


# mapping between file names and spectral-type temperatures (ish)
spectral_temps = {
    "ModelFittingPart4/StandardSpectra/ukb8v.dat": 12000,
    "ModelFittingPart4/StandardSpectra/ukb9v.dat": 10500,
    "ModelFittingPart4/StandardSpectra/ukf0v.dat": 7300,
    "ModelFittingPart4/StandardSpectra/ukf5v.dat": 6500,
    "ModelFittingPart4/StandardSpectra/ukm2v.dat": 3500,
}

# run the analysis for 5 stars
for fname, Tspec in spectral_temps.items():
    fit_and_plot_spectrum(fname, Tspec)

# 18.6 - Fitting unknown stellar spectra
'''
1. Choose one of these spectra and read it in. Then find the two that are closest to it. Generate a plot of the unknown with these two standards overplotted.
'''
import numpy as np
import matplotlib.pyplot as plt

print("\n18.6 - Fitting unknown stellar spectra")
print("Problem 1: Choosing a spectra and finding two neighbors; plotting")

def load_unknown(filename):
    """
    Load an unknown spectrum (6-column format).

    Inputs:
        filename (string): e.g. "unknown1.dat"

    Outputs:
        wave_A : array of wavelengths (Å)
        flux   : array of flux values (we use column 1)
    """
    data = np.loadtxt(filename, comments="#")
    wave_A = data[:, 0]   # wavelength
    flux   = data[:, 1]   # use column 1 as the flux
    return wave_A, flux

def load_standard(filename):
    """
    Load a standard spectrum (2-column format).

    Inputs:
        filename (string): e.g. "uka7v.dat"

    Outputs:
        wave_A : array of wavelengths (Å)
        flux   : array of normalized flux values
    """
    data = np.loadtxt(filename, comments="#")
    wave_A = data[:, 0]   # extract wavelength column (in Angstroms)
    flux   = data[:, 1]   # extract corresponding flux values
    return wave_A, flux

# load the wavelength and flux arrays for first unknown spectrum
unk_wave1, unk_flux1 = load_unknown("ModelFittingPart4/UnknownSpectra/unknown1")

plt.figure(figsize=(10,6))
plt.plot(unk_wave1, unk_flux1 / unk_flux1.max(), color="black", label="Unknown 1")
plt.xlabel("Wavelength [Å]")
plt.ylabel("Normalized Flux")
plt.title("Unknown1 Spectrum")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

std_wave1, std_flux1 = load_standard("ModelFittingPart4/StandardSpectra/ukb8v.dat")  # change to the standard you want

# normalize both so we only compare shape, not absolute flux
unk_norm1  = unk_flux1 / unk_flux1.max()
std_norm1  = std_flux1 / std_flux1.max()

plt.figure(figsize=(10,6))
plt.plot(unk_wave1, unk_norm1, color="black", label="Unknown 1")
plt.plot(std_wave1, std_norm1, color="red", label="Standard: uka7v.dat")

plt.xlabel("Wavelength [Å]")
plt.ylabel("Normalized Flux")
plt.title("Unknown1 vs Standard")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# unknown (we already loaded unk_wave, unk_flux)
unk_norm1 = unk_flux1 / unk_flux1.max()

# load the two closest standards
std1_wave1, std1_flux1 = load_standard("ModelFittingPart4/StandardSpectra/ukb8v.dat")  # closest match #1
std2_wave1, std2_flux1 = load_standard("ModelFittingPart4/StandardSpectra/ukb9v.dat")  # closest match #2

std1_norm1 = std1_flux1 / std1_flux1.max()
std2_norm1 = std2_flux1 / std2_flux1.max()

plt.figure(figsize=(10,6))
plt.plot(unk_wave1,  unk_norm1,  color="black", label="Unknown 1")
plt.plot(std1_wave1, std1_norm1, color="red",   label="Standard: ukb8v.dat")
plt.plot(std2_wave1, std2_norm1, color="blue",  label="Standard: ukb9v.dat")

plt.xlabel("Wavelength [Å]")
plt.ylabel("Normalized Flux")
plt.title("Unknown1 with Two Closest Standard Spectra")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


'''
2. Fit a blackbody curve to your unknown spectrum. Obtain the best-fit temperature and the uncertainty on this value.
'''
print("\nProblem 2: Fitting blockbody curve to unknown spectrum and obtaining best-fit temperature")
# load unknown spectrum
wave1, flux1 = load_unknown("ModelFittingPart4/UnknownSpectra/unknown1")

# normalize flux
flux_norm1 = flux1 / flux1.max()

# pick only continuum-like points using simple mask
mask1 = flux_norm1 > 0.6
lam_cont1  = wave1[mask1]
flux_cont1 = flux_norm1[mask1]

# fit the blackbody model
popt1, pcov1 = curve_fit(bb_model, lam_cont1, flux_cont1, p0=[11000, 1])
T_fit1, A_fit1 = popt1
T_err1 = np.sqrt(np.diag(pcov1))[0]

print("Best-fit temperature:", T_fit1, "K")
print("Uncertainty:", T_err1, "K")

# compute best-fit curve for plotting
bb_fit1 = bb_model(wave1, T_fit1, A_fit1)

# Plot
plt.figure(figsize=(10,6))
plt.plot(wave1, flux_norm1, color="black", label="Unknown 1")
plt.plot(wave1, bb_fit1,   color="red",   label=f"Best-fit BB: {T_fit1:.0f} K")
plt.xlabel("Wavelength [Å]")
plt.ylabel("Normalized Flux")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

'''
3.1 Do the same to 3 other unknown spectra
'''
print("\nProblem 3.1: Doing the same to 3 other unknown spectra")
unk_wave2, unk_flux2 = load_unknown("ModelFittingPart4/UnknownSpectra/unknown2")
# this block is to find the 2 closest spectra

std_wave2, std_flux2 = load_standard("ModelFittingPart4/StandardSpectra/ukb0v.dat")  # change to the standard you want

# normalize both so we only compare shape, not absolute flux
unk_norm2  = unk_flux2 / unk_flux2.max()
std_norm2  = std_flux2 / std_flux2.max()

plt.figure(figsize=(10,6))
plt.plot(unk_wave2, unk_norm2, color="black", label="Unknown 2")
plt.plot(std_wave2, std_norm2, color="red", label="Standard: uka7v.dat")

plt.xlabel("Wavelength [Å]")
plt.ylabel("Normalized Flux")
plt.title("Unknown2 vs Standard")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# this block is to plot uknown 2 with the two closest standard spectra

# unknown (we already loaded unk_wave, unk_flux)
unk_norm2 = unk_flux2 / unk_flux2.max()

# load the two closest standards
std1_wave2, std1_flux2 = load_standard("ModelFittingPart4/StandardSpectra/ukb0v.dat")  # closest match #1
std2_wave2, std2_flux2 = load_standard("ModelFittingPart4/StandardSpectra/uko9v.dat")  # closest match #2

std1_norm2 = std1_flux2 / std1_flux2.max()
std2_norm2 = std2_flux2 / std2_flux2.max()

plt.figure(figsize=(10,6))
plt.plot(unk_wave2,  unk_norm2,  color="black", label="Unknown 2")
plt.plot(std1_wave2, std1_norm2, color="red",   label="Standard: ukb0v.dat")
plt.plot(std2_wave2, std2_norm2, color="blue",  label="Standard: uko9v.dat")

plt.xlabel("Wavelength [Å]")
plt.ylabel("Normalized Flux")
plt.title("Unknown2 with Two Closest Standard Spectra")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# this block is to plot the unknown and its best fit blackbody curve

# load unknown spectrum
wave2, flux2 = load_unknown("ModelFittingPart4/UnknownSpectra/unknown2")

# normalize flux
flux_norm2 = flux2 / flux2.max()

# pick only continuum-like points using simple mask
mask2 = flux_norm2 > 0.5
lam_cont2  = wave2[mask2]
flux_cont2 = flux_norm2[mask2]

# fit the blackbody model
popt2, pcov2 = curve_fit(bb_model, lam_cont2, flux_cont2, p0=[20000, 1], maxfev=2000)
T_fit2, A_fit2 = popt2
T_err2 = np.sqrt(np.diag(pcov2))[0]

print("Best-fit temperature:", T_fit2, "K")
print("Uncertainty:", T_err2, "K")

# compute best-fit curve for plotting
bb_fit2 = bb_model(wave2, T_fit2, A_fit2)

# plot
plt.figure(figsize=(10,6))
plt.plot(wave2, flux_norm2, color="black", label="Unknown 2")
plt.plot(wave2, bb_fit2,   color="red",   label=f"Best-fit BB: {T_fit2:.0f} K")
plt.xlabel("Wavelength [Å]")
plt.ylabel("Normalized Flux")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


'''
3.2 Same thing
'''
print("\nProblem 3.2")
unk_wave3, unk_flux3 = load_unknown("ModelFittingPart4/UnknownSpectra/unknown3")

# this block is to find the 2 closest spectra

std_wave3, std_flux3 = load_standard("ModelFittingPart4/StandardSpectra/uko5v.dat")  # change to the standard you want

# normalize both so we only compare shape, not absolute flux
unk_norm3  = unk_flux3 / unk_flux3.max()
std_norm3  = std_flux3 / std_flux3.max()

plt.figure(figsize=(10,6))
plt.plot(unk_wave3, unk_norm3, color="black", label="Unknown 3")
plt.plot(std_wave3, std_norm3, color="red", label="Standard: uka7v.dat")

plt.xlabel("Wavelength [Å]")
plt.ylabel("Normalized Flux")
plt.title("Unknown3 vs Standard")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# this block is to plot uknown 3 with the two closest standard spectra

# unknown (we already loaded unk_wave, unk_flux)
unk_norm3 = unk_flux3 / unk_flux3.max()

# load the two closest standards
std1_wave3, std1_flux3 = load_standard("ModelFittingPart4/StandardSpectra/ukb8v.dat")  # closest match #1
std2_wave3, std2_flux3 = load_standard("ModelFittingPart4/StandardSpectra/uko5v.dat")  # closest match #2

std1_norm3 = std1_flux3 / std1_flux3.max()
std2_norm3 = std2_flux3 / std2_flux3.max()

plt.figure(figsize=(10,6))
plt.plot(unk_wave3,  unk_norm3,  color="black", label="Unknown 3")
plt.plot(std1_wave3, std1_norm3, color="red",   label="Standard: ukb8v.dat")
plt.plot(std2_wave3, std2_norm3, color="blue",  label="Standard: uko5v.dat")

plt.xlabel("Wavelength [Å]")
plt.ylabel("Normalized Flux")
plt.title("Unknown3 with Two Closest Standard Spectra")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# this block is to plot the unknown and its best fit blackbody curve

# load unknown spectrum
wave3, flux3 = load_unknown("ModelFittingPart4/UnknownSpectra/unknown3")

# normalize flux
flux_norm3 = flux3 / flux3.max()

# pick only continuum-like points using simple mask
mask3 = flux_norm3 > 0.5
lam_cont3  = wave3[mask3]
flux_cont3 = flux_norm3[mask3]

# fit the blackbody model
popt3, pcov3 = curve_fit(bb_model, lam_cont3, flux_cont3, p0=[20000, 1], maxfev=2000)
T_fit3, A_fit3 = popt3
T_err3 = np.sqrt(np.diag(pcov3))[0]

print("Best-fit temperature:", T_fit3, "K")
print("Uncertainty:", T_err3, "K")

# compute best-fit curve for plotting
bb_fit3 = bb_model(wave3, T_fit3, A_fit3)

# plot
plt.figure(figsize=(10,6))
plt.plot(wave3, flux_norm3, color="black", label="Unknown 3")
plt.plot(wave3, bb_fit3,   color="red",   label=f"Best-fit BB: {T_fit3:.0f} K")
plt.xlabel("Wavelength [Å]")
plt.ylabel("Normalized Flux")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

unk_wave4, unk_flux4 = load_unknown("ModelFittingPart4/UnknownSpectra/unknown4")

print("\nProblem 3.3:")

# this block is to find the 2 closest spectra

std_wave4, std_flux4 = load_standard("ModelFittingPart4/StandardSpectra/ukf5v.dat")  # change to the standard you want

# normalize both so we only compare shape, not absolute flux
unk_norm4  = unk_flux4 / unk_flux4.max()
std_norm4  = std_flux4 / std_flux4.max()

plt.figure(figsize=(10,6))
plt.plot(unk_wave4, unk_norm4, color="black", label="Unknown 4")
plt.plot(std_wave4, std_norm4, color="red", label="Standard: uka7v.dat")

plt.xlabel("Wavelength [Å]")
plt.ylabel("Normalized Flux")
plt.title("Unknown3 vs Standard")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# this block is to plot uknown 4 with the two closest standard spectra

# unknown (we already loaded unk_wave, unk_flux)
unk_norm4 = unk_flux4 / unk_flux4.max()

# load the two closest standards
std1_wave4, std1_flux4 = load_standard("ModelFittingPart4/StandardSpectra/uka7v.dat")  # closest match #1
std2_wave4, std2_flux4 = load_standard("ModelFittingPart4/StandardSpectra/ukf0v.dat")  # closest match #2

std1_norm4 = std1_flux4 / std1_flux4.max()
std2_norm4 = std2_flux4 / std2_flux4.max()

plt.figure(figsize=(10,6))
plt.plot(unk_wave4,  unk_norm4,  color="black", label="Unknown 4")
plt.plot(std1_wave4, std1_norm4, color="red",   label="Standard: uka7v.dat")
plt.plot(std2_wave4, std2_norm4, color="blue",  label="Standard: ukf0v.dat")

plt.xlabel("Wavelength [Å]")
plt.ylabel("Normalized Flux")
plt.title("Unknown4 with Two Closest Standard Spectra")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# this block is to plot the unknown 4 and its best fit blackbody curve

# load unknown spectrum
wave4, flux4 = load_unknown("ModelFittingPart4/UnknownSpectra/unknown4")

# normalize flux
flux_norm4 = flux4 / flux4.max()

# pick only continuum-like points using simple mask
mask4 = flux_norm4 > 0.5
lam_cont4  = wave4[mask4]
flux_cont4 = flux_norm4[mask4]

# fit the blackbody model
popt4, pcov4 = curve_fit(bb_model, lam_cont4, flux_cont4, p0=[20000, 1], maxfev=2000)
T_fit4, A_fit4 = popt4
T_err4 = np.sqrt(np.diag(pcov4))[0]

print("Best-fit temperature:", T_fit4, "K")
print("Uncertainty:", T_err4, "K")

# compute best-fit curve for plotting
bb_fit4 = bb_model(wave4, T_fit4, A_fit4)

# Plot
plt.figure(figsize=(10,6))
plt.plot(wave4, flux_norm4, color="black", label="Unknown 4")
plt.plot(wave4, bb_fit4,   color="red",   label=f"Best-fit BB: {T_fit4:.0f} K")
plt.xlabel("Wavelength [Å]")
plt.ylabel("Normalized Flux")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 18.7 (slightly) more advanced model fitting
'''
1. Isochrone Fitting
'''
# Go back to the star clusters and isochrones question. Based on least squares, or an approach 
# of your choice, think about how you might actually fit the isochrones to the data. 
# Then, attempt to fit the models to the data, using your adopted approach. Comment on its efficacy.

# especially helpful https://astronomy.stackexchange.com/questions/34526/how-to-evaluate-the-fit-of-an-isochrone-to-a-stellar-population
# not very technical but useful for my understanding https://voyages.sdss.org/expeditions/expedition-to-the-milky-way/star-clusters/isochrone-fitting/
# in-depth and technical https://academic.oup.com/mnras/article/373/3/1251/1064007
# this problem took me a while to understand, the instructions felt a bit ambiguous to me
import numpy as np
import matplotlib.pyplot as plt

# cluster data
data = np.genfromtxt("ModelFittingPart4/StarClusters/M67/M67_raw.csv", delimiter=",", names=True)
g_mag = data['gMeanPSFMag']
r_mag = data['rMeanPSFMag']
color_data = g_mag - r_mag  # g - r

# isochrones and ages
iso_files = ["M67_1GY.txt","M67_2GY.txt","M67_3GY.txt","M67_4GY.txt","M67_5GY.txt","M67_6GY.txt","M67_7GY.txt","M67_8GY.txt"]
iso_ages  = [1,2,3,4,5,6,7,8]

# trial magnitude offsets
magnitude_offsets = np.linspace(8.5, 11.0, 51)
best_S = np.inf
best_age = None
best_mu = None
best_g_shifted = None
best_color_iso = None

# using a for-loop grid-search method https://stackoverflow.com/questions/13370570/elegant-grid-search-in-python-numpy
# also neat https://stackoverflow.com/questions/20754230/grid-search-function-in-python
for fname, age in zip(iso_files, iso_ages):
    iso = np.loadtxt(f"ModelFittingPart4/StarClusters/M67/{fname}", comments="#")
    gp1 = iso[:, 6]
    rp1 = iso[:, 7]
    color_iso = gp1 - rp1  # gp1 - rp1
    # need gp1 to be increasing in Xfor np.interp to work, so sorting everything together
    indices = np.argsort(gp1)
    gp1 = gp1[indices]
    color_iso = color_iso[indices]
    # syntax/logic reminder https://numpy.org/doc/stable/user/basics.indexing.html#boolean-or-mask-index-arrays
    for mu in magnitude_offsets:
        g_shift = gp1 + mu
        mask = (g_mag >= g_shift.min()) & (g_mag <= g_shift.max())
        
        if not np.any(mask):
            continue
        # check logichttps://www.geeksforgeeks.org/python/numpy-interp-function-python/
        # https://docs.scipy.org/doc//numpy-1.8.0/reference/generated/numpy.interp.htm
        color_model = np.interp(g_mag[mask], g_shift, color_iso)
        residuals = color_data[mask] - color_model
        S = np.sum(residuals**2)
        
        if S < best_S:
            best_S = S
            best_age = age
            best_mu = mu
            best_g_shifted = g_shift
            best_color_iso = color_iso

print("\n18.7 - (slightly) more advanced model fitting")
print("Problem 1: Least-squares fit of isochrones to M67")
print(f"Best-fit age: {best_age} Gyr")
print(f"Best-fit magnitude shift (mu): {best_mu:.2f} mag")
print(f"Minimum sum of squared residuals: {best_S:.2e}")
print ("I fit the isochrones with a simple least-squares grid search over age and magnitude shift, minimizing the sum of squared color " \
       "residuals between the data and the model. This works reasonably well as it picks an age and shift that agree with my eyeballed estimate "
       "and gives an objective way to decide which isochrone matches the CMD best. However, it’s still an approximation since it only uses " \
       "vertical color residuals and relies on a coarse parameter grid rather than a more sophisticated fitting method.")

plt.scatter(g_mag, color_data, s=10, alpha=0.3, color='tab:orange', label='M67 stars')
plt.plot(best_g_shifted, best_color_iso, color='tab:purple', linewidth=5.0, alpha=0.5, label=f'Isochrone: {best_age} Gyr (mu={best_mu:.2f})')

plt.xlabel('gMeanPSFMag (mag)')
plt.ylabel('gMeanPSFMag - rMeanPSFMag (mag)')
plt.title('M67 Color-Magnitude Diagram with Least-Squares Isochrone Fit')
plt.gca().invert_xaxis()
plt.legend()
plt.tight_layout()
plt.savefig('M67_CMD_isochrone_least_squares.pdf')
plt.show()

# Output:
# 18.7 - (slightly) more advanced model fitting
# Problem 1: Least-squares fit of isochrones to M67
# Best-fit age: 2 Gyr
# Best-fit magnitude shift (mu): 10.75 mag
# Minimum sum of squared residuals: 1.24e+03
# I fit the isochrones with a simple least-squares grid search over age and magnitude shift, minimizing the sum of squared color residuals 
# between the data and the model. This works reasonably well as it picks an age and shift that agree with my eyeballed estimate and gives 
# an objective way to decide which isochrone matches the CMD best. However, it’s still an approximation since it only uses vertical color 
# residuals and relies on a coarse parameter grid rather than a more sophisticated fitting method.