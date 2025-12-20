# 1. Compute the integral. The output from quad will be two numbers. What are they? How does the result of this numerical integration compare to performing the integration analytically?
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html

from scipy.integrate import quad

print("26.1 Introductory Numerical Integration Exercises\nProblem 1: Basic integral")

def f1(x):
    """
    This is a simple function for integration: f(x) = x^2.

    Input:
        x (value to evaluate)

    Output:
        x squared
    """
    return x**2

result, error = quad(f1, 2, 5)
print(result, error)

# 2. Again using scipy.quad, evaluate the following integral
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html

from scipy.integrate import quad

print("\nProblem 2: Another basic integral")

def f2(x):
    """
    Computes f(x) = 3x^3 for use in numerical integration.

    Input:
        x (value where the function is evaluated)

    Output:
        the value of 3*x^3.
    """
    return 3*x**3

result, error = quad(f2, 0, 1)
print(result, error)

# 3. Try evaluating the integral in Equation 34 using ”fixed quad” and ”romberg” within scipy.
# https://adocs.scipy.org/doc/scipy/reference/generated/scipy.integrate.fixed_quad.html
# https://www.geeksforgeeks.org/python/python-scipy-integrate-romberg-method/

from scipy.integrate import fixed_quad

print("\nProblem 3: Fixed quad and romberg (which is entirely deprecated and EOL)")
print("This code may or may not fail, depending on your version of SciPy.")

# n = number of evaluation points
result_fixed, _ = fixed_quad(f2, 0, 1, n=5)
print("fixed_quad result:", result_fixed)


from scipy.integrate import romberg

result_romb = romberg(f2, 0, 1)
print("romberg result:", result_romb)

# 4. Write a script that uses quad to evaluate the following integral. Write your script such that the equation being integrated is a Python function that sccipy.quad calls (this function can be a normal function r a Lambda function, your choice).

from scipy.integrate import quad

print("\nProblem 4: Scripting to solve an integral")

# cubic function to integrate
def cubic(x, alpha, beta, gamma, eps):
    """
    Computes the cubic function f(x) = αx^3 + βx^2 + γx + ε.

    Inputs:
        x (point to evaluate)
        alpha, beta, gamma, eps (coefficients of the cubic)

    Output:
        float (value of the cubic at x)
    """
    return alpha*x**3 + beta*x**2 + gamma*x + eps

# example 1
res1, err1 = quad(cubic, 0, 1, args=(1, 0, 0, 0))   # f(x) = x^3
print("Integral 1:", res1, "Error:", err1)

# example 2
res2, err2 = quad(cubic, -1, 2, args=(2, -1, 0.5, 3))
print("Integral 2:", res2, "Error:", err2)

# example 3
res3, err3 = quad(cubic, 0, 3, args=(0, 1, -2, 1))
print("Integral 3:", res3, "Error:", err3)

# All 3 methods give basically the same value: 0.75, but the fixed quad result is off by a very small fraction of 10. This matches
# the analytical result of 3/4. Numerically, these two methods agree with quad to within machine precision. Quad also returns an
# error estimate, while these two only give the integral value. For this example, all three methods were instant, but that might not
# be the case for more complicated examples.

# The first value (39) is the estimated value of the integral. The second value (4.33*10^-13) is an estimate of the absolute error in that value.
# The analytic solution is 39. Quad returned 39.00000000000001 which is super duper close, as in within machine precision.

# 26.2 A Little More Challenging Numerical Integration Exercises
# 1. Write function that takes 1-D array of s values, values for a, b, andσint, + returns a 1-D array of µvalues.

import numpy as np

print("\n26.2 A Little More Challenging Numerical Integration Exercises\nProblem 1: 1-D Array function")

def gmugivens(mu, s, a, b, sigma_int):
    """
    Compute g(mu | s) from equation 39.

    Inputs:
    mu (Log black hole masses)
    s (Log stellar masses)
    a, b (Intercept and slope)
    sigma_int (Intrinsic scatter)

    Outputs:
    g (Probability g(mu | s))
    """
    prefactor = 1.0 / (np.sqrt(2 * np.pi) * sigma_int)
    exponent = -((mu - a - b * (s - 11))**2) / (2 * sigma_int**2)
    return prefactor * np.exp(exponent)

# 2. Write a function that evaluates the product of equations 39 and 45, and returns the output.

print("\nProblem 2: Function evaluation equations 39 & 45")
def gfphis(x, mu, a, b, sigma_int, phi_star, alpha, s_star):
    """
    Product of equation 39 and equation 45.

    Inputs:
    x (log stellar mass; first argument for integration)
    mu (log black hole mass)
    a, b (Intercept and slope)
    sigma_int (Intrinsic scatter)
    phi_star (SMF normalization)
    alpha (Low-mass slope)
    s_star (Characteristic stellar mass)

    Output:
    g(mu | s) * Phi_*(s)
    """

    # eq39: g(mu | s)
    g = gmugivens(mu, x, a, b, sigma_int)

    # eq45: Phi_*(s)
    phi_s = np.log(10) * phi_star * \
            10**((x - s_star) * (alpha + 1)) * \
            np.exp(-10**(x - s_star))

    return g * phi_s

# 3. Write a function that takes in the inputs for Equation 46 and returns the output from this equation.

print("\nProblem 3: Function with inputs for equation 46")
def PhiBHAC(x, phi_star, alpha, mu_star, beta):
    """
    Equation 46: Active black hole mass function.

    Inputs:
    x (log black hole mass; first argument for integration)
    phi_star (Normalization)
    alpha (Low-mass slope)
    mu_star (Characteristic black hole mass)
    beta (High-mass cutoff)

    Output:
    Phi_bh,act(mu)
    """

    return (np.log(10) * phi_star *
            10**((x - mu_star) * (alpha + 1)) *
            np.exp(-(10**(x - mu_star))**beta))

# 4. Write a function that takes in the inputs for Equation 47 and returns its output.

print("\nProblem 4: Function with inputs for equation 47")
def PhiER(x, phi_star, alpha_lambda, lambda_lambda):
    """
    Equation 47: Eddington ratio distribution function.

    Inputs:
    x (Eddington ratio λ; first argument for integration)
    phi_star (Normalization)
    alpha_lambda (Power-law slope)
    lambda_lambda (Characteristic Eddington ratio)

    Output:
    Phi_lambda(λ)
    """

    return (phi_star / (np.log(np.e) * lambda_lambda) *
            (x / lambda_lambda)**alpha_lambda *
            np.exp(-x / lambda_lambda))

# 5. Write a function that evaluates the product of Equations 41 and 39. Call this function omegagf.

from scipy.integrate import quad
import numpy as np

print("\nProblem 5: Function evaluation the product of equations 39 & 41")

def omegagf(x, s, a, b, sigma_int, minlum,
            phi_star_smf, alpha_smf, s_star,
            phi_star_bh, alpha_bh, mu_star, beta,
            phi_star_er, alpha_lambda,
            lambda0=1.0, kappa=0.074):
    """
    Product of equations 41 and 39, built based on the pdf.

    Inputs:
    x (mu = log black hole mass)
    s (log stellar mass values)
    a, b (parameters of the mu–s relation)
    sigma_int (intrinsic scatter in mu at fixed s)
    minlum (survey limiting luminosity)

    phi_star_smf, alpha_smf, s_star (stellar Mass Function parameters (eq45))

    phi_star_bh, alpha_bh, mu_star, beta (active Black Hole Mass Function parameters (eq46))

    phi_star_er (ERDF normalization)
    alpha_lambda (ERDF power-law slope)

    lambda0 (ERDF normalization constant in eq48 (default = 1.0))
    kappa (ERDF slope in eq48 (default = 0.074))

    Outputs:
    omega (selection-weighted probability density evaluated at each s value)
    
    """

    # evaluate PhiBHAC (eq 46) at mu=x
    phi_bh_act = PhiBHAC(x, phi_star_bh, alpha_bh, mu_star, beta)

    # integrate gfphis over s to get Phi_bh,tot(mu) (eq40)
    # (quad returns (result, error) but only want result)
    phi_bh_tot, _ = quad(gfphis, 8.0, 13.0, args=(x, a, b, sigma_int, phi_star_smf, alpha_smf, s_star))

    # active fraction p_act(mu) = Phi_act / Phi_tot (eq42)
    p_act = phi_bh_act / phi_bh_tot

    # evaluate g(mu|s) (eq39) for vector s
    g = gmugivens(x, s, a, b, sigma_int)

    # minimum Eddington ratio (given snippet)
    minedd = minlum / (1.26e38 * (10.0**x))

    # compute lambda-hat via eq48 (account for log; lambda0=1.0 and kappa=0.074)
    log10_lhat = np.log10(lambda0) + kappa * (x - 8.0)
    lhat = 10.0**log10_lhat

    # integrate PhiER from minedd to minedd+10
    int_phier, _ = quad(PhiER, minedd, minedd + 10.0,
                        args=(phi_star_er, alpha_lambda, lhat))

    # return product: (integral of PhiER) * (active fraction) * g(mu|s)
    return int_phier * p_act * g

# 6. Write a function identical to omegagf except that it returns the product of the product of four variables: those stated before and x.

print("\nProblem 6: Function identical to omegagf")
def xomegag(x, s, a, b, sigma_int, minlum,
            phi_star_smf, alpha_smf, s_star,
            phi_star_bh, alpha_bh, mu_star, beta,
            phi_star_er, alpha_lambda,
            lambda0=1.0, kappa=0.074):
    """
    Identical to omegagf, except returns x * omegagf.

    Inputs:
    x (mu = log black hole mass)
    s (log stellar mass values)
    a, b (parameters of the mu–s relation)
    sigma_int (intrinsic scatter in mu at fixed s)
    minlum (survey limiting luminosity)

    phi_star_smf, alpha_smf, s_star (stellar Mass Function parameters (eq45))

    phi_star_bh, alpha_bh, mu_star, beta (active Black Hole Mass Function parameters (eq46))

    phi_star_er (ERDF normalization)
    alpha_lambda (ERDF power-law slope)

    lambda0 (ERDF normalization constant in eq48 (default = 1.0))
    kappa (ERDF slope in eq48 (default = 0.074))

    Outputs:
    xomega (x times the output of omegagf)
    """

    return x * omegagf(x, s, a, b, sigma_int, minlum,
                       phi_star_smf, alpha_smf, s_star,
                       phi_star_bh, alpha_bh, mu_star, beta,
                       phi_star_er, alpha_lambda,
                       lambda0, kappa)


# 7. Next, we need to write a function that ties all of the above together. This function will evaluate Equation 44.

from scipy.integrate import quad
import numpy as np

print("\nProblem 7: Tying it all together to evaluate equation 44")

def VSBias(s, minlum, a, b, sigma_int,
           phi_star_smf, alpha_smf, s_star,
           phi_star_bh, alpha_bh, mu_star, beta,
           phi_star_er, alpha_lambda,
           lambda0=1.0, kappa=0.074):
    """
    Evaluate equation 44 as shown in pdf.

    Inputs:
    s (logged stellar masses)
    minlum (survey limiting luminosity)
    a, b, sigma_int (parameters for g(mu|s))
    phi_star_smf, alpha_smf, s_star (SMF parameters)
    phi_star_bh, alpha_bh, mu_star, beta (active BHMF parameters)
    phi_star_er, alpha_lambda (ERDF parameters)
    lambda0, kappa (constants for eq48)

    Outputs:
    mu_obs_mean (predicted observed mean logged SMBH masses for each input s)
    
    """

    s = np.asarray(s)
    mu_obs_mean = np.zeros_like(s, dtype=float)

    # integrate over mu from 6.5 to 10.0 as instructed
    mu_lo, mu_hi = 6.5, 10.0

    for i, si in enumerate(s):

        # integrate xomegag (numerator integrand)
        num, _ = quad(xomegag, mu_lo, mu_hi,
                      args=(np.array([si]), a, b, sigma_int, minlum,
                            phi_star_smf, alpha_smf, s_star,
                            phi_star_bh, alpha_bh, mu_star, beta,
                            phi_star_er, alpha_lambda,
                            lambda0, kappa))

        # integrate omegagf (denominator integrand)
        den, _ = quad(omegagf, mu_lo, mu_hi,
                      args=(np.array([si]), a, b, sigma_int, minlum,
                            phi_star_smf, alpha_smf, s_star,
                            phi_star_bh, alpha_bh, mu_star, beta,
                            phi_star_er, alpha_lambda,
                            lambda0, kappa))

        # ratio = <mu>(s)
        mu_val = num / den

        # "sensible number" check
        if (not np.isfinite(mu_val)) or (mu_val < 0) or (mu_val > 20):
            mu_val = np.nan

        mu_obs_mean[i] = mu_val

    return mu_obs_mean

# 8. We are nearly there!

import numpy as np
import matplotlib.pyplot as plt

print("\nProblem 8: We are nearly there!")

# s array (20 vals from 10.5 to 12.0)
s = np.linspace(10.5, 12.0, 20)

# constants
a = 8.2
b = 1.1
sigma_int = 0.4
minlum = 3e45

# SMF (eq45) params given as logs if needed
phi_star_smf = 10**(-3.01)   # Φ*
s_star = 10.86
alpha_smf = -1.37

# active BHMF (eq46) params
phi_star_bh = 10**(-4.88)    # Φ•*
mu_star = 8.06
alpha_bh = -1.19
beta = 0.57

# ERDF (eq47) params
# ERDF normalization cancels in VSBias ratio, so set to 1.0
phi_star_er = 1.0
alpha_lambda = -1.09

# eq48 constants
# prompt gives λ^ = -1.02 -> treat as log10(lambda0)
lambda0 = 10**(-1.02)
kappa = 0.074

# "true" mu from eq38
mu_true = a + b * (s - 11)

# "predicted" mu from VSBias (eq44 model)
mu_pred = VSBias(s, minlum, a, b, sigma_int,
                 phi_star_smf, alpha_smf, s_star,
                 phi_star_bh, alpha_bh, mu_star, beta,
                 phi_star_er, alpha_lambda,
                 lambda0=lambda0, kappa=kappa)

# plot both
plt.figure(figsize=(8,5))
plt.plot(s, mu_true, label="True (Eq. 38)")
plt.plot(s, mu_pred, label="Predicted observed (VSBias)")
plt.xlabel("s = log M*")
plt.ylabel("μ = log M_BH")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# The predicted observed relation sits above the true one (especially at lower stellar masses) which shows that the observations are 
# biased toward higher black hole masses. This bias gets smaller at higher stellar masses, where the two lines overlap.

# Because of this, an observer would probably incorrectly conclude that black holes in lower-mass galaxies are more massive than 
# they really are. THey would also conclude that the black hole–stellar mass relation is flatter than the true underlying relation.

# 26.3 Monte Carlo Exercises
'''
1. Modify the coinflip code in §23.1.1 to do the following:
• Do a total of 1000 flips
• Store the value of ’results’ every five flips (so, create a 200 element array
  where each element contains the value of results after five (further) flips)
• At the end of the simulation, produce a plot of the value of ’results’ on the
  y axis, vs. the number of flips on the x axis.
'''

import numpy as np
from matplotlib import pyplot as pt

print("26.3 Monte Carlo Exercises\nProblem 1: Modified coin-flip code and plot")

def flipcoin():
    """
    INPUTS: none
    OUTPUTS: outcome (int; 0=tails, 1=heads)
    DESCRIPTION: Simulates a coin flip by returning 0 or 1 with equal probability
    """
    return np.random.randint(0, 2) # 0=tails, 1=heads

nflips = 1000
N = 5
results = [] # empty array for flips
flips = [] # x-axis values
running_total = 0

for i in range(1, nflips + 1):
    running_total += flipcoin()
    if i % N == 0:
        results.append(running_total)
        flips.append(i)

print(f"In this simulation, we did:\n{nflips} flips of the coin.")

pt.plot(flips, results)
pt.xlabel("Number of flips")
pt.ylabel("results (running total of heads)")
pt.title("1000 coin flips; heads vs tails")
pt.show()

# Output:
# 26.3 Monte Carlo Exercises
# Problem 1: Modified coin-flip code and plot
# In this simulation, we did:
# 1000 flips of the coin.

'''
2. The code in §23.1.1 is illustrative, but hideously inefficient. It’s possible to sim-
ulation the average of an arbitrary number of coinflips and print the result to the
screen without writing a function, without using a for loop, in one single com-
mand. Write such a single line of code. Hint, you still need np.random.randint,
but have a look at its documentation and recall you can ’nest’ commands).
'''

import numpy as np

print("\nProblem 2: Single line coinflip simulated mean")

N = 1000 
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html
print(np.random.randint(0, 2, size=N).mean()) # arbitary number, N, as per the question, not entirely sure if this is what is expected here

# Output: 
# Problem 2: Single line coinflip simulated mean
# 0.516

'''
3. Using the code in §23.1.2, set the bias in the function to any number of your
choice between 0.2 and 0.7, specified to five decimal places. Approximately how
many flips must you perform to then recover this bias to an accuracy of 2%?
'''

import numpy as np

print("\nProblem 3: Bias recovery")

def flipbiasedcoin():
    """
    INPUTS: none
    OUTPUTS: outcome (int; 0=tails, 1=heads)
    DESCRIPTION: Simulates a biased coin flip with P(tails)=bias and returns 0 for tails or 1 for heads
    """
    bias = 0.45678  # P(tails)
    return 0 if np.random.uniform(0.0, 1.0) < bias else 1  # 0 = tails, 1 = heads


# sigma_p = sqrt(p(1-p)/n)
# n = p(1-p)/(0.02p)^2
true_bias = 0.45678
target_frac = 0.02
nflips = int(np.ceil(true_bias * (1 - true_bias) / ((target_frac * true_bias) ** 2)))

results = 0
for _ in range(nflips):
    results += flipbiasedcoin()  # sums 1s; mean = P(heads) = 1 - bias

p_heads = results / nflips
derived_bias = 1 - p_heads

print("True bias (tails):", true_bias)
print("Approx flips needed for ~2% accuracy:", nflips)
print("Derived bias (tails):", derived_bias)
print("Relative error:", abs(derived_bias - true_bias) / true_bias)

# Output: 
# Problem 3: Bias recovery
# True bias (tails): 0.45678
# Approx flips needed for ~2% accuracy: 2974
# Derived bias (tails): 0.4583053127101546
# Relative error: 0.0033392721006931182

'''
4. Write a Python function that simulates rolling two six sided dice and adding
their values together (so for example if your dice came up 2 and 3, the result
of the simulation would be 5). Write a Monte Carlo simulation to predict the
most probable total from rolling these two wix sided dice.
'''

import numpy as np

print("\nProblem 4: Monte Carlo dice-roll simulation")

def roll_two_dice_sum():
    """
    INPUTS: none
    OUTPUTS: total (int; sum of two 6-sided dice, 2 through 12)
    DESCRIPTION: Simulates rolling two fair six-sided dice and returns the sum of their face values
    """
    die1 = np.random.randint(1, 7)
    die2 = np.random.randint(1, 7)
    return die1 + die2


N = 1000  # monte carlo trials
totals = np.empty(N, dtype=int)

for i in range(N):
    totals[i] = roll_two_dice_sum()

counts = np.bincount(totals, minlength=13)
most_prob_total = np.argmax(counts[2:13]) + 2

print("Most probable total:", most_prob_total)
print("Estimated probability:", counts[most_prob_total] / N)

# Output:
# Problem 4: Monte Carlo dice-roll simulation
# Most probable total: 7
# Estimated probability: 0.159

'''
5. You have 6 6-sided dice, 5 5-sided dice, 4 4-sided dice, and 2 19-sided dice. You
roll all of these dice at once and add their values to get the total. Write a Monte
Carlo simulation to calculate:
    • The most likely total value
    • The 68% and 95% range of values on either side of this most probable
      value (hint: make a histogram of the values and use ’percentile’ within
      numpy).
'''

import numpy as np
from matplotlib import pyplot as pt

print("\nProblem 5: Many different dice rolls")

def roll_lots_of_dice():
    """
    INPUTS: none
    OUTPUTS: total (int; sum of 6d6 + 5d5 + 4d4 + 2d19)
    DESCRIPTION: Simulates rolling many dice at once (six 6-sided, five 5-sided, four 4-sided, two 19-sided) and returns the total
    """
    return (np.random.randint(1, 7, size=6).sum() + np.random.randint(1, 6, size=5).sum() + np.random.randint(1, 5, size=4).sum() + np.random.randint(1, 20, size=2).sum())


N = 1000
totals = np.array([roll_lots_of_dice() for i in range(N)])

counts = np.bincount(totals)
most_likely_value = np.argmax(counts) # easiest I came up with https://numpy.org/doc/stable/reference/generated/numpy.argmax.html

# syntax reminder https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
percentile_16, percentile_84 = np.percentile(totals, [16, 84])
percentile_2_5, percentile_97_5 = np.percentile(totals, [2.5, 97.5])

print(f"Most likely total: {most_likely_value}")
print(f"68% interval: {percentile_16} to {percentile_84}")
print(f"95% interval: {percentile_2_5} to {percentile_97_5}")

# refresher, forgot binning https://matplotlib.org/stable/gallery/statistics/hist.html
bars = np.arange(totals.min(), totals.max() + 2)
pt.hist(totals, bins = bars, edgecolor='k')
pt.axvline(most_likely_value, linewidth=2)
pt.axvline(percentile_16, linestyle='--')
pt.axvline(percentile_84, linestyle='--')
pt.axvline(percentile_2_5, linestyle='--')
pt.axvline(percentile_97_5, linestyle='--')
pt.title("Histogram of Total Rolls (6d6 + 5d5 + 4d4 + 2d19)")
pt.xlabel("Total")
pt.ylabel("Count")
pt.show()

# Output:
# Problem 6: Many different dice rolls
# Most likely total: 68
# 68% interval: 56.0 to 76.0
# 95% interval: 46.0 to 86.0

'''
6. Improve the code in §23.1.3 to solve the genralized Monty Hall problem; a
game with D doors behind which are M goats and N cars (D, M, N are positive
integers, D ≥ 3, M, N ≥ 1, and M + N = D). Is switching always the right
strategy?27 Hint: This requires more work than it might look as you have to
deal with the calculation of the probability after the host has removed a door
(with 3 doors this is trivial but with more than 3 it takes a bit of thought).
'''

import numpy as np

print("\nProblem 6: Improved Monty Hall problem")

# this was a fun one to figure out
def monty_hall_general(D, M, N, ngames):
    """
    INPUTS: D (int; number of doors), M (int; number of goats), N (int; number of cars), ngames (int; Monte Carlo trials)
    OUTPUTS: stick_rate (float), switch_rate (float)
    DESCRIPTION: Simulates generalized Monty Hall with D doors (M goats, N cars). Player picks 1 door, host opens 1 other door (goat if possible), then player either sticks or switches to a random remaining unopened door.
    """
    stick_wins = 0
    switch_wins = 0

    for _ in range(ngames):
        # randomly choose which doors are cars this game
        car_doors = set(np.random.choice(D, size=N, replace=False)) # replace=False to give N unique door indices, otherwise same door could be picked. I think np.random.permutation could work but I couldn't figure out the logic
        choice = np.random.randint(D)
        
        # host opens a goat door not picked, if possible
        goat_options = [d for d in range(D) if d != choice and d not in car_doors]
        if goat_options:
            opened = goat_options[np.random.randint(len(goat_options))]
        else:
            opened = np.random.choice([d for d in range(D) if d != choice])

        stick_wins += (choice in car_doors)

        # switch to a random unopened door (excluding pick and opened)
        remaining = [d for d in range(D) if d != choice and d != opened]
        new_pick = remaining[np.random.randint(len(remaining))]
        switch_wins += (new_pick in car_doors)

    return stick_wins / ngames, switch_wins / ngames

D, M, N = 10, 7, 3
ngames = 200000

stick_rate, switch_rate = monty_hall_general(D, M, N, ngames)
print(f"Stick win rate: {stick_rate}")
print(f"Switch win rate: {switch_rate}")

# Output:
# Problem 6: Improved Monty Hall problem
# Stick win rate: 0.300175
# Switch win rate: 0.33839

'''
7. Pick any two of the integrals in §26.1 and approximate their solution via Monte
Carlo methods. How close are your answers to those using ’quad’, or an equva-
lent? Which approach seems faster?
'''

import numpy as np

print("\nProblem 7: Monte Carlo Integral")

# https://www.geeksforgeeks.org/python/monte-carlo-integration-in-python/
def monte_carlo_integral(f, a, b, N):
    """
    INPUTS: f (function; vectorized integrand), a (float; lower bound), b (float; upper bound), N (int; number of samples)
    OUTPUTS: I (float; Monte Carlo estimate of integral from a to b)
    DESCRIPTION: Approximates ∫_a^b f(x) dx by sampling N uniform random points in [a,b] and averaging f(x)
    """
    x = np.random.uniform(a, b, size=N)
    return (b - a) * f(x).mean()

N = 1000

# Integral 1: ∫_2^5 x^2 dx
f = lambda x: x**2
I1_monte_carlo = monte_carlo_integral(f, 2.0, 5.0, N)
print(f"Monte Carlo estimate for I1 = ∫_2^5 x^2 dx: {I1_monte_carlo}")

# Integral 2: ∫_0^1 3x^3 dx
I2_monte_carlo = monte_carlo_integral(lambda x: 3 * x**3, 0.0, 1.0, N)
print(f"Monte Carlo estimate for I2 = ∫_0^1 3x^3 dx: {I2_monte_carlo}")

# Output:
# Problem 7: Monte Carlo Integral
# Monte Carlo estimate for I1 = ∫_2^5 x^2 dx: 39.50174825396091
# Monte Carlo estimate for I2 = ∫_0^1 3x^3 dx: 0.7465681764222729

'''
8. Hypothetical star system
'''
# Consider a hypothetical solar system which contains an outer asteroid belt and an inner asteroid belt. Just after the solar system is formed, 
# there are a large (integer) number, N , asteroids in the outer belt and none in the inner belt. However, nearby planets can make asteroids transfer 
# from one belt to the other (either direction). In a single year, the probability that a given asteroid transfers from one belt to the other is given 
# by a truncated normal distribution, centered at: C = nsubi/√αN in which ni is the number of asteroids in the belt the asteroid starts the year in, 
# N is the total number of asteroids in the solar system, and α > 1.05 is some constant. The bounds on the normal distribution are zero on the low side 
# and 0.95 on the high side (a probability of 1.0 means 100% likely to transfer). Write a Monte Carlo simulation to answer the following questions: 
# • For N = 1000 and α = 2, how many years will it take for the number of asteroids in the inner and outer belt to be approximately equal? 
# • For N = 50, make a plot of how many years it takes for the number of asteroids in the initially empty belt to reach 18, as a function of α. 
# • What is the largest value of N for which your code (and computer) can complete the first of these three questions in under five seconds? 
# Hint: This question is made easy with the ’time’ library.

import numpy as np
from matplotlib import pyplot as pt
import time

# just fyi this problem look me longer than the rest of this section combined so I don't think it needs to be made harder for next term
# personal understanding https://en.wikipedia.org/wiki/Discrete-time_Markov_chain

print("\nProblem 8: Hypothetical star system")

sigma = 0.10  # width of the truncated normal for transfer p

def step_year(n_outer, N, alpha):
    """
    INPUTS: n_out (int; number of asteroids in outer belt at start of year), N (int; total asteroids), alpha (float; transfer parameter)
    OUTPUTS: n_out_new (int; updated number of asteroids in outer belt after one year)
    DESCRIPTION: Advances the asteroid-belt system by one year by randomly transferring asteroids between belts using normal probabilities centered at C = n_i / sqrt(alpha*N)
    """
    # n_outer asteroids start in outer belt, n_inner start in inner belt
    n_inner = N - n_outer

    # outer-belt asteroid: get p_out, decide if it transfers
    C_out = n_outer / np.sqrt(alpha * N)
    p_out = np.clip(np.random.normal(C_out, sigma, size=n_outer), 0.0, 0.95)
    out_to_in = (np.random.rand(n_outer) < p_out).sum()

    # inner-belt asteroid: get p_in, decide if it transfers
    C_in = n_inner / np.sqrt(alpha * N)
    p_in = np.clip(np.random.normal(C_in, sigma, size=n_inner), 0.0, 0.95)
    in_to_out = (np.random.rand(n_inner) < p_in).sum()

    # update outer belt after both directions of transfers
    return n_outer - out_to_in + in_to_out

def years_to_balance(N, alpha):
    """
    INPUTS: N (int; total asteroids), alpha (float; transfer parameter)
    OUTPUTS: years (int; number of years until belts are approximately equal)
    DESCRIPTION: Runs the asteroid transfer simulation until the inner and outer belts differ by no more than 2% of N, then returns the elapsed years
    """
    # run until inner and outer are within 2% of N
    n_outer = N
    years = 0
    while abs((N - n_outer) - n_outer) > 0.02 * N:
        years += 1
        n_outer = step_year(n_outer, N, alpha)
    return years

def years_to_reach_inner(N, alpha, target):
    """
    INPUTS: N (int; total asteroids), alpha (float; transfer parameter), target (int; desired inner-belt asteroid count)
    OUTPUTS: years (int; number of years until inner belt reaches target)
    DESCRIPTION: Runs the asteroid transfer simulation starting with an empty inner belt and returns how many years it takes for the inner belt to reach the specified target count
    """
    # run until inner belt reaches target asteroids
    n_outer = N
    years = 0
    while (N - n_outer) < target:
        years += 1
        n_outer = step_year(n_outer, N, alpha)
    return years


# 1) N=1000, alpha=2: years until belts are approx equal
print(f"Years to equal (N=1000, alpha=2): {years_to_balance(1000, 2.0)}")

# 2) N=50: plot years to reach 18 in empty inner belt vs alpha
alphas = np.linspace(1.06, 2000.0, 60)
years = [years_to_reach_inner(50, a, 18) for a in alphas]

pt.plot(alphas, years, marker='o')
pt.xlabel("alpha")
pt.ylabel("Years to reach 18 in inner belt")
pt.title("N=50: Years to reach 18 vs alpha")
pt.show()

# 3) largest N for which part 1 finishes in under 5 seconds (alpha at 2)
Ns = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000, 2048000, 4096000, 8192000]

times = []
for N in Ns:
    time_0 = time.time() # https://www.geeksforgeeks.org/python/python-time-time-method/
    years_to_balance(N, 2.0)
    times.append(time.time() - time_0) # appending to empty times array

# I think this is doable with numpy instead of a for loop but this was easier. Sorry for the inefficiency :-(
best_N = 0
for N, t in zip(Ns, times):
    if t < 5.0:
        best_N = N

print(f"Largest N under 5 seconds: {best_N}")

# Output:
# Problem 8: Hypothetical star system
# Years to ~equal (N=1000, alpha=2): 34
# Largest N under 5 seconds: 4096000

'''
9. Star fusion script
'''
# Suppose there exists inside a star a fusion process that has three distinct steps.
# Put crudely, the first step takes about two minutes, the second takes about five
# minutes, and the third takes about ten minutes. Put more precisely, the times
# for the three stages are:
# • A normal distribution centered on two minites, with a width of 1.5 minutes.
# • A uniform distribution centered on five minutes, with a total width of 2
# minutes.
# • A normal distribution centered on ten minutes, with a width of 3 minutes.
# Write a Monte Carlo simulation in Python that evaluates the probability that
# the whole process (all three steps) takes more, or less, than a given amount
# of time. Use your script to determine how likely it is that the whole process
# finishes in more than 18 minutes or less than 12 minutes.

import numpy as np

print("\nProblem 9: Star fusion script Monte Carlo sims")

N = 500000

# normal (mean=2, sigma=1.5)
t1 = np.random.normal(2.0, 1.5, size=N)
# uniform centered at 5 with total width 2
t2 = np.random.uniform(4.0, 6.0, size=N)
# normal (mean=10, sigma=3)
t3 = np.random.normal(10.0, 3.0, size=N)

total = t1 + t2 + t3

over_18 = (total > 18.0).mean()
under_12 = (total < 12.0).mean()

print(f"Probability of total > 18 min: {over_18}")
print(f"Probability of total < 12 min: {under_12}")

# Output:
# Problem 9: Star fusion script Monte Carlo sims
# Probability of total > 18 min: 0.384424
# Probability of total < 12 min: 0.070588

'''
10. And finally, a nice little essay question to end things. Describe in a few sentences
how you might use Monte Carlo methods to get an uncertainty on a Kendall-τ
test statistic.
'''
# https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.stats.kendalltau.html
# https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
print("\nProblem 10: Short response on Kendall-τ test statistic")
print("We can treat measured data as noisier versions of some underlying 'true' values, and then resample " \
    "many datasets by changing each point according to its uncertainty model. For each dataset, we can " \
    "compute the Kendall-τ statistic. The resulting τ values are a sampling distribution where the standard deviation " \
    "is the Monte Carlo uncertainty on τ and the 16-84 percentiles give the standard 68% interval.")

# Output:
# Problem 10: Short response on Kendall-τ test statistic
# We can treat measured data as noisier versions of some underlying 'true' values, and then resample many datasets 
# by changing each point according to its uncertainty model. For each dataset, we can compute the Kendall-τ statistic. 
# The resulting τ values are a sampling distribution where the standard deviation is the Monte Carlo uncertainty on τ 
# and the 16-84 percentiles give the standard 68% interval

# 26.4 Bayesian Statistics Exercises
'''
1. Starting with the code in §24.3.1, answer the following question. The second
jar has 60 blue and 60 yellow marbles. The first jar has X blue and 15 yellow
marbles. Without looking at which jar you chose from, you reach out and grab
a marble. It is blue. What is the minimum value of X such that the probability
you chose this marble from the first jar is at least 53.5%?
'''

print("\n26.4 Bayesian Statistics Exercises\nProblem 1: Jars and marbles")

jar2blue = 60
jar2yellow = 60
prior = 0.5

p_blue_given_jar2 = jar2blue / (jar2blue + jar2yellow)

for X in range(1, 10000):  # X is integer
    jar1blue = X
    jar1yellow = 15

    p_blue_given_jar1 = jar1blue / (jar1blue + jar1yellow)

    evidence = prior * p_blue_given_jar1 + prior * p_blue_given_jar2
    posterior = (p_blue_given_jar1 * prior) / evidence

    if posterior >= 0.535:
        print(f"Minimum X: {X}")
        print(f"Posterior P(jar1blue): {posterior}")
        break

# Output:
# 26.4 Bayesian Statistics Exercises
# Problem 1: Jars and marbles
# Minimum X: 21
# Posterior P(jar1blue): 0.5384615384615384

'''
2. Starting with the example in §24.3.2, write a script that evaluates the posterior
probability of having the disease after an arbitrary number of positive tests.
Use your script to evaluate how many positive tests in a row are needed to be
more than 99% sure you have the disease.
'''

import numpy as np

print("\nProblem 2: Test efficacy")

test_eff = 0.99
prior_0 = 1e-6  # 1 in a million

# posterior after an arbitrary number of positive tests
n_tests = 5  # could be anything but 5 is the correct answer so I'm going with 5
prior = prior_0

for _ in range(n_tests): # from 24.3.2
    likelihood = test_eff
    evidence = (likelihood * prior) + ((1.0 - likelihood) * (1.0 - prior))
    prior = (likelihood * prior) / evidence

print(f"Posterior after {n_tests} positive tests: {prior}")

# how many positives in a row to be > 99% sure
prior = prior_0
k = 0

while prior <= 0.99:
    k += 1
    likelihood = test_eff
    evidence = (likelihood * prior) + ((1.0 - likelihood) * (1.0 - prior))
    prior = (likelihood * prior) / evidence

print(f"Minimum positive tests in a row for >99%: {k}")
print(f"Posterior then: {prior}")

# Output:
# Problem 2: Test efficacy
# Posterior after 10 positive tests: 0.9999999999999889
# Minimum positive tests in a row for >99%: 5
# Posterior then: 0.9998948575899611

'''
3. Still on the example in §24.3.2, how accurate must the test be so that three
positive tests in a row gives a > 99% probability of having the disease?
'''

import numpy as np

print("\nProblem 3: Accuracy required for 3 positive tests in a row to give a 99% probability of being DISEASED!")
prior = 1e-6
target = 0.99
k = 3  # 3 positive tests

prior_odds = prior / (1 - prior)
target_odds = target / (1 - target)

odds_multiplier_req = (target_odds / prior_odds)**(1 / k)
test_eff = odds_multiplier_req / (1 + odds_multiplier_req)

print(f"Required test accuracy (eff): {test_eff}")

# quick check
p = prior
for _ in range(3):
    evidence = (test_eff * p) + ((1 - test_eff) * (1 - p))
    p = (test_eff * p) / evidence

print(f"Posterior after 3 positives: {p}")

# Output:
# Problem 3: Accuracy required for 3 positive tests in a row to give a 99% probability of being DISEASED!
# Required test accuracy (eff): 0.9978429976054395
# Posterior after 3 positives: 0.9900000000000005

'''
4. (this one is trickier than it sounds, consider Bayes theorem very carefully) Write
a script that evaluates the posterior probability of having the disease after an
arbitrary number of test results, each of which can be positive or negative.
'''

import numpy as np

print("\nProblem 4: Sctipting posterior probability of being diseased after arbitrary number of random results")

test_eff = 0.99
prior_0 = 1e-6

# you get to pick :D
n_tests = int(input("Enter number of tests to simulate: "))

p = prior_0
has_disease = (np.random.rand() < prior_0) # "true" value for this sim
results = ""

for _ in range(n_tests):
    if has_disease:
        positive = (np.random.rand() < test_eff) # true positive w/ prob eff
    else:
        positive = (np.random.rand() < (1 - test_eff)) # false positive w/ prob (1-eff)

    if positive:
        results += "+"
        p_positive = (test_eff * p) + ((1.0 - test_eff) * (1.0 - p))
        p = (test_eff * p) / p_positive
    else:
        results += "-"
        p_negative = ((1.0 - test_eff) * p) + (test_eff * (1.0 - p))
        p = ((1.0 - test_eff) * p) / p_negative

print(f"True disease state (simulated): {has_disease}")
print(f"Random results: {results}")
print(f"Posterior: {p}")

# Output:
# Problem 4: Sctipting posterior probability of being diseased after arbitrary number of random results
# Enter number of tests to simulate: 100
# True disease state (simulated): False
# Random results: ---------------+-----------------------------------------------------------------------+------------
# Posterior: 2.624349988461823e-198

