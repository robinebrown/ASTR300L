'''
1. Using the file of orbital phase and flux measurements, plot the orbital phase light curve similar to the
one above. Label both x and y axes.
'''
import numpy as np
import matplotlib.pyplot as plt

# load csv
phase, flux = np.loadtxt("transit.real.csv", delimiter=",", unpack=True)

# plot
plt.figure(figsize=(8,5))
plt.scatter(phase, flux, s=8)
plt.xlabel("Orbital phase")
plt.ylabel("Flux (photons per 30 s)")
plt.title("Orbital Phase Light Curve")
plt.xlim(0.0, 1.0)
plt.grid(True)

# save/display
plt.savefig("orbital_phase_light_curve.png", bbox_inches="tight", dpi=200)
plt.show()

'''
2. Examine your figure and identify the start and end phase for the transit. [2pts]
'''
import numpy as np
import matplotlib.pyplot as plt

phase, flux = np.loadtxt("transit.real.csv", delimiter=",", unpack=True)

# baseline and scatter
med = np.median(flux)
mad = np.median(np.abs(flux - med)) # median absolute deviation
sigma = 1.4826 * mad # standard deviation estimate

# transit = flux well below baseline
threshold = med - 3.0 * sigma
in_transit = flux < threshold

# start/end phase of transit window
phase_start = phase[in_transit].min()
phase_end = phase[in_transit].max()

print(f"\n2. Transit start/end phase = {phase_start:.2f}, {phase_end:.2f}")

# plt.figure(figsize=(8,5))
# plt.scatter(phase, flux, s=8, color='k', label="Flux measurements")
# plt.axhline(med, linestyle="--", label="Baseline (median)")
# plt.axhline(threshold, linestyle=":", label="Transit threshold (median − 3*sigma)")
# plt.axvspan(phase_start, phase_end, alpha=0.15, label="Transit window")
# plt.xlabel("Orbital phase")
# plt.ylabel("Flux (photons per 30 s)")
# plt.title("Transit Window with Baseline and Threshold")
# plt.xlim(0.0, 1.0)
# plt.grid(True)
# plt.legend()
# plt.show()

'''
3. Measure the average flux (in photons per measurement , not per second) of the star for the portion out of transit.
'''
import numpy as np

phase, flux = np.loadtxt("transit.real.csv", delimiter=",", unpack=True)

# detect transit as before
med = np.median(flux)
mad = np.median(np.abs(flux - med))
sigma = 1.4826 * mad
threshold = med - 3.0 * sigma
in_transit = flux < threshold

# out of transit average flux
out_flux_mean = np.mean(flux[~in_transit])

print(f"\n3. Out-of-transit mean flux (photons per measurement): {out_flux_mean:.2f}")

'''
4. Determine the error on the average out-of-transit flux. [4pts]
'''
import numpy as np

phase, flux = np.loadtxt("transit.real.csv", delimiter=",", unpack=True)

# detect transit as before
med = np.median(flux)
mad = np.median(np.abs(flux - med))
sigma_mad = 1.4826 * mad
threshold = med - 3.0 * sigma_mad
in_transit = flux < threshold

# out-of-transit
out_of_transit = flux[~in_transit]
N_out_of_transit = out_of_transit.size

# error on the mean
s_out_out_transit = np.std(out_of_transit, ddof=1) # sample std
sem_out_of_transit = s_out_out_transit / np.sqrt(N_out_of_transit)

print(f"\n4. Out-of-transit mean flux error: {sem_out_of_transit:.2f}")

'''
5. Measure the average flux of the star for the portion in the transit. [2pts]
'''
import numpy as np

phase, flux = np.loadtxt("transit.real.csv", delimiter=",", unpack=True)

# detect transit as before
med = np.median(flux)
mad = np.median(np.abs(flux - med))
sigma_mad = 1.4826 * mad
threshold = med - 3.0 * sigma_mad
in_transit = flux < threshold

# in-transit average flux
in_flux_mean = np.mean(flux[in_transit])

print(f"\n5. In-transit mean flux (photons per measurement): {in_flux_mean:.2f}")

'''
6. Determine the error on the average in-transit flux.
'''
import numpy as np

phase, flux = np.loadtxt("transit.real.csv", delimiter=",", unpack=True)

# detect transit as before
med = np.median(flux)
mad = np.median(np.abs(flux - med))
sigma_mad = 1.4826 * mad
threshold = med - 3.0 * sigma_mad
in_transit = flux < threshold

# error on in-transit mean
flux_in_transit = flux[in_transit]
s_in_transit = np.std(flux_in_transit, ddof=1)
sem_in_transit = s_in_transit / np.sqrt(flux_in_transit.size)

print(f"\n6. In-transit mean flux error: {sem_in_transit:.2f}")

'''
7. Calculate the transit depth delta F and the error on delta F.
'''
import numpy as np

# Load data
phase, flux = np.loadtxt("transit.real.csv", delimiter=",", unpack=True)

# detect transit as before
med = np.median(flux)
mad = np.median(np.abs(flux - med))
sigma_mad = 1.4826 * mad
threshold = med - 3.0 * sigma_mad
in_transit = flux < threshold

# means
flux_out_transit = flux[~in_transit]
flux_in_transit  = flux[in_transit]
mean_out_transit = np.mean(flux_out_transit)
mean_in_transit  = np.mean(flux_in_transit)

# errors on means
s_out_transit   = np.std(flux_out_transit, ddof=1)
s_in_transit    = np.std(flux_in_transit,  ddof=1)
sem_out_transit = s_out_transit / np.sqrt(flux_out_transit.size)
sem_in_transit  = s_in_transit  / np.sqrt(flux_in_transit.size)

# transit depth and error
delta_F = mean_out_transit - mean_in_transit
sigma_delta_F = np.sqrt(sem_out_transit**2 + sem_in_transit**2)

print(f"\n7. Transit depth delta F: {delta_F:.2f}, Error on delta F: {sigma_delta_F:.2f}")

'''
8. Calculate the fractional depth 𝞺 = delta F / F and the error on 𝞺.
'''
import numpy as np

phase, flux = np.loadtxt("transit.real.csv", delimiter=",", unpack=True)

# detect transit as before
med = np.median(flux)
mad = np.median(np.abs(flux - med))
sigma_mad = 1.4826 * mad
threshold = med - 3.0 * sigma_mad
in_transit = flux < threshold

# means
flux_out_transit = flux[~in_transit]
flux_in_transit  = flux[in_transit]
mean_out_transit = np.mean(flux_out_transit)
mean_in_transit  = np.mean(flux_in_transit)

# errors on means
s_out_transit   = np.std(flux_out_transit, ddof=1)
s_in_transit    = np.std(flux_in_transit,  ddof=1)
sem_out_transit = s_out_transit / np.sqrt(flux_out_transit.size)
sem_in_transit  = s_in_transit  / np.sqrt(flux_in_transit.size)

# transit depth and error
delta_F = mean_out_transit - mean_in_transit
sigma_delta_F = np.sqrt(sem_out_transit**2 + sem_in_transit**2)

# fractional depth rho and error
rho = delta_F / mean_out_transit
sigma_rho = np.sqrt( (sigma_delta_F / mean_out_transit)**2 +(delta_F * sem_out_transit / (mean_out_transit**2))**2 )

print(f"\n8. Fractional depth ρ: {rho:.6f}, Error on ρ: {sigma_rho:.6f}")

'''
9. What is the planet radius, R p as a fraction of the stellar radius R s .
'''
import numpy as np

phase, flux = np.loadtxt("transit.real.csv", delimiter=",", unpack=True)

# detect transit as before
med = np.median(flux)
mad = np.median(np.abs(flux - med))
sigma_mad = 1.4826 * mad
threshold = med - 3.0 * sigma_mad
in_transit = flux < threshold

# means
flux_out_transit = flux[~in_transit]
flux_in_transit  = flux[in_transit]
mean_out_transit = np.mean(flux_out_transit)
mean_in_transit  = np.mean(flux_in_transit)

# transit and fractional depth
delta_F = mean_out_transit - mean_in_transit
rho = delta_F / mean_out_transit

# planet-to-star radius ratio
rp_over_rs = np.sqrt(rho)

print(f"\n9. Planet radius/ star radius: {rp_over_rs:.6f}")

'''
10. What is the error on the planet radius?
'''
import numpy as np

# Load data
phase, flux = np.loadtxt("transit.real.csv", delimiter=",", unpack=True)

# detect transit as before
med = np.median(flux)
mad = np.median(np.abs(flux - med))
sigma_mad = 1.4826 * mad
threshold = med - 3.0 * sigma_mad
in_transit = flux < threshold

# means
flux_out_transit = flux[~in_transit]
flux_in_transit  = flux[in_transit]
mean_out_transit = np.mean(flux_out_transit)
mean_in_transit  = np.mean(flux_in_transit)

# errors on means
s_out_transit   = np.std(flux_out_transit, ddof=1)
s_in_transit    = np.std(flux_in_transit,  ddof=1)
sem_out_transit = s_out_transit / np.sqrt(flux_out_transit.size)
sem_in_transit  = s_in_transit  / np.sqrt(flux_in_transit.size)

# transit depth error
delta_F = mean_out_transit - mean_in_transit
sigma_delta_F = np.sqrt(sem_out_transit**2 + sem_in_transit**2)

# fractional depth error
rho = delta_F / mean_out_transit
sigma_rho = np.sqrt( (sigma_delta_F / mean_out_transit)**2 +(delta_F * sem_out_transit / (mean_out_transit**2))**2 )

# planet radius / star radius and error
rp_over_rs = np.sqrt(rho)
sigma_rp_over_rs = (0.5 / np.sqrt(rho)) * sigma_rho

print(f"\n10. Planet radius / Star radius: {rp_over_rs:.6f}, Error on planet radius / star radius: {sigma_rp_over_rs:.6f}")

