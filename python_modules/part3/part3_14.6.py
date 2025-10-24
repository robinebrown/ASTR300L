'''
1.
'''
from astropy.io import fits
from astropy.time import Time

fn = "Data/StellarPhotometry/coj1m011-kb05-20140607-0113-e90.fits"

hdul = fits.open(fn)
hdr  = hdul[0].header

date_obs = hdr.get("DATE-OBS")
time_obs = hdr.get("TIME-OBS")
mjd_obs  = hdr.get("MJD-OBS")
exptime  = hdr.get("EXPTIME")
filt     = hdr.get("FILTER")
tel      = hdr.get("TELESCOP")
inst     = hdr.get("INSTRUME")
obj      = hdr.get("OBJECT")
ra_hdr   = hdr.get("RA") or hdr.get("OBJCTRA")
dec_hdr  = hdr.get("DEC") or hdr.get("OBJCTDEC")

if date_obs and "T" in date_obs:
    t = Time(date_obs, format="isot", scale="utc")
elif date_obs and time_obs:
    t = Time(f"{date_obs}T{time_obs}", format="isot", scale="utc")
else:
    t = Time(mjd_obs, format="mjd", scale="utc") if mjd_obs is not None else None

print("\n--- Header summary ---")
print(f"Object    : {obj}")
print(f"Date/Time : {t.isot if t else 'N/A'} (UTC)")
print(f"MJD-OBS   : {mjd_obs}")
print(f"EXPTIME   : {exptime} s")
print(f"FILTER    : {filt}")
print(f"Telescope : {tel}")
print(f"Instrument: {inst}")
print(f"RA (hdr)  : {ra_hdr}")
print(f"Dec (hdr) : {dec_hdr}")

'''
2. 
'''
from astropy.io import fits

# Problem 2: copy data and header into their own variables
fn = "Data/StellarPhotometry/coj1m011-kb05-20140607-0113-e90.fits"

image, header = fits.getdata(fn, header=True)

'''
3.
'''
from astropy.io import fits
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
import matplotlib.pyplot as plt

# Problem 3: Display image with appropriate scaling
fn = "Data/StellarPhotometry/coj1m011-kb05-20140607-0113-e90.fits"

data, hdr = fits.getdata(fn, header=True)

norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())

plt.figure()
plt.imshow(data, origin="lower", cmap="gray", norm=norm)
plt.colorbar(label="Counts")
plt.title(f"{hdr.get('OBJECT','')}  |  {hdr.get('FILTER','')}  |  t={hdr.get('EXPTIME','')}s")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.tight_layout()
plt.show()

'''
4.
'''
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
import astropy.units as u
import matplotlib.pyplot as plt

# Problem 4: Identify the star using WCS (world→pixel)
fn = "Data/StellarPhotometry/coj1m011-kb05-20140607-0113-e90.fits"

data, hdr = fits.getdata(fn, header=True)
wcs = WCS(hdr)

# Target coordinates given in the exercise
target = SkyCoord("16h14m20.3s -19d06m48.1s", frame="icrs")

# Convert to pixel coordinates (returns x, y in FITS convention for imshow with origin="lower")
xp, yp = wcs.world_to_pixel(target)

# Display with sensible scaling and overlay a crosshair
norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())

plt.figure()
plt.imshow(data, origin="lower", cmap="gray", norm=norm)
plt.scatter([xp], [yp], s=120, marker="+")  # crosshair at target
plt.title(f"{hdr.get('OBJECT','')}  |  RA/Dec → pix: ({xp:.1f}, {yp:.1f})")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.show()

print(f"Target pixel coordinates: x={xp:.2f}, y={yp:.2f}")

'''
5.
'''
from astropy.io import fits
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
from astropy.stats import SigmaClip
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Problem 5: Choose a sky region and estimate background
fn = "Data/StellarPhotometry/coj1m011-kb05-20140607-0113-e90.fits"
data, hdr = fits.getdata(fn, header=True)

# --- EDIT THESE four numbers after you inspect the image ---
# Pixel-box [x0:x1, y0:y1] that is free of stars/cosmic rays (make it, say, ~30–80 px on a side)
x0, x1 = 50, 130   # columns
y0, y1 = 50, 130   # rows
# -----------------------------------------------------------

sky_box = data[y0:y1, x0:x1]

# Robust background estimate (clip outliers like faint stars/cosmics)
sigclip = SigmaClip(sigma=3.0, maxiters=5)
sky_clipped = sigclip(sky_box)

bkg_median = np.nanmedian(sky_clipped)
bkg_mean   = np.nanmean(sky_clipped)
bkg_std    = np.nanstd(sky_clipped)  # ~“background RMS”
n_pix      = np.sum(~np.isnan(sky_clipped))

print("--- Sky background summary ---")
print(f"Box: x[{x0}:{x1}], y[{y0}:{y1}]  -> {sky_box.shape[1]}×{sky_box.shape[0]} px, {n_pix} used after clipping")
print(f"Median background  : {bkg_median:.3f} counts/pixel")
print(f"Mean background    : {bkg_mean:.3f} counts/pixel")
print(f"Background RMS (σ) : {bkg_std:.3f} counts/pixel")

# Visual confirmation of the chosen sky box
norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())
fig, ax = plt.subplots()
ax.imshow(data, origin="lower", cmap="gray", norm=norm)
ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                       fill=False, linewidth=1.5))
ax.set_title("Chosen sky box (no stars)")
ax.set_xlabel("X (pixels)")
ax.set_ylabel("Y (pixels)")
plt.tight_layout()
plt.show()

'''
6.
'''
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

# Problem 6: Define a star region (aperture) and compute total counts in it
fn = "Data/StellarPhotometry/coj1m011-kb05-20140607-0113-e90.fits"

data, hdr = fits.getdata(fn, header=True)
wcs = WCS(hdr)

# Target coordinates from the prompt
target = SkyCoord("16h14m20.3s -19d06m48.1s", frame="icrs")
xc, yc = wcs.world_to_pixel(target)   # floating-point pixel center

# --- Adjust this aperture radius after inspecting the image ---
r = 6.0  # pixels
# --------------------------------------------------------------

# Build a circular mask around (xc, yc)
ny, nx = data.shape
y, x = np.ogrid[:ny, :nx]
mask = (x - xc)**2 + (y - yc)**2 <= r**2

total_counts = data[mask].sum()

print(f"Aperture center (pix): x={xc:.2f}, y={yc:.2f}")
print(f"Aperture radius      : r={r:.1f} px")
print(f"Total counts in region (no bkg sub): {total_counts:.2f}")

# Quick visualization with sensible scaling and the aperture outline
norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())

theta = np.linspace(0, 2*np.pi, 180)
ap_x = xc + r*np.cos(theta)
ap_y = yc + r*np.sin(theta)

plt.figure()
plt.imshow(data, origin="lower", cmap="gray", norm=norm)
plt.plot(ap_x, ap_y)  # aperture circle
plt.scatter([xc], [yc], s=60, marker="+")  # center marker
plt.title(f"{hdr.get('OBJECT','')}  |  aperture r={r:.1f}px")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.tight_layout()
plt.show()

'''
7.
'''
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
import astropy.units as u
import numpy as np

# Problem 7: Convert counts → electrons using GAIN (e-/ADU)
fn = "Data/StellarPhotometry/coj1m011-kb05-20140607-0113-e90.fits"

data, hdr = fits.getdata(fn, header=True)
wcs = WCS(hdr)

# Target coordinates (ICRS) from the exercise
target = SkyCoord("16h14m20.3s -19d06m48.1s", frame="icrs")
xc, yc = wcs.world_to_pixel(target)

# --- Adjust aperture radius to your image/seeing ---
r = 6.0  # pixels
# ---------------------------------------------------

ny, nx = data.shape
y, x = np.ogrid[:ny, :nx]
mask = (x - xc)**2 + (y - yc)**2 <= r**2

# Total counts (ADU) inside aperture (no background subtraction here)
total_counts_adu = np.sum(data[mask])

# Pull gain from header (typical keywords)
gain = (hdr.get("GAIN") or hdr.get("EGAIN") or hdr.get("GAIN1"))
if gain is None:
    raise KeyError("No GAIN keyword found (tried GAIN, EGAIN, GAIN1). Check this file's header.")

# Convert counts (ADU) → electrons
total_electrons = total_counts_adu * float(gain)

# Shot noise (Poisson) in electrons
shot_noise_e = np.sqrt(total_electrons)

print("--- Counts → Electrons ---")
print(f"GAIN (e-/ADU): {gain}")
print(f"Aperture center (pix): x={xc:.2f}, y={yc:.2f}   radius={r:.1f} px")
print(f"Total in aperture: {total_counts_adu:.2f} ADU  →  {total_electrons:.2f} e-")
print(f"Poisson shot noise: {shot_noise_e:.2f} e-")

'''
8.
'''
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip
import numpy as np

# Problem 8: background-subtracted aperture counts
fn = "Data/StellarPhotometry/coj1m011-kb05-20140607-0113-e90.fits"

data, hdr = fits.getdata(fn, header=True)
wcs = WCS(hdr)

# --- Target aperture (adjust radius r if needed) ---
target = SkyCoord("16h14m20.3s -19d06m48.1s", frame="icrs")
xc, yc = wcs.world_to_pixel(target)
r = 6.0  # pixels

ny, nx = data.shape
y, x = np.ogrid[:ny, :nx]
ap_mask = (x - xc)**2 + (y - yc)**2 <= r**2
n_ap = int(ap_mask.sum())

total_counts_adu = data[ap_mask].sum()

# --- Sky region (edit these bounds to a clean, source-free patch) ---
x0, x1 = 50, 130   # columns
y0, y1 = 50, 130   # rows
sky_box = data[y0:y1, x0:x1]

# Robust mean sky per pixel (sigma-clipped)
sigclip = SigmaClip(sigma=3.0, maxiters=5)
sky_clipped = sigclip(sky_box)
mean_sky_per_pix = np.nanmean(sky_clipped)

# Background to subtract from the aperture
bkg_in_ap = mean_sky_per_pix * n_ap
net_counts_adu = total_counts_adu - bkg_in_ap

print("--- Background subtraction ---")
print(f"Aperture center (pix): x={xc:.2f}, y={yc:.2f}   r={r:.1f} px")
print(f"Aperture pixels      : {n_ap}")
print(f"Mean sky / pixel     : {mean_sky_per_pix:.3f} ADU")
print(f"Total in aperture    : {total_counts_adu:.2f} ADU")
print(f"Sky in aperture      : {bkg_in_ap:.2f} ADU")
print(f"Net (source) counts  : {net_counts_adu:.2f} ADU")

'''
9.
'''
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip
import numpy as np

# Problem 9: Uncertainty on background-subtracted source electrons

fn = "Data/StellarPhotometry/coj1m011-kb05-20140607-0113-e90.fits"

data, hdr = fits.getdata(fn, header=True)
wcs = WCS(hdr)

# ---- target aperture (adjust r after inspecting the image) ----
target = SkyCoord("16h14m20.3s -19d06m48.1s", frame="icrs")
xc, yc = wcs.world_to_pixel(target)
r = 6.0  # pixels

ny, nx = data.shape
y, x = np.ogrid[:ny, :nx]
ap_mask = (x - xc)**2 + (y - yc)**2 <= r**2
n_ap = int(ap_mask.sum())

# raw aperture sum (ADU)
ap_sum_adu = data[ap_mask].sum()

# ---- sky box (edit to a clean, source-free patch) ----
x0, x1 = 50, 130
y0, y1 = 50, 130
sky_box = data[y0:y1, x0:x1]

# robust sky stats (per pixel, ADU)
sigclip = SigmaClip(sigma=3.0, maxiters=5)
sky_clip = sigclip(sky_box)
mean_sky_adu = np.nanmean(sky_clip)          # background level per pixel
std_sky_adu  = np.nanstd(sky_clip, ddof=1)   # background RMS per pixel
N_sky        = int(np.sum(~np.isnan(sky_clip)))

# background in aperture (ADU) and net source (ADU)
bkg_in_ap_adu = mean_sky_adu * n_ap
net_adu       = ap_sum_adu - bkg_in_ap_adu

# ---- counts → electrons (≈ photons) ----
gain = (hdr.get("GAIN") or hdr.get("EGAIN") or hdr.get("GAIN1"))
if gain is None:
    raise KeyError("No GAIN keyword found (tried GAIN, EGAIN, GAIN1).")
gain = float(gain)

net_e        = net_adu * gain
std_sky_e    = std_sky_adu * gain            # per-pixel background RMS in electrons
mean_sky_se_adu = std_sky_adu / np.sqrt(N_sky) if N_sky > 0 else np.nan
mean_sky_se_e   = mean_sky_se_adu * gain     # standard error of the mean sky (per pixel)

# Optional read noise (e- per pixel). Common header keys shown.
rdnoise = hdr.get("RDNOISE") or hdr.get("READNOIS") or hdr.get("RON")
rdnoise_e = float(rdnoise) if rdnoise is not None else 0.0

# ---- Uncertainty components (electrons^2) ----
# 1) Shot noise of the source (Poisson on net electrons; clamp at 0 if tiny/negative)
var_source_shot = max(net_e, 0.0)

# 2) Pixel-to-pixel background noise inside the aperture
#    (sum of n_ap independent background pixels)
var_bkg_pixels = n_ap * (std_sky_e ** 2)

# 3) Uncertainty from subtracting the MEAN background measured from N_sky pixels
#    (we subtract n_ap * mean_sky; variance adds as (n_ap * SE)^2 )
var_bkg_mean_sub = (n_ap ** 2) * (mean_sky_se_e ** 2) if N_sky > 0 else 0.0

# 4) Read noise inside the aperture (if available), per pixel
var_read = n_ap * (rdnoise_e ** 2)

# Total variance and 1-sigma uncertainty (electrons)
var_tot_e = var_source_shot + var_bkg_pixels + var_bkg_mean_sub + var_read
sigma_e   = np.sqrt(var_tot_e)

print("--- Photometry & Uncertainty ---")
print(f"Aperture center (pix): x={xc:.2f}, y={yc:.2f}   r={r:.1f} px   pixels in ap: {n_ap}")
print(f"Sky box: x[{x0}:{x1}], y[{y0}:{y1}]   N_sky used: {N_sky}")
print(f"Mean sky per pixel: {mean_sky_adu:.3f} ADU   RMS: {std_sky_adu:.3f} ADU")
print(f"GAIN: {gain:.3f} e-/ADU    Read noise: {rdnoise_e:.2f} e- per pix")
print(f"Raw ap sum: {ap_sum_adu:.2f} ADU   Sky in ap: {bkg_in_ap_adu:.2f} ADU")
print(f"Net source: {net_adu:.2f} ADU  ->  {net_e:.2f} e-")

print("\nUncertainty components (e-):")
print(f"  Source shot noise      : sqrt({var_source_shot:.2f}) = {np.sqrt(var_source_shot):.2f}")
print(f"  Bkg pixel RMS in ap    : sqrt({var_bkg_pixels:.2f}) = {np.sqrt(var_bkg_pixels):.2f}")
print(f"  Mean-sky subtraction   : sqrt({var_bkg_mean_sub:.2f}) = {np.sqrt(var_bkg_mean_sub):.2f}")
print(f"  Read noise in ap       : sqrt({var_read:.2f}) = {np.sqrt(var_read):.2f}")

print(f"\nTotal 1σ uncertainty     : {sigma_e:.2f} e-")

'''
10.
'''
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip
import astropy.units as u
import numpy as np

# Problem 10: r-mag of J1614-1906 via standard star (r = 13.50)

fn = "Data/StellarPhotometry/coj1m011-kb05-20140607-0113-e90.fits"
data, hdr = fits.getdata(fn, header=True)
wcs = WCS(hdr)

# Known comparison star (constant r = 13.50)
m_comp = 13.50
coord_comp   = SkyCoord("16h14m20.912s -19d06m04.70s", frame="icrs")
coord_target = SkyCoord("16h14m20.3s   -19d06m48.1s",  frame="icrs")

# Same aperture for both stars (adjust after inspecting image)
r = 6.0  # pixels

# Choose a clean sky box (adjust to your image)
x0, x1 = 50, 130   # columns
y0, y1 = 50, 130   # rows

# ---- helpers ----
def aperture_sum_electrons(img, xc, yc, rpx, gain_e_per_adu, rdnoise_e=0.0,
                           sky_box=None):
    """Return (net_e, sigma_e, n_ap, mean_sky_adu, std_sky_adu)."""
    ny, nx = img.shape
    y, x = np.ogrid[:ny, :nx]
    ap_mask = (x - xc)**2 + (y - yc)**2 <= rpx**2
    n_ap = int(ap_mask.sum())

    ap_sum_adu = img[ap_mask].sum()

    # Sky stats from provided box (sigma-clipped)
    sky = img[sky_box] if isinstance(sky_box, tuple) else img
    sigclip = SigmaClip(sigma=3.0, maxiters=5)
    sky_clip = sigclip(sky)
    mean_sky_adu = np.nanmean(sky_clip)
    std_sky_adu  = np.nanstd(sky_clip, ddof=1)
    N_sky        = int(np.sum(~np.isnan(sky_clip)))

    # Background subtraction (ADU)
    bkg_in_ap_adu = mean_sky_adu * n_ap
    net_adu       = ap_sum_adu - bkg_in_ap_adu

    # Convert to electrons
    g  = float(gain_e_per_adu)
    net_e       = net_adu * g
    std_sky_e   = std_sky_adu * g
    se_mean_sky_e = (std_sky_adu / np.sqrt(N_sky) * g) if N_sky > 0 else 0.0

    # Variance terms (electrons^2)
    var_source_shot  = max(net_e, 0.0)                    # Poisson on net source
    var_bkg_pixels   = n_ap * (std_sky_e ** 2)            # pixel-to-pixel sky noise
    var_bkg_mean_sub = (n_ap ** 2) * (se_mean_sky_e ** 2) # subtracting mean sky
    var_read         = n_ap * (float(rdnoise_e) ** 2)     # read noise (if available)

    sigma_e = np.sqrt(var_source_shot + var_bkg_pixels + var_bkg_mean_sub + var_read)
    return net_e, sigma_e, n_ap, mean_sky_adu, std_sky_adu

# Header values
gain = (hdr.get("GAIN") or hdr.get("EGAIN") or hdr.get("GAIN1"))
if gain is None:
    raise KeyError("No GAIN keyword found (tried GAIN, EGAIN, GAIN1).")
rdnoise = hdr.get("RDNOISE") or hdr.get("READNOIS") or hdr.get("RON") or 0.0

# Pixel positions
xc_t, yc_t = wcs.world_to_pixel(coord_target)
xc_c, yc_c = wcs.world_to_pixel(coord_comp)

# Define sky-box tuple indexer
sky_box = (slice(y0, y1), slice(x0, x1))

# Target photometry
Ft_e, sFt_e, n_ap_t, mean_sky_t, std_sky_t = aperture_sum_electrons(
    data, xc_t, yc_t, r, gain, rdnoise, sky_box
)

# Comparison star photometry
Fc_e, sFc_e, n_ap_c, mean_sky_c, std_sky_c = aperture_sum_electrons(
    data, xc_c, yc_c, r, gain, rdnoise, sky_box
)

# Differential magnitude: m_t = m_c - 2.5 log10(F_t / F_c)
mt = m_comp - 2.5 * np.log10(Ft_e / Fc_e)

# Uncertainty propagation: σ_m = (2.5/ln10) * sqrt( (σFt/Ft)^2 + (σFc/Fc)^2 )
fac = 2.5 / np.log(10.0)
sigma_mt = fac * np.sqrt( (sFt_e / Ft_e)**2 + (sFc_e / Fc_e)**2 )

print("--- Differential Photometry (r-band) ---")
print(f"Aperture radius: {r:.1f} px   Sky box: x[{x0}:{x1}], y[{y0}:{y1}]")
print(f"GAIN: {float(gain):.3f} e-/ADU   Read noise: {float(rdnoise):.2f} e-/px")
print(f"Target  net: {Ft_e:.1f} e-  ± {sFt_e:.1f} e-")
print(f"Comp    net: {Fc_e:.1f} e-  ± {sFc_e:.1f} e-   (m_r = {m_comp:.2f})")
print(f"\nJ1614-1906 r-magnitude:  m_r = {mt:.3f} ± {sigma_mt:.3f}")

'''
11.
'''
# Problem 11: Batch differential photometry across all images in TheStar/
#
# Output per image:
# - ISO timestamp (UTC)
# - target net electrons ± sigma
# - comp   net electrons ± sigma
# - r-band magnitude of target ± sigma_m
#
# Assumes all images are the same filter (r/rp). Uses WCS + local centroid refine.

from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip
import astropy.units as u
import numpy as np
import glob
import os

# ---------- Configuration ----------
# Paths
PATTERN = "Data/StellarPhotometry/TheStar/*.fits"

# Catalog positions (ICRS)
coord_target = SkyCoord("16h14m20.3s  -19d06m48.1s", frame="icrs")
coord_comp   = SkyCoord("16h14m20.912s -19d06m04.70s", frame="icrs")

# Comparison star known magnitude (r-band)
m_comp = 13.50

# Aperture and sky annulus (pixels) — adjust if your seeing/PSF differs
r_ap   = 6.0      # source aperture radius
r_in   = 10.0     # inner radius of sky annulus
r_out  = 15.0     # outer radius of sky annulus

# Local recentering around WCS-predicted position:
# half-size of the square used to compute a brightness-weighted centroid
recenter_halfwin = 8  # pixels

# Sigma-clip for sky pixels in annulus
sigclip = SigmaClip(sigma=3.0, maxiters=5)
# -----------------------------------

def _timestamp_from_header(hdr):
    date_obs = hdr.get("DATE-OBS")
    time_obs = hdr.get("TIME-OBS")
    mjd_obs  = hdr.get("MJD-OBS")
    if date_obs and "T" in date_obs:
        t = Time(date_obs, format="isot", scale="utc")
    elif date_obs and time_obs:
        t = Time(f"{date_obs}T{time_obs}", format="isot", scale="utc")
    elif mjd_obs is not None:
        t = Time(float(mjd_obs), format="mjd", scale="utc")
    else:
        return "N/A"
    return t.isot

def _circle_mask(shape, xc, yc, r):
    ny, nx = shape
    y, x = np.ogrid[:ny, :nx]
    return (x - xc)**2 + (y - yc)**2 <= r**2

def _annulus_mask(shape, xc, yc, r_in, r_out):
    ny, nx = shape
    y, x = np.ogrid[:ny, :nx]
    rsq = (x - xc)**2 + (y - yc)**2
    return (rsq >= r_in**2) & (rsq <= r_out**2)

def _recenter_centroid(img, x0, y0, halfwin):
    """Brightness-weighted centroid in a small window around (x0,y0)."""
    ny, nx = img.shape
    x0i, y0i = int(round(x0)), int(round(y0))
    x_min = max(0, x0i - halfwin)
    x_max = min(nx, x0i + halfwin + 1)
    y_min = max(0, y0i - halfwin)
    y_max = min(ny, y0i + halfwin + 1)
    cut = img[y_min:y_max, x_min:x_max]
    if cut.size == 0:
        return x0, y0  # fallback
    # subtract local median to reduce bias from flat background
    med = np.median(cut)
    w = np.clip(cut - med, a_min=0, a_max=None)
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    wsum = w.sum()
    if wsum <= 0:
        return x0, y0
    xc = (w * xx).sum() / wsum
    yc = (w * yy).sum() / wsum
    return xc, yc

def aperture_photometry_with_annulus(img, xc, yc, r_src, r_in, r_out, gain_e_per_adu, rdnoise_e=0.0):
    """
    Returns:
      net_e, sigma_e, n_ap, mean_sky_adu, std_sky_adu
    """
    # Aperture mask
    ap_mask = _circle_mask(img.shape, xc, yc, r_src)
    n_ap = int(ap_mask.sum())
    ap_sum_adu = img[ap_mask].sum()

    # Annulus sky
    ann_mask = _annulus_mask(img.shape, xc, yc, r_in, r_out)
    sky_vals = img[ann_mask]
    sky_clip = sigclip(sky_vals)
    mean_sky_adu = np.nanmean(sky_clip)
    std_sky_adu  = np.nanstd(sky_clip, ddof=1)
    N_sky        = int(np.sum(~np.isnan(sky_clip)))

    # Background subtraction in ADU
    bkg_in_ap_adu = mean_sky_adu * n_ap
    net_adu       = ap_sum_adu - bkg_in_ap_adu

    # Convert to electrons
    g = float(gain_e_per_adu)
    net_e      = net_adu * g
    std_sky_e  = std_sky_adu * g
    se_mean_e  = (std_sky_adu / np.sqrt(N_sky) * g) if N_sky > 0 else 0.0

    # Variance terms (electrons^2)
    var_source_shot  = max(net_e, 0.0)                 # Poisson on net source
    var_bkg_pixels   = n_ap * (std_sky_e ** 2)         # pixel-to-pixel sky noise inside aperture
    var_bkg_mean_sub = (n_ap ** 2) * (se_mean_e ** 2)  # uncertainty in mean-sky subtraction
    var_read         = n_ap * (float(rdnoise_e) ** 2)  # read noise (if known)

    sigma_e = np.sqrt(var_source_shot + var_bkg_pixels + var_bkg_mean_sub + var_read)
    return net_e, sigma_e, n_ap, mean_sky_adu, std_sky_adu

# Collect all files (sorted for readability)
files = sorted(glob.glob(PATTERN))

rows = []
for fn in files:
    data, hdr = fits.getdata(fn, header=True)
    w = WCS(hdr)

    # Gain / read noise
    gain = (hdr.get("GAIN") or hdr.get("EGAIN") or hdr.get("GAIN1"))
    if gain is None:
        raise KeyError(f"{os.path.basename(fn)}: No GAIN keyword (tried GAIN/EGAIN/GAIN1).")
    rdnoise = hdr.get("RDNOISE") or hdr.get("READNOIS") or hdr.get("RON") or 0.0

    # Timestamp (UTC)
    ts = _timestamp_from_header(hdr)

    # WCS → pixel, then local recentering
    xt0, yt0 = w.world_to_pixel(coord_target)
    xc0, yc0 = w.world_to_pixel(coord_comp)

    xt, yt = _recenter_centroid(data, xt0, yt0, recenter_halfwin)
    xc, yc = _recenter_centroid(data, xc0, yc0, recenter_halfwin)

    # Photometry
    Ft_e, sFt_e, napt, mean_sky_t, std_sky_t = aperture_photometry_with_annulus(
        data, xt, yt, r_ap, r_in, r_out, gain, rdnoise
    )
    Fc_e, sFc_e, napc, mean_sky_c, std_sky_c = aperture_photometry_with_annulus(
        data, xc, yc, r_ap, r_in, r_out, gain, rdnoise
    )

    # Differential magnitude and uncertainty:
    # m_t = m_c - 2.5 log10(Ft/Fc)
    mt = m_comp - 2.5 * np.log10(Ft_e / Fc_e)
    fac = 2.5 / np.log(10.0)
    sigma_mt = fac * np.sqrt((sFt_e / Ft_e)**2 + (sFc_e / Fc_e)**2)

    rows.append({
        "file": os.path.basename(fn),
        "timestamp_utc": ts,
        "target_x": xt, "target_y": yt,
        "comp_x": xc, "comp_y": yc,
        "Ft_e": Ft_e, "sFt_e": sFt_e,
        "Fc_e": Fc_e, "sFc_e": sFc_e,
        "m_r_target": mt, "sigma_m": sigma_mt,
        "gain_e_per_adu": float(gain),
        "rdnoise_e": float(rdnoise),
        "r_ap": r_ap, "r_in": r_in, "r_out": r_out
    })

# Pretty print results
print("\n=== Differential Photometry Results (r-band) ===")
for r in rows:
    print(f"{r['file']}")
    print(f"  time (UTC) : {r['timestamp_utc']}")
    print(f"  target pix : ({r['target_x']:.2f}, {r['target_y']:.2f})   comp pix: ({r['comp_x']:.2f}, {r['comp_y']:.2f})")
    print(f"  Ft (e-)    : {r['Ft_e']:.1f} ± {r['sFt_e']:.1f}    Fc (e-): {r['Fc_e']:.1f} ± {r['sFc_e']:.1f}")
    print(f"  m_r(target): {r['m_r_target']:.3f} ± {r['sigma_m']:.3f}")
    print("")

# Optional: quick summary line you can paste into your notes
print("file,timestamp_utc,m_r_target,sigma_m")
for r in rows:
    print(f"{r['file']},{r['timestamp_utc']},{r['m_r_target']:.4f},{r['sigma_m']:.4f}")

