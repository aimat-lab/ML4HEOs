""" This script operates on the data file "XRD_6_component_systems_repeat.csv" and removes the background.
After this, the ratio of the two highest peaks is calculated.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
from numpy.linalg import norm
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import csv
import os

# baseline_arPLS parameters:
arPLS_ratio = 0.1
arPLS_lam = 10 ** 7
arPLS_niter = 10

# smoothing window width:
window_width = 100

# additional width to the left and right for the gaussian fit:
additional_left_right = 0.7

# parameters for find_peaks:
find_peaks_distance = 0.5
# find_peaks_prominence = 10.5
find_peaks_prominence = 8.4
# find_peaks_height = 6

# This function is from https://stackoverflow.com/questions/29156532/python-baseline-correction-library
def baseline_arPLS(y, ratio=1e-6, lam=100, niter=10, full_output=False):
    L = len(y)

    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2 * diag, diag], [0, -1, -2], L, L - 2)

    H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252

    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)

    crit = 1
    count = 0

    while crit > ratio:
        z = linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]

        m = np.mean(dn)
        s = np.std(dn)

        w_new = 1 / (1 + np.exp(2 * (d - (2 * s - m)) / s))

        crit = norm(w_new - w) / norm(w)

        w = w_new
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values

        count += 1

        if count > niter:
            print("Maximum number of iterations exceeded")
            break

    if full_output:
        info = {"num_iter": count, "stop_criterion": crit}
        return z, d, info
    else:
        return z


def fit_baseline(xs, ys):

    ys_baseline = baseline_arPLS(
        ys, ratio=arPLS_ratio, lam=arPLS_lam, niter=arPLS_niter
    )

    plt.subplot(221)
    plt.plot(xs, ys, rasterized=True)
    plt.plot(xs, ys_baseline, rasterized=True)
    plt.xlabel(r"$ 2 \theta \, / \, ° $")
    plt.ylabel("Intensity")

    plt.title("Raw data with baseline fit")

    return ys_baseline


def gaussian(x, a, x0, sigma):
    return (
        a
        * 1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    )


def fit_gaussian(peak_index, xs, ys, ys_smoothed):

    # find first index where function increases again (to the left and to the right)

    last_entry = ys_smoothed[peak_index]
    for i in range(peak_index + 1, len(ys_smoothed)):
        if ys_smoothed[i] > last_entry:
            if xs[i] - xs[peak_index] > 0.5:
                break
        else:
            last_entry = ys_smoothed[i]
    right = i

    last_entry = ys_smoothed[peak_index]
    for i in reversed(range(0, peak_index)):
        if ys_smoothed[i] > last_entry:
            if xs[peak_index] - xs[i] > 0.5:
                break
        else:
            last_entry = ys_smoothed[i]
    left = i

    step = xs[1] - xs[0]
    additional = int(additional_left_right / step)
    left = np.max([0, left - additional])
    right = np.min([len(xs) - 1, right + additional])

    fit = curve_fit(
        gaussian,
        xs[left : right + 1],
        ys[left : right + 1],
        p0=[100, xs[peak_index], 0.3],
        maxfev=5000,
    )

    return (
        *fit[0],
        left,
        right,
    )  # return the fit parameters + left and right bounds of fitting procedure


def peak_to_str(fit_results, height_raw, height_after_removal):
    return "Area: {}\nMean: {}\nSigma: {}\nFWHM: {}\nHeight raw: {}\nHeight after removal: {}".format(
        str(fit_results[0]),
        str(fit_results[1]),
        str(fit_results[2]),
        str(2 * np.sqrt(2 * np.log(2)) * fit_results[2]),
        height_raw,
        height_after_removal,
    )


def get_real_maximum(xs, ys, index, tollerance_left_right):

    step = xs[1] - xs[0]
    no_steps = int(tollerance_left_right / step)

    ys_current = ys[index - no_steps : index + no_steps + 1]

    true_maximum_index = np.argmax(ys_current)

    return (true_maximum_index + index - no_steps, ys_current[true_maximum_index])


def process_peak(params, index, xs_current, ys_current, ys_baseline_removed):

    real_maximum_raw = get_real_maximum(xs_current, ys_current, index, 0.3)
    real_maximum_baseline_removed = get_real_maximum(
        xs_current, ys_baseline_removed, index, 0.3
    )

    result_text = (
        peak_to_str(params, real_maximum_raw[1], real_maximum_baseline_removed[1],)
        + "\n"
    )

    result_properties = (
        params[0],
        params[1],
        params[2],
        2 * np.sqrt(2 * np.log(2)) * params[2],
        real_maximum_raw[1],
        real_maximum_baseline_removed[1],
        # real_maximum_smoothed[1],
    )

    plt.subplot(223)
    plt.scatter(
        xs_current[index], ys_smoothed[index], c="r", rasterized=True,
    )
    plt.subplot(223)
    # print(zipped_sorted[-1][0])
    plt.scatter(
        [xs_current[params[3]], xs_current[params[4]]],
        [ys_smoothed[params[3]], ys_smoothed[params[4]]],
        c="r",
        rasterized=True,
    )
    plt.subplot(222)
    plt.plot(
        xs_current, gaussian(xs_current, *params[0:3]), rasterized=True,
    )

    return result_text, result_properties


data_file_path = "../data/XRD_6_component_systems_repeat.csv"
df = pd.read_csv(data_file_path, sep=";")

x = np.array(df.iloc[:, 0])
xs = np.repeat(x[:, np.newaxis], len(df.columns.values) - 1, axis=1)
ys = np.array(df.iloc[:, list(range(1, len(df.columns.values)))])

names = df.columns.values[1::]

ratios = []
properties_peak_0 = []  # first peak
properties_peak_1 = []  # second peak

os.system("mkdir -p plots")

for i in range(0, xs.shape[1]):

    fig, ax = plt.subplots(2, 2)

    fig.canvas.manager.set_window_title("Plot {} of {}".format(i + 1, xs.shape[1]))
    plt.suptitle("Sample " + names[i])

    ax[1, 1].set_axis_off()
    fig.set_size_inches(18.5, 10.5)

    print("Processing {} of {}: {}".format(i + 1, xs.shape[1], names[i]))

    xs_current = xs[:, i]

    ys_current = ys[:, i]

    ys_baseline = fit_baseline(xs_current, ys_current)

    ys_baseline_removed = ys_current - ys_baseline

    plt.subplot(222)
    plt.plot(xs, ys_baseline_removed, rasterized=True)
    plt.xlabel(r"$ 2 \theta \, / \, ° $")
    plt.ylabel("Intensity")
    plt.title("Raw data with baseline removed and gauß fits")

    # use moving average smoothing
    ys_smoothed = np.convolve(
        ys_baseline_removed, np.ones(window_width) / window_width, mode="same"
    )
    # ys_smoothed = savgol_filter(ys_baseline_removed, 201, 4)

    (
        peaks,
        props,
    ) = find_peaks(  # TODO: Change these parameters when doing more sophisticated analysis
        ys_smoothed,
        distance=find_peaks_distance / (xs_current[1] - xs_current[0]),
        prominence=find_peaks_prominence,
        # height=find_peaks_height,
    )

    plt.subplot(223)
    plt.plot(xs_current, ys_smoothed, rasterized=True)
    plt.xlabel(r"$ 2 \theta \, / \, ° $")
    plt.ylabel("Intensity")
    plt.title("Smoothed, baseline removed, with marked peaks")

    # plt.show()

    parameters = []
    for peak in peaks:
        para = fit_gaussian(peak, xs_current, ys_baseline_removed, ys_smoothed)
        parameters.append(para)
    zipped_sorted = list(
        reversed(sorted(zip(parameters, peaks), key=lambda x: x[0][1]))
    )  # sort by mean

    near_0 = sorted(zipped_sorted, key=lambda x: np.abs(x[0][1] - 31.26))
    near_1 = sorted(zipped_sorted, key=lambda x: np.abs(x[0][1] - 44.94))

    text = ""
    if (
        len(peaks) > 1
        and np.abs(near_0[0][0][1] - 31.26) < 2
        and np.abs(near_1[0][0][1] - 44.94) < 2
        and len(peaks) < 6
        and not names[i] == "HTA12"  # just noise
    ):

        text = "\nFirst peak:\n"

        result_text, result_properties = process_peak(
            near_0[0][0], near_0[0][1], xs_current, ys_current, ys_baseline_removed,
        )
        text += result_text
        properties_peak_0.append(result_properties)

        text += "\n\nSecond peak:\n"
        result_text, result_properties = process_peak(
            near_1[0][0], near_1[0][1], xs_current, ys_current, ys_baseline_removed,
        )
        text += result_text
        properties_peak_1.append(result_properties)

        text += "\nRatio of the two first peaks:\n"
        ratio = near_0[0][0][0] / near_1[0][0][0]
        text += str(ratio) + "\n"
        ratios.append(ratio)

    else:

        text = "Found less than two peaks."
        properties_peak_0.append(("None", "None", "None", "None", "None", "None"))
        properties_peak_1.append(("None", "None", "None", "None", "None", "None"))
        ratios.append("None")

    ax[1, 1].text(
        0.5, 0.5, text, horizontalalignment="center", verticalalignment="center"
    )

    plt.savefig("plots/" + names[i] + ".pdf", dpi=300)
    # plt.show()

with open("ratios.csv", "w") as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=";")

    # transpose the lists:
    properties_peak_0 = list(map(list, zip(*properties_peak_0)))
    properties_peak_1 = list(map(list, zip(*properties_peak_1)))

    header = [
        "Sample name",
        "Ratio",
        "Peak_0 area",
        "Peak_0 mean",
        "Peak_0 sigma",
        "Peak_0 FWHM",
        "Peak_0 height raw",
        "Peak_0 height after removal",
        "Peak_1 area",
        "Peak_1 mean",
        "Peak_1 sigma",
        "Peak_1 FWHM",
        "Peak_1 height raw",
        "Peak_1 height after removal",
    ]

    # print(names)
    # print(ratios)
    # print(*properties_peak_0)
    # print(*properties_peak_1)

    data = zip(names, ratios, *properties_peak_0, *properties_peak_1)

    csv_writer.writerow(header)
    csv_writer.writerows(data)

# Combine all pdfs into one:
os.system("pdfunite plots/*.pdf plots/all.pdf")
