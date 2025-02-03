import matplotlib.pyplot as plt
import math
import numpy as np
import collections
import pandas as pd
from scipy.signal import find_peaks
import pywt

#Load file and initialize
csi_file = 'c:\\Test-Files\\first-attempts\\single-action\\a-punch-test.csv'
init_df = pd.read_csv(csi_file)
init_df.info()
amplitudes_list = []
df_amplitudes = []

#Calculate phase and amplitude of csi in file
def process(csi_data):
    csi_data = csi_data.split(",")
    csi_data = csi_data[25].split(" ")
    csi_data[0] = csi_data[0].replace("[", "")
    csi_data[-1] = csi_data[-1].replace("]", "")
    csi_data.pop()
    csi_data = [int(c) for c in csi_data if c]

    imaginary = []
    real = []
    for i, val in enumerate(csi_data):
        if i % 2 == 0:
            imaginary.append(val)
        else:
            real.append(val)

    csi_size = len(csi_data)
    amplitudes = []
    phases = []
    if len(imaginary) > 0 and len(real) > 0:
        for j in range(int(csi_size / 2)):
            amplitude_calc = math.sqrt(imaginary[j] ** 2 + real[j] ** 2)
            phase_calc = math.atan2(imaginary[j], real[j])
            amplitudes.append(amplitude_calc)
            phases.append(phase_calc)

    return amplitudes

#CSI data
with open(csi_file, "r") as f:
    for line in f:
        if "CSI_DATA" in line:
            amplitudes = process(line)
            amplitudes_list.append(amplitudes)

df_amplitudes = pd.DataFrame(amplitudes_list, columns=["Subcarrier{}".format(i) for i in range(64)])
subcarriers_to_drop = ['Subcarrier0', 'Subcarrier1', 'Subcarrier2', 'Subcarrier3', 'Subcarrier4', 'Subcarrier5', 'Subcarrier32', 'Subcarrier59', 'Subcarrier60', 'Subcarrier61', 'Subcarrier62', 'Subcarrier63']
df_amplitudes = df_amplitudes.dropna(how='all')
df_amplitudes = df_amplitudes.drop(columns=subcarriers_to_drop)

#discrete wavelet transform (DWT)
wavelet = 'db4' 
level = 2  
coeffs = pywt.wavedec(df_amplitudes.values, wavelet, level=level)

#thresholding
threshold = 100 
for i in range(1, len(coeffs)):
    coeffs[i] = pywt.threshold(coeffs[i], threshold)

#Reconstruct the filtered amplitudes using the inverse DWT
filtered_amplitudes = pywt.waverec(coeffs, wavelet)

#Remove extreme values
threshold = 30  # Threshold for extreme values
processed_amplitudes = np.where(filtered_amplitudes > threshold, np.nan, filtered_amplitudes)

#Create a new DataFrame with the processed amplitudes
df_processed_amplitudes = pd.DataFrame(processed_amplitudes, columns=df_amplitudes.columns)

#Plot original data
fig1, ax1 = plt.subplots(figsize=(12, 6))
df_amplitudes.plot(ax=ax1)
plt.xlabel("CSI samples")
plt.ylabel("Amplitude")
plt.title("Amplitudes (Original)")


#Plot processed data
fig2, ax2 = plt.subplots(figsize=(12, 6))
df_processed_amplitudes.plot(ax=ax2)
plt.xlabel("CSI samples")
plt.ylabel("Amplitude")
plt.title("Amplitudes (Processed)")

plt.show()
