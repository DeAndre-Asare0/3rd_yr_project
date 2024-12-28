import matplotlib.pyplot as plt
import math
import pandas as pd

#select csv file that contains raw CSI data.
csi_file = 0 #'c:\\Test-Files\\first-attempts\\single-action\\a-punch-test.csv'
init_df = pd.read_csv(csi_file)
init_df.info()
amplitudes_list = []
phases_list = []

#Calculate subcarriers 0-63, amplitude and phase.
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

    return amplitudes, phases

#read the file, calculate amplitude-time signal.
with open(csi_file, "r") as f:
    for line in f:
        if "CSI_DATA" in line:
            amplitudes, phases = process(line)
            amplitudes_list.append(amplitudes)
            phases_list.append(phases)

df_amplitudes = pd.DataFrame(amplitudes_list, columns=["Subcarrier{}".format(i) for i in range(64)])
df_phases = pd.DataFrame(phases_list, columns=["Subcarrier{}".format(i) for i in range(64)])
#remove useless subcarriers
subcarriers_to_drop = ['Subcarrier0', 'Subcarrier1', 'Subcarrier2', 'Subcarrier3', 'Subcarrier4', 'Subcarrier5', 'Subcarrier32', 'Subcarrier59', 'Subcarrier60', 'Subcarrier61', 'Subcarrier62', 'Subcarrier63']
df_amplitudes = df_amplitudes.drop(columns=subcarriers_to_drop)
df_phases = df_phases.drop(columns=subcarriers_to_drop)

#plot amplitude and phase.
fig1, ax1 = plt.subplots(figsize=(12, 6))
df_amplitudes.plot(ax=ax1)
plt.xlabel("CSI samples")
plt.ylabel("Amplitude")
plt.title("Amplitudes")

fig2, ax2 = plt.subplots(figsize=(12, 6))
df_phases.plot(ax=ax2)
plt.xlabel("CSI samples")
plt.ylabel("Phase")
plt.title("Phases")
plt.show()
