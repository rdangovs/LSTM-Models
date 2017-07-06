import os 
import sys 
import scipy as sp
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import scipy.stats as stats

#guide to the main directory
main_dir_copying = "output/copying"
main_dir_character = "output/character"
dir_task_character = "/text8"
T = "/T=50"

#if bits per character 
bpc = False 

goru64_fft = "/GORU_N=64_FFT.txt"
grru64_A_Mi = "/GRRU_N=64_A_Mi.txt"
grru64_A_Mi_Adam = "/GRRU_N=64_A_Mi_Adam.txt"
grru64_A_Mi_sigmoid = "/GRRU_N=64_A_Mi_sigmoid.txt"
grru64 = "/GRRU_N=64.txt"
lstm64 = "/LSTM_N=64.txt"
gru64 = "/GRU_N=64.txt"
grru64_A_Mi_E_25 = "/GRRU_N=64_A_Mi_E=25.txt"
lstm64_E_25 = "/LSTM_N=64_E25.txt"

main_dir = main_dir_character + dir_task_character + T
plt.figure("GRU/LSTM Unitary-like Models with Rotation")
plt.suptitle(r"Text8 Task", fontsize=16)
if bpc: 
	plt.ylabel("BPC", fontsize=14)
else:
	plt.ylabel("Loss", fontsize=14)
plt.xlabel("Training iterations", fontsize=14)

"""
f = open(main_dir + grru64_A_Mi, "r")
lines = f.readlines()
result = [] 
iters =[]
for x in lines[6:-1]:
	x = x.rstrip()
	a = x.split("\t")
	iters.append(int(a[0]))
	result.append(float(a[1]))
f.close()
if bpc: 
	grru64a_mi, = plt.plot(iters, np.array(result)/np.log(2), label=r"GRRU 64 A Mi")
else: 
	grru64a_mi, = plt.plot(iters, result, label=r"GRRU 64 A Mi")

f = open(main_dir + grru64_A_Mi_sigmoid, "r")
lines = f.readlines()
result = [] 
iters =[]
for x in lines[6:-1]:
	x = x.rstrip()
	a = x.split("\t")
	iters.append(int(a[0]))
	result.append(float(a[1]))
f.close()
if bpc: 
	grru64a_mi_sig, = plt.plot(iters, np.array(result)/np.log(2), label=r"GRRU 64 A Mi sig")
else: 
	grru64a_mi_sig, = plt.plot(iters, result, label=r"GRRU 64 A Mi $\sigma$")

f = open(main_dir + grru64, "r")
lines = f.readlines()
result = [] 
iters =[]
for x in lines[6:-1]:
	x = x.rstrip()
	a = x.split("\t")
	iters.append(int(a[0]))
	result.append(float(a[1]))
f.close()
if bpc: 
	grru64, = plt.plot(iters, np.array(result)/np.log(2), label=r"GRRU 64")
else: 
	grru64, = plt.plot(iters, result, label=r"GRRU 64")

f = open(main_dir + lstm64, "r")
lines = f.readlines()
result = [] 
iters =[]
for x in lines[6:-1]:
	x = x.rstrip()
	a = x.split("\t")
	iters.append(int(a[0]))
	result.append(float(a[1]))
f.close()
if bpc: 
	lstm64, = plt.plot(iters, np.array(result)/np.log(2), label=r"LSTM 64")
else: 
	lstm64, = plt.plot(iters, result, label=r"LSTM 64")

f = open(main_dir + grru64_A_Mi_Adam, "r")
lines = f.readlines()
result = [] 
iters =[]
for x in lines[6:-1]:
	x = x.rstrip()
	a = x.split("\t")
	iters.append(int(a[0]))
	result.append(float(a[1]))
f.close()
if bpc: 
	grru64a_mi_ad, = plt.plot(iters, np.array(result)/np.log(2), label=r"GRRU 64 A Mi Ad")
else: 
	grru64a_mi_ad, = plt.plot(iters, result, label=r"GRRU 64 A Mi Ad")

f = open(main_dir + goru64_fft, "r")
lines = f.readlines()
result = [] 
iters =[]
for x in lines[6:-1]:
	x = x.rstrip()
	a = x.split("\t")
	iters.append(int(a[0]))
	result.append(float(a[1]))
f.close()
if bpc: 
	goru64, = plt.plot(iters, np.array(result)/np.log(2), label=r"GORU 64 FFT")
else: 
	goru64, = plt.plot(iters, result, label=r"GORU 64 FFT")

f = open(main_dir + gru64, "r")
lines = f.readlines()
result = [] 
iters =[]
for x in lines[6:-1]:
	x = x.rstrip()
	a = x.split("\t")
	iters.append(int(a[0]))
	result.append(float(a[1]))
f.close()
if bpc: 
	gru64, = plt.plot(iters, np.array(result)/np.log(2), label=r"GRU 64")
else: 
	gru64, = plt.plot(iters, result, label=r"GRU 64")

plt.legend(handles = [grru64a_mi, grru64a_mi_sig, grru64, lstm64, grru64a_mi_ad, goru64, gru64])
"""

f = open(main_dir + grru64_A_Mi_E_25, "r")
lines = f.readlines()
result = [] 
iters =[]
for x in lines[6:-1]:
	x = x.rstrip()
	a = x.split("\t")
	iters.append(int(a[0]))
	result.append(float(a[1]))
f.close()
if bpc: 
	grru64amie25, = plt.plot(iters, np.array(result)/np.log(2), label=r"GRRU 64 A Mi E 25")
else: 
	grru64amie25, = plt.plot(iters, result, label=r"GRRU 64 A Mi E 25")

f = open(main_dir + lstm64_E_25, "r")
lines = f.readlines()
result = [] 
iters =[]
for x in lines[6:-1]:
	x = x.rstrip()
	a = x.split("\t")
	iters.append(int(a[0]))
	result.append(float(a[1]))
f.close()
if bpc: 
	lstm64e25, = plt.plot(iters, np.array(result)/np.log(2), label=r"LSTM 64 E 25")
else: 
	lstm64e25, = plt.plot(iters, result, label=r"LSTM 64 E 25")



plt.legend(handles = [grru64amie25, lstm64e25])

plt.show()



