import os 
import sys 
import scipy as sp
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import scipy.stats as stats

#guide to the main directory
main_dir_character = "output/copying"
dir_task_character = ""
T = "/T=200"

#if bits per character 
bpc = False 

#goru64_fft = "/GORU_N=128_FFT.txt"
goru128 = "/GORU_N=128_L=2.txt"
gru256 = "/GRU_N=256.txt"
lstm256 = "/LSTM_N=256.txt"
grru100 = "/GRRU_100_Mi.txt"
grru100_new = "/GRRU_100_Mi_new.txt"


main_dir = main_dir_character + dir_task_character + T
plt.figure("GRU/LSTM Unitary-like Models with Rotation")
plt.suptitle(r"Copying Memory Task", fontsize=16)
if bpc: 
	plt.ylabel("BPC", fontsize=14)
else:
	plt.ylabel("Loss", fontsize=14)
plt.xlabel("Training iterations", fontsize=14)

f = open(main_dir + goru128, "r")
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
	goru_128, = plt.plot(iters, np.array(result)/np.log(2), label=r"GORU 128")
else: 
	goru_128, = plt.plot(iters, result, label=r"GORU 128")

f = open(main_dir + gru256, "r")
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
	gru_256, = plt.plot(iters, np.array(result)/np.log(2), label=r"GRU 256")
else: 
	gru_256, = plt.plot(iters, result, label=r"GRU 256")

f = open(main_dir + lstm256, "r")
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
	lstm_256, = plt.plot(iters, np.array(result)/np.log(2), label=r"LSTM 256")
else: 
	lstm_256, = plt.plot(iters, result, label=r"LSTM 256")

f = open(main_dir + grru100, "r")
lines = f.readlines()
result = [] 
iters =[]
for x in lines[6:-1]:
	x = x.rstrip()
	a = x.split("\t")
	if float(a[1]) > 1000.0: 
		break
	iters.append(int(a[0]))
	result.append(float(a[1]))
f.close()
if bpc: 
	grru_100, = plt.plot(iters, np.array(result)/np.log(2), label=r"GRRU 100")
else: 
	grru_100, = plt.plot(iters, result, label=r"GRRU 100")

f = open(main_dir + grru100_new, "r")
lines = f.readlines()
result = [] 
iters =[]
for x in lines[6:-1]:
	x = x.rstrip()
	a = x.split("\t")
	if float(a[1]) > 1000.0: 
		break
	iters.append(int(a[0]))
	result.append(float(a[1]))
f.close()
if bpc: 
	grru_100_new, = plt.plot(iters, np.array(result)/np.log(2), label=r"GRRU 100 w. clip.")
else: 
	grru_100_new, = plt.plot(iters, result, label=r"GRRU 100 w. clip.")

plt.legend(handles = [goru_128, gru_256, lstm_256, grru_100, grru_100_new])
axes = plt.gca()
axes.set_ylim([0.00,0.40])

plt.show()



