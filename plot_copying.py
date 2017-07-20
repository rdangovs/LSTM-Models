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
grru126 = "/GRRU_N=126_lambda=0.001_decay=0.9.txt"


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

f = open(main_dir + grru126, "r")
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
	grru_126, = plt.plot(iters, np.array(result)/np.log(2), label=r"DRUM 126")
else: 
	grru_126, = plt.plot(iters, result, label=r"DRUM 126")

plt.legend(handles = [goru_128, gru_256, lstm_256, grru_126])
axes = plt.gca()
axes.set_ylim([0.00,0.40])
plt.savefig("./plots/DRUM_copying.pdf", transparent = True, format = 'pdf')
# # plt.show()



