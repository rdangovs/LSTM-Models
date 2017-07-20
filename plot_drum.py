import os 
import sys 
import scipy as sp
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import scipy.stats as stats

#guide to the main directory
main_dir_character = "output/character"
dir_task_character = ""
T = "/T=50"

#if bits per character 
bpc = False 

#goru64_fft = "/GORU_N=128_FFT.txt"
lstm350 = "/LSTM_N=350_lambda=0.001_beta=0.9.txt"
grru500tanh = "/GRRU_N=500_lambda=0.001_beta=0.9_tanh.txt"
grru490 = "/GRRU_N=490_lambda=0.001_beta=0.9.txt"
grru490_ru = "/GRRU_N=490_lambda=0.01_beta=0.9.txt"
grru490_ru_lrd = "/GRRU_N=490_lambda=0.01_beta=0.9_lrd.txt"
grru490_ru2 = "/GRRU_N=490_lambda=0.01_beta=0.5.txt"
grru490_ru2_lrd = "/GRRU_N=490_lambda=0.01_beta=0.5.txt"
grru490_prd = "/GRRU_N=490_lambda=0.001_beta=0.9_prd.txt"
drum490 = "/DRUM_N=490_lambda=0.001_beta=0.9.txt"

main_dir = main_dir_character + dir_task_character + T
plt.figure("GRU/LSTM Unitary-like Models with Rotation")
plt.suptitle(r"PTB Task", fontsize=16)
if bpc: 
	plt.ylabel("BPC", fontsize=14)
else:
	plt.ylabel("Loss", fontsize=14)
plt.xlabel("Training iterations", fontsize=14)

f = open(main_dir + drum490, "r")
lines = f.readlines()
result = [] 
iters =[]
for x in lines[6:-3]:
	x = x.rstrip()
	a = x.split("\t")
	iters.append(int(a[0]))
	result.append(float(a[1]))
f.close()
if bpc: 
	drum_490, = plt.plot(iters, np.array(result)/np.log(2), label=r"DRUM 490")
else: 
	drum_490, = plt.plot(iters, result, label=r"DRUM 490")

f = open(main_dir + lstm350, "r")
lines = f.readlines()
result = [] 
iters =[]
for x in lines[6:-3]:
	x = x.rstrip()
	a = x.split("\t")
	iters.append(int(a[0]))
	result.append(float(a[1]))
f.close()
if bpc: 
	lstm_350, = plt.plot(iters, np.array(result)/np.log(2), label=r"LSTM 350")
else: 
	lstm_350, = plt.plot(iters, result, label=r"LSTM 350")


plt.legend(handles = [drum_490, lstm_350])
axes = plt.gca()
axes.set_ylim([1.05,1.6])

plt.savefig("./plots/DRUM_LSTM_ptb.pdf", transparent = True, format = 'pdf')
# #plt.show()



