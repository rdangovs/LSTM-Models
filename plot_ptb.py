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
grru64_A_Mi = "/GRRU_N=64_A_Mi.txt"
grru64_Mi = "/GRRU_N=64_Mi.txt"
lstm46 = "/LSTM_N=46.txt"
lstm350 = "/LSTM_N=350_lambda=0.001_beta=0.9.txt"
grru415 = "/GRRU_N=415_lambda=0.001_beta=0.1_modrelu_-0.001.txt"

main_dir = main_dir_character + dir_task_character + T
plt.figure("GRU/LSTM Unitary-like Models with Rotation")
plt.suptitle(r"PTB Task", fontsize=16)
if bpc: 
	plt.ylabel("BPC", fontsize=14)
else:
	plt.ylabel("Loss", fontsize=14)
plt.xlabel("Training iterations", fontsize=14)

f = open(main_dir + grru415, "r")
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
	grru_415, = plt.plot(iters, np.array(result)/np.log(2), label=r"GRRU 415 modReLU")
else: 
	grru_415, = plt.plot(iters, result, label=r"GRRU 415 modReLU")

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


plt.legend(handles = [grru_415, lstm_350])

plt.show()



