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

f = open(main_dir + grru490_prd, "r")
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
	grru_490_prd, = plt.plot(iters, np.array(result)/np.log(2), label=r"GRRU 490 $\lambda=0.001$, $\beta=0.9$ PRD")
else: 
	grru_490_prd, = plt.plot(iters, result, label=r"GRRU 490 $\lambda=0.001$, $\beta=0.9$ PRD")



f = open(main_dir + grru490_ru2_lrd, "r")
result = [] 
iters =[]
for x in lines[6:-3]:
	x = x.rstrip()
	a = x.split("\t")
	iters.append(int(a[0]))
	result.append(float(a[1]))
f.close()
if bpc: 
	grru_490_ru2_lrd, = plt.plot(iters, np.array(result)/np.log(2), label=r"GRRU 490 $\lambda=0.01$, $\beta=0.5$ LRD")
else: 
	grru_490_ru2_lrd, = plt.plot(iters, result, label=r"GRRU 490 $\lambda=0.01$, $\beta=0.5$ LRD")


f = open(main_dir + grru490_ru2, "r")
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
	grru_490_ru2, = plt.plot(iters, np.array(result)/np.log(2), label=r"GRRU 490 $\lambda=0.01$, $\beta=0.5$")
else: 
	grru_490_ru2, = plt.plot(iters, result, label=r"GRRU 490 $\lambda=0.01$, $\beta=0.5$")


f = open(main_dir + grru490_ru_lrd, "r")
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
	grru_490_ru_lrd, = plt.plot(iters, np.array(result)/np.log(2), label=r"GRRU 490 $\lambda=0.01$ LRD")
else: 
	grru_490_ru_lrd, = plt.plot(iters, result, label=r"GRRU 490 $\lambda=0.01$ LRD")


f = open(main_dir + grru490_ru, "r")
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
	grru_490_ru, = plt.plot(iters, np.array(result)/np.log(2), label=r"GRRU 490 $\lambda=0.01$")
else: 
	grru_490_ru, = plt.plot(iters, result, label=r"GRRU 490 $\lambda=0.01$")


f = open(main_dir + grru490, "r")
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
	grru_490, = plt.plot(iters, np.array(result)/np.log(2), label=r"GRRU 490")
else: 
	grru_490, = plt.plot(iters, result, label=r"GRRU 490")

f = open(main_dir + grru500tanh, "r")
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
	grru_500_tanh, = plt.plot(iters, np.array(result)/np.log(2), label=r"GRRU 500 tanh")
else: 
	grru_500_tanh, = plt.plot(iters, result, label=r"GRRU 500 tanh")

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


plt.legend(handles = [grru_490_prd, grru_490_ru_lrd, grru_490_ru, grru_490, grru_500_tanh, lstm_350, grru_490_ru2, grru_490_ru2_lrd])

plt.show()



