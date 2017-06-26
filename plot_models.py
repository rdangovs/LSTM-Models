import os 
import sys 
import scipy as sp
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import scipy.stats as stats

main_dir = "output/copying/T=200"
lstm_dir = "/LSTM_N=200_lambda=0.001_ismatrix=None.txt"
lstrm_dir_1d = "/LSTRM_N=200_lambda=0.001_ismatrix=False.txt"
lstrm_dir_2d = "/LSTRM_N=200_lambda=0.001_ismatrix=True.txt"
goru_dir_l2 = "/GORU_N=200_lambda=0.001_ismatrix=None_L=2.txt"
gru_dir = "/GRU_N=200_lambda=0.001_ismatrix=None.txt"
test_dir = "/LSTRM_N=128_lambda=0.001_ismatrix=True.txt"

plt.figure("LSTM Unitary-like Models")
plt.suptitle(r"Copying Task Performance ($T=200$, $H=200$, $\lambda=0.001$)", fontsize=16)
plt.ylabel("Loss", fontsize=14)
plt.xlabel("Training iterations", fontsize=14)

f = open(main_dir + lstm_dir,"r")
lines = f.readlines()
result = [] 
result=[]
for x in lines[6:7008]:
	x = x.rstrip()
	a = x.split("\t")
	result.append(float(a[1]))
f.close()
p1, = plt.plot(result[:7000], label="LSTM")

f = open(main_dir + lstrm_dir_1d,"r")
lines = f.readlines()
result = [] 
result=[]
for x in lines[6:7008]:
	x = x.rstrip()
	a = x.split("\t")
	result.append(float(a[1]))
f.close()
p2, = plt.plot(result[:7000], label="LSTRM 1D")

f = open(main_dir + lstrm_dir_2d,"r")
lines = f.readlines()
result = [] 
result=[]
for x in lines[6:7008]:
	x = x.rstrip()
	a = x.split("\t")
	result.append(float(a[1]))
f.close()
p3, = plt.plot(result[:7000], label="LSTRM 2D")

f = open(main_dir + goru_dir_l2,"r")
lines = f.readlines()
result = [] 
result=[]
for x in lines[6:7008]:
	x = x.rstrip()
	a = x.split("\t")
	result.append(float(a[1]))
f.close()
p4, = plt.plot(result[:7000], label=r"GORU $L=2$")

f = open(main_dir + gru_dir,"r")
lines = f.readlines()
result = [] 
result=[]
for x in lines[6:7008]:
	x = x.rstrip()
	a = x.split("\t")
	result.append(float(a[1]))
f.close()
p5, = plt.plot(result[:7000], label=r"GRU")

plt.legend(handles = [p1, p2, p3, p4, p5])
axes = plt.gca()
axes.set_ylim([0,0.4])
plt.savefig("LSTM-Models-Plot.png")
plt.show()