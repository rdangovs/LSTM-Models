import os 
import sys 
import scipy as sp
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import scipy.stats as stats

task = "text8"

main_dir_copying = "output/copying"
main_dir_character = "output/character"
dir_task_character = "/text8"
T = "/T=50"

lstm_dir = "/LSTM_N=200_lambda=0.001_ismatrix=None.txt"
lstrm_dir_1d = "/LSTRM_N=200_lambda=0.001_ismatrix=False.txt"
lstrm_dir_2d = "/LSTRM_N=200_lambda=0.001_ismatrix=True.txt"
goru_dir_l2 = "/GORU_N=200_lambda=0.001_ismatrix=None_L=2.txt"
gru_dir = "/GRU_N=200_lambda=0.001_ismatrix=None.txt"
test_dir = "/LSTRM_N=128_lambda=0.001_ismatrix=True.txt"

lstm_char = ["/LSTM_N=200_IM=None.txt", "/LSTM_N=400_IM=None.txt", "/LSTM_N=350_IM=None.txt"]
goru_char = ["/GORU_N=200_IM=None_L=8.txt", "/GORU_N=512_IM=None_FFT.txt"]
lstrm1d_char = ["/LSTRM_N=200_IM=False.txt", "/LSTRM_N=400_IM=False.txt", "/LSTRM_N=512_IM=False.txt"]
eunn_char = ["/EURNN_N=2048_IM=None_FFT.txt"]
gru_char = ["/GRU_N=415_IM=None.txt"]
lstrm2d_char = ["/LSTRM_N=350_IM=True.txt"]

if task == "copying":
	main_dir = main_dir_copying
	plt.figure("LSTM Unitary-like Models")
	plt.suptitle(r"Copying Task Performance ($T=200$, $H=200$, $\lambda=0.001$)", fontsize=16)
	plt.ylabel("Loss", fontsize=14)
	plt.xlabel("Training iterations", fontsize=14)

	f = open(main_dir + lstm_dir,"r")
	lines = f.readlines()
	result = [] 
	for x in lines[6:7008]:
		x = x.rstrip()
		a = x.split("\t")
		result.append(float(a[1]))
	f.close()
	p1, = plt.plot(result[:7000], label="LSTM")

	f = open(main_dir + lstrm_dir_1d,"r")
	lines = f.readlines()
	result = [] 
	for x in lines[6:7008]:
		x = x.rstrip()
		a = x.split("\t")
		result.append(float(a[1]))
	f.close()
	p2, = plt.plot(result[:7000], label="LSTRM 1D")

	f = open(main_dir + lstrm_dir_2d,"r")
	lines = f.readlines()
	result = [] 
	for x in lines[6:7008]:
		x = x.rstrip()
		a = x.split("\t")
		result.append(float(a[1]))
	f.close()
	p3, = plt.plot(result[:7000], label="LSTRM 2D")

	f = open(main_dir + goru_dir_l2,"r")
	lines = f.readlines()
	result = [] 
	for x in lines[6:7008]:
		x = x.rstrip()
		a = x.split("\t")
		result.append(float(a[1]))
	f.close()
	p4, = plt.plot(result[:7000], label=r"GORU $L=2$")

	f = open(main_dir + gru_dir,"r")
	lines = f.readlines()
	result = [] 
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

elif task == "text8":
	main_dir = main_dir_character + dir_task_character + T
	plt.figure("LSTM Unitary-like Models")
	plt.suptitle(r"Text8 Task", fontsize=16)
	plt.ylabel("Loss", fontsize=14)
	#plt.ylabel("BPC", fontsize=14)
	plt.xlabel("Training iterations", fontsize=14)

	f = open(main_dir + lstrm1d_char[2], "r")
	lines = f.readlines()
	result = [] 
	iters =[]
	for x in lines[6:-1]:
		x = x.rstrip()
		a = x.split("\t")
		iters.append(int(a[0]))
		result.append(float(a[1]))
	f.close()
	#lstrm1d_l, = plt.plot(iters, np.array(result)/np.log(2), label=r"LSTRM1D")
	lstrm1d_l, = plt.plot(iters, result, label=r"LSTRM1D")

	f = open(main_dir + lstrm2d_char[0], "r")
	lines = f.readlines()
	result = [] 
	iters =[]
	for x in lines[6:-1]:
		x = x.rstrip()
		a = x.split("\t")
		iters.append(int(a[0]))
		result.append(float(a[1]))
	f.close()
	#lstrm2d_l, = plt.plot(iters, np.array(result)/np.log(2), label=r"LSTRM2D")
	lstrm2d_l, = plt.plot(iters, result, label=r"LSTRM2D")

	f = open(main_dir + goru_char[1], "r")
	lines = f.readlines()
	result = [] 
	iters =[]
	for x in lines[6:-1]:
		x = x.rstrip()
		a = x.split("\t")
		iters.append(int(a[0]))
		result.append(float(a[1]))
	f.close()
	#goru_l, = plt.plot(iters, np.array(result)/np.log(2), label=r"GORU")
	goru_l, = plt.plot(iters, result, label=r"GORU")
	
	f = open(main_dir + gru_char[0], "r")
	lines = f.readlines()
	result = [] 
	iters = []
	for x in lines[6:-1]:
		x = x.rstrip()
		a = x.split("\t")
		result.append(float(a[1]))
		iters.append(int(a[0]))
	f.close()
	#gru_l, = plt.plot(iters, np.array(result)/np.log(2), label="GRU")
	gru_l, = plt.plot(iters, result, label="GRU")

	f = open(main_dir + lstm_char[2], "r")
	lines = f.readlines()
	result = [] 
	iters = []
	for x in lines[6:-1]:
		x = x.rstrip()
		a = x.split("\t")
		result.append(float(a[1]))
		iters.append(int(a[0]))
	f.close()
	#lstm_l, = plt.plot(iters, np.array(result)/np.log(2), label="LSTM")
	lstm_l, = plt.plot(iters, result, label="LSTM")

	f = open(main_dir + eunn_char[0], "r")
	lines = f.readlines()
	result = [] 
	iters = []
	for x in lines[6:-1]:
		x = x.rstrip()
		a = x.split("\t")
		result.append(float(a[1]))
		iters.append(int(a[0]))
	f.close()
	#eunn_l, = plt.plot(iters, np.array(result)/np.log(2), label="EUNN")
	eunn_l, = plt.plot(iters, result, label="EUNN")
	
	#plt.savefig("LSTM-Models-Text8-H200.png")
	#plt.savefig("LSTM-Models-Text8-H400.png")

	plt.legend(handles = [lstrm1d_l, lstrm2d_l, goru_l, gru_l, lstm_l, eunn_l])
	#plt.legend(handles = [q1, q3])
	axes = plt.gca()
	#axes.set_ylim([1.25 / np.log(2),1.85 / np.log(2)])
	axes.set_ylim([1.25,1.85])
	
	plt.show()



