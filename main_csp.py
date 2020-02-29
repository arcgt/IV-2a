#!/usr/bin/env python3

'''
Model for common spatial pattern (CSP) feature calculation
'''

import os
import numpy as np
import time

# import self defined functions
from csp import generate_projection,generate_eye,extract_feature
import get_data as get
from filters import load_filterbank
from ranking import ranking, channel_selection

__author__ = "Michael Hersche and Tino Rellstab"
__email__ = "herschmi@ethz.ch,tinor@ethz.ch"

os.makedirs(f'results', exist_ok=True)

class CSP_Model:
	def __init__(self):
		self.data_path 	= '/usr/scratch/xavier/herschmi/EEG_data/physionet/' #data path
		self.useCSP = True
		self.fs = 160. # sampling frequency
		self.NO_channels = 64 # number of EEG channels
		self.NO_subjects = 105 # number of subjects
		self.NO_csp = 12 # Total number of CSP feature per band and timewindow

		self.bw = np.array([26]) # bandwidth of filtered signals
		self.ftype = 'butter' # 'fir', 'butter'
		self.forder= 2 # 4
		self.filter_bank = load_filterbank(self.bw,self.fs,order=self.forder,max_freq=30,ftype = self.ftype) # get filterbank coeffs
		# self.filter_bank = self.filter_bank[18:27] # use only 4Hz bands
		self.NO_bands = self.filter_bank.shape[0]

		time_windows_flt = np.array([
									[0,1],
									[0.5,1.5],
									[1,2],
									[1.5,2.5],
									[2,3],
									[0,2],
									[0.5,2.5],
									[1,3],
									[0,3]])*self.fs # time windows in [s] x fs for using as a feature
		self.time_windows = time_windows_flt.astype(int)
		self.time_windows = self.time_windows[8] # use only largest timewindow
		self.NO_time_windows = int(self.time_windows.size/2)

		self.NO_features = self.NO_csp*self.NO_bands*self.NO_time_windows

	def run_csp(self):
		# obtain spatial filters
		if self.useCSP:
			w = generate_projection(self.train_data,self.train_label, self.NO_csp,self.filter_bank,self.time_windows)
		else:
			w = generate_eye(self.train_data,self.train_label,self.filter_bank,self.time_windows)

		return w

	def load_data(self):
		#load data
		npzfile = np.load(self.data_path+f'{4}class.npz')
		self.train_data, self.train_label = npzfile['X_Train'], npzfile['y_Train']

def main():
	print("Starting program...")
	model = CSP_Model()

	w_sum = 0 # sum of filters for each subject

	for model.subject in range(1,model.NO_subjects+1):
		start = time.time()
		model.load_data()
		w_sum += model.run_csp() # adding filter of individual subject to sum
		end = time.time()
		print("Subject " + str(model.subject)+": Time elapsed = " + str(end - start) + " s")

	w_avg = w_sum[0][0] / 105 # calculating average of filters

	np.savetxt(f'results/w_avg.csv', w_avg) # saving file
	# w_avg = numpy.loadtxt(open("w_avg.csv", "rb"), delimiter=" ") #[[12] x64]
	print(w_avg)

	w_squared_sum_sorted = ranking(w_avg, model.NO_channels)
	selected_channels = channel_selection(w_squared_sum_sorted, model.NO_channels, 8) #NO_selected = 8, 16, 24

	print("The selected channels are: ")
	print(selected_channels)

if __name__ == '__main__':
	main()
