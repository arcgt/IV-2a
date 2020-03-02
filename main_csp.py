#!/usr/bin/env python3

'''
Model for Common Spatial Pattern (CSP) feature calculation and channel selection.
'''

import os
import numpy as np
import time
import tensorflow as tf

# import self defined functions
import get_data as get
from filters import load_filterbank
from csp import generate_projection,generate_eye,extract_feature
from ranking import dimension_reduction, channel_selection_csprank, ranking_avg, channel_selection_avg

__author__ = "Michael Hersche and Tino Rellstab"
__email__ = "herschmi@ethz.ch,tinor@ethz.ch"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class CSP_Model:
	def __init__(self):
		self.data_path 	= '/usr/scratch/xavier/herschmi/EEG_data/physionet/' #data path
		self.useCSP = True
		self.fs = 160. # sampling frequency
		self.NO_channels = 64 # number of EEG channels
		self.NO_selected_channels = 16 # number of selected channels
		self.NO_subjects = 105 # number of subjects
		self.NO_csp = 12 # Total number of CSP feature per band and timewindow

		self.bw = np.array([26]) # bandwidth of filtered signals
		# self.filter_bank = self.filter_bank[18:27] # use only 4Hz bands
		self.ftype = 'butter' # 'fir', 'butter'
		self.forder= 2 # 4
		self.filter_bank = load_filterbank(self.bw,self.fs,order=self.forder,max_freq=30,ftype = self.ftype) # get filterbank coeffs
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

	# obtaining the set of 12 spatial filters across an average of all subjects.
	#
	# w_sum = 0 # sum of filters for each subject
	#
	# for model.subject in range(1,model.NO_subjects+1):
	# # for model.subject in range(1,3): #test
	# 	start = time.time()
	# 	model.load_data()
	# 	w_sum += model.run_csp() # adding filter of individual subject to sum
	# 	end = time.time()
	# 	print("Subject " + str(model.subject)+": Time elapsed = " + str(end - start) + " s")
	#
	# w_avg_4d = w_sum / 105 # calculating average of filters
	#
	# # w_avg = dimension_reduction(w_avg_4d, model.NO_channels, model.NO_csp) # for multigrade CSP
	# w_avg = w_avg_4d[0][0]
	# print(w_avg)
	# np.savetxt(f'w_avg.csv', w_avg) # saving file

	# channel selection from saved spatial filters
	w_avg = np.loadtxt(open("w_avg.csv", "rb"), delimiter=" ")

	# # V1 using ranking with average energy
	# w_squared_sum_sorted = ranking_avg(w_avg, model.NO_channels)
	# # print(w_squared_sum_sorted)
	# selected_channels = channel_selection_avg(w_squared_sum_sorted, model.NO_channels, model.NO_selected_channels)

	# V2 using CSP-ranking
	selected_channels = channel_selection_csprank(w_avg, model.NO_channels, model.NO_selected_channels, model.NO_csp)

	print("The selected channels are: ")
	print(selected_channels)

if __name__ == '__main__':
	main()
