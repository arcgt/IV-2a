#!/usr/bin/env python3

'''
Model for Common Spatial Pattern (CSP) feature extraction and channel selection.
'''

import os
import numpy as np
import time

# import self defined functions
import get_data as get
from filters import load_filterbank
from csp import generate_projection,generate_eye,extract_feature
from ranking import dimension_reduction, channel_selection_csprank, channel_selection_squared_sum

__author__ = "Michael Hersche, Tino Rellstab, Tianhong Gan"
__email__ = "herschmi@ethz.ch, tinor@ethz.ch, tianhonggan@outlook.com"

results_dir=f'results'
os.makedirs(f'{results_dir}', exist_ok=True)

class CSP_Model:
	def __init__(self):
		self.data_path 	= '/usr/scratch/xavier/herschmi/EEG_data/physionet/' #data path
		self.obtain_filter = False # need to reobtain filters
		self.run_channel_selection = True # need to select channels
		self.channel_selection_method = 1 # 1: w squared sum, 2: csp-rank

		self.fs = 160. # sampling frequency
		self.NO_channels = 64 # number of EEG channels
		self.NO_selected_channels = 38 # number of selected channels
		self.NO_subjects = 105 # number of subjects
		self.NO_csp = 2 # Total number of CSP features per band and timewindow
		self.NO_classes = 2

		self.bw = np.array([26]) # bandwidth of filtered signals
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

	def load_data(self):
		#load data
		npzfile = np.load(self.data_path+f'{self.NO_classes}class.npz')
		self.train_data, self.train_label = npzfile['X_Train'], npzfile['y_Train']

	def run_csp(self):
		# obtaining the set of 12 spatial filters across an average of all subjects.
		w_sum = 0 # sum of filters for each subject

		for self.subject in range(1,self.NO_subjects+1):
		# for self.subject in range(1,3): #test
			start = time.time()
			self.load_data()
			w_sum += generate_projection(self.train_data,self.train_label,self.NO_csp,self.filter_bank,self.time_windows,self.NO_classes) # adding filter of individual subject to sum
			end = time.time()
			print("Subject " + str(self.subject)+": Time elapsed = " + str(end - start) + " s")

		w_avg_4d = w_sum / self.NO_subjects # calculating average of filters
		w_avg = dimension_reduction(w_avg_4d, self.NO_channels, self.NO_csp) # dimension reduction (for multiscale CSP)

		np.savetxt(f'{results_dir}/w_avg_{self.NO_classes}class_csp.csv', w_avg) # saving file

	def channel_selection(self):
		# channel selection from saved spatial filters
		w_avg = np.loadtxt(open(f'{results_dir}/w_avg_{self.NO_classes}class_csp.csv', "rb"), delimiter=" ")

		if self.channel_selection_method == 1: #V1 using w squared sum
			selected_channels = channel_selection_squared_sum(w_avg, self.NO_channels, self.NO_selected_channels)
		elif self.channel_selection_method == 2: # V2 using CSP-ranking
			selected_channels = channel_selection_csprank(w_avg, self.NO_channels, self.NO_selected_channels, self.NO_csp)

		return selected_channels


def main():
	print("Starting program...")
	model = CSP_Model()

	if model.obtain_filter:
		model.run_csp()

	if model.run_channel_selection:
		print("The selected channels are: ")
		print(model.channel_selection())

if __name__ == '__main__':
	main()
