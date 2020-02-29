#!/usr/bin/env python3

''' Functions used for ranking and channel selection '''

import numpy as np

__author__ = "Tianhong Gan"
__email__ = "tianhonggan@outlook.com"

def ranking(w_avg, NO_channels):
    ''' ranking the energy of each channel obtained from the set of 12 spatial filters

     Keyword arguments:
     w_avg -- set of 12 spatial filters obtained from a average of filters obtained from each subject of size [NO_channels, NO_csp]
     NO_channels - total number of channels

     Return: channels sorted by energy usage of size [NO_channels, 2]
    '''
    w_squared_sum = np.zeros((NO_channels,2)) # creating an empty of dimension NO_channels x 2 for channel number and energy

    index = 0 # iterator

    for channel in w_avg:
        w_squared_sum[index][0] = index + 1 # set channel number
        for filter in channel:
            w_squared_sum[index][1] += filter**2 # set channel energy
        index+=1

    w_squared_sum_sorted = w_squared_sum[w_squared_sum[:,1].argsort()][::-1] # sort energy in descending order

    return w_squared_sum_sorted


def channel_selection(w_squared_sum_sorted, NO_channels, NO_selected):
    ''' select channels with highest energy usage

    Keyword arguments:
    w_squared_sum_sorted -- channels sorted by energy usage of size [NO_channels, 2]
    NO_selected -- number of channels to select

    Return: 'NO_selected' channels with the highest energy usage
    '''
    if NO_selected <= NO_channels:
        selected_channels = np.sort(w_squared_sum_sorted[0:NO_selected, 0])
    else:
        selected_channels = np.sort(w_squared_sum_sorted[0:NO_channels, 0])

    return selected_channels
