import numpy as np

def ranking(w_avg, NO_channels):
    w_squared_sum = np.zeros((NO_channels,2))

    index = 0

    for channel in w_avg:
        w_squared_sum[index][0] = index
        for filter in channel:
            w_squared_sum[index][1] += filter**2
        index++

    w_squared_sum_sorted = w_squared_sum[w_squared_sum[:,1].argsort()]

    return w_squared_sum_sorted


def channel_selection(w_squared_sum_sorted, NO_channels, NO_selected):
    eight_channel = w_squared_sum_sorted[[56:NO_channels], 0]
    sixteen_channel = w_squared_sum_sorted[[48:NO_channels], 0]
    twentyfour_channel = w_squared_sum_sorted[[40:NO_channels], 0]

    if NO_selected == 8:
        return eight_channel
    elif NO_selected == 16:
        return sixteen_channel
    else:
        return twentyfour_channel
