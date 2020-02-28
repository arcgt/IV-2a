# w_avg = numpy.loadtxt(open("w_avg.csv", "rb"), delimiter=" ") #[[12] x64]

def ranking(w_avg):
    w_squared_sum = np.zeros((64,2))

    index = 0

    for channel in w_avg:
        w_squared_sum[index][0] = index
        for filter in channel:
            w_squared_sum[index][1] += filter**2
        index++

    w_squared_sum_sorted = w_squared_sum[w_squared_sum[:,1].argsort()]

    eight_channel = w_squared_sum_sorted[[56:64], 0]
    sixteen_channel = w_squared_sum_sorted[[48:64], 0]
    twentyfour_channel = w_squared_sum_sorted[[40:64], 0]

    print("8 channels: ")
    print(eight_channel)
    print("16 channels: ")
    print(sixteen_channel)
    print("24 channels: ")
    print(twentyfour_channel)
