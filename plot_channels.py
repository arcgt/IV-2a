import os
import numpy as np
from PIL import Image
from keras.models import load_model

from ranking import get_cspweights

__author__ = "Tianhong Gan"
__email__ = "tianhonggan@outlook.com"

def plot_channels(channels, num_classes, n_ch):
    ''' plot channels selected on 64 channel template

    Keyword arguments:
    channels -- array of selected channels
    num_classes -- number of classes
    n_ch -- number of channels selected
    '''
    os.makedirs(f'plot_channels/plots', exist_ok=True)
    background = Image.open(f'plot_channels/64_channel_sharbrough_bg.png')
    for i in channels:
        channel = i + 1
        img = Image.open(f'plot_channels/channels/{channel}.png')
        background.paste(img, (0, 0), img)
    background.save(f'plot_channels/plots/class{num_classes}_nch{n_ch}.png',"PNG")

def plot_heatmap(num_classes):
    os.makedirs(f'plot_channels/plots/heatmap', exist_ok=True)

    w = np.loadtxt(open(f'results/w_{num_classes}class_csp.csv', "rb"), delimiter=" ")
    w = get_cspweights(w)

    background = Image.open(f'plot_channels/64_channel_sharbrough_bg.png')

    channel = 1
    for i in w:
        if i > 1:
            img = Image.open(f'plot_channels/channels/{channel}.png')
            background.paste(img, (0, 0), img)
        elif i > 0.6:
            img = Image.open(f'plot_channels/channels/{channel}_o.png')
            background.paste(img, (0, 0), img)
        elif i > 0.4:
            img = Image.open(f'plot_channels/channels/{channel}_y.png')
            background.paste(img, (0, 0), img)
        else:
            pass
        channel += 1
    background.save(f'plot_channels/plots/heatmap/class{num_classes}.png',"PNG")

# plot_heatmap(4)

''' make white background transparent '''

# for i in range(64):
#     channel = i + 1
#     img = Image.open(f'plot_channels/channels/{channel}_y.png')
#     img = img.convert("RGBA")
#     datas = img.getdata()
#
#     newData = []
#     for item in datas:
#         if item[0] == 255 and item[1] == 255 and item[2] == 255:
#             newData.append((255, 255, 255, 0))
#         else:
#             newData.append(item)
#
#     img.putdata(newData)
#     img.save(f'plot_channels/channels/{channel}_y.png', "PNG")
