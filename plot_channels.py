import os
import numpy as np
from PIL import Image

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

def plot_heatmap_avg(num_classes,type):
    ''' plot heatmap of 64-channels on 10-10 international system - average across all 12 filters

    Keyword arguments:
    num_classes -- number of classes
    type -- 'channels' for coloured ring around channels or 'channels_fill' for opaque colour heatmap
    '''
    os.makedirs(f'plot_channels/plots/heatmap', exist_ok=True)

    w_temp = np.loadtxt(open(f'results/w_{num_classes}class_csp.csv', "rb"), delimiter=" ")
    w = np.zeros(64) # creating an empty of dimension 64 x 2 for channel number and energy

    index = 0 # iterator

    for channel in w_temp:
        for filter in channel:
            w[index] += np.sqrt(filter**2) # set channel energy
        index+=1

    mean = np.mean(w)
    sd = np.std(w)

    background = Image.open(f'plot_channels/64_channel_sharbrough_bg.png')

    channel = 1
    for i in w:
        if i > mean+sd:
            img = Image.open(f'plot_channels/{type}/{channel}.png')
            background.paste(img, (0, 0), img)
        elif i > mean:
            img = Image.open(f'plot_channels/{type}/{channel}_o.png')
            background.paste(img, (0, 0), img)
        elif i > mean-sd:
            img = Image.open(f'plot_channels/{type}/{channel}_y.png')
            background.paste(img, (0, 0), img)
        else:
            pass
        channel += 1
    background.save(f'plot_channels/plots/heatmap/class{num_classes}.png',"PNG")

def plot_heatmap(num_classes,type):
    ''' plot heatmap of 64-channels on 10-10 international system - for all 12 csp filters

    Keyword arguments:
    num_classes -- number of classes
    type -- 'channels' for coloured ring around channels or 'channels_fill' for opaque colour heatmap
    '''
    os.makedirs(f'plot_channels/plots/heatmap', exist_ok=True)

    w = np.loadtxt(open(f'results/w_{num_classes}class_csp.csv', "rb"), delimiter=" ")

    mean = np.mean(w)
    sd = np.std(w)

    for filter in range(12):
        background = Image.open(f'plot_channels/64_channel_sharbrough_bg.png')

        channel = 1
        for i in w[:,filter]:
            if i > mean+sd:
                img = Image.open(f'plot_channels/{type}/{channel}.png')
                background.paste(img, (0, 0), img)
            elif i > mean:
                img = Image.open(f'plot_channels/{type}/{channel}_o.png')
                background.paste(img, (0, 0), img)
            elif i > mean-sd:
                img = Image.open(f'plot_channels/{type}/{channel}_y.png')
                background.paste(img, (0, 0), img)
            else:
                pass
            channel += 1
        background.save(f'plot_channels/plots/heatmap/class{num_classes}_filter{filter}.png',"PNG")

# plot_heatmap_avg(4,'channels_fill')
# plot_heatmap(4,'channels_fill')


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
