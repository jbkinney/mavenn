from __future__ import division
import pdb
import pylab
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib as mpl

# Class used to label subplots
class Labeler:
    def __init__(self,xpad=.1,ypad=.1,fontsize=None):
        self.xpad = xpad
        self.ypad = ypad
        if not fontsize:
            self.fontsize = 10 # Default fontsize for labels
        else:
            self.fontsize = fontsize


    def label_subplot(self,ax,label,xpad_adjust=0,ypad_adjust=0,fontsize=None):
        '''
        Adds a label to a specified subplot axes
        '''
        # Set focus on current axes
        plt.sca(ax)

        bounds = ax.get_position().bounds
        fig = ax.get_figure()
        y = bounds[1] + bounds[3] + self.ypad + ypad_adjust
        x = bounds[0] - self.xpad - xpad_adjust
        if not fontsize:
            fontsize = self.fontsize
        fig.text(x,y,label,fontsize=fontsize,ha='right',va='bottom')