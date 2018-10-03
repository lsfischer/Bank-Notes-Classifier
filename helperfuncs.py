#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 09:47:38 2018

@author: lucas
"""

import numpy as np 


def read_data_file(filename, delim):
    """ Reads the data file and gets data separated by delimiter delim """
    """ input: filename to read (filename), delimeter (delim) """
    """ ouput: data """
    # Load the data from file
    data = np.loadtxt(filename,delimiter=delim)
    return data
