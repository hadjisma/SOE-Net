###################################################
# configuration file
# Coded by Isma Hadji (hadjisma@cse.yorku.ca)
###################################################
"""
Prepare all global varibales and data paths.
"""
import os


DATASET     = ''
DIR_HOME    = ''
DIR_DATA   = os.path.join(DIR_HOME,DATASET)

EPS = 1.1920928955078125e-06
NUML = 5
SPEEDS = 1
NUM_DIRECTIONS = 10
ORIENTATIONS  = "standard"
FILTER_TAPS = 7
EPSILON = "std_based"
REC_STYLE = 'two_path'
