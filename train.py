import keras as k
import pandas as pd
import numpy as np
import os

# Create a Pandas DataFrame object from the .cht files in dataset.
# Each chart in shape [difficulty, notedata[ beat[ arrows,bpm ], beat[ arrows,bpm ], beat[ arrows,bpm ], ...] ]
def create_dframe(dataset):
    arr = np.ndarray(dtype=object)

    for f in os.listdir(dataset):
        fp = os.path.join(dataset,f)
        row = []
        difficulty = int( f[ f.rfind('[')+1 : f.rfind(']') ] )

        notedata = []
        with open(fp, 'r') as data:
            datalines = data.readlines()

            for l in datalines:
                #Turn info about a beat into array [arrows, bpm (float)]
                pointinfo = l.strip('\n').split(',')
                beat = [pointinfo[0], float(pointinfo[1])]
                notedata.append(beat)
        row = [difficulty, notedata]
        arr = np.append(arr, row)
    
    print(np.shape(arr))

create_dframe('./data/testset/')
