#import autokeras as ak
import pandas as pd
import numpy as np
import os

# Create a Pandas DataFrame object from the .cht files in dataset.
# Each chart in shape [difficulty, notedata[ beat[ arrows,bpm ], beat[ arrows,bpm ], beat[ arrows,bpm ], ...] ]
def create_dframe(dataset):
    arr = []

    maxm = 0
    with open(os.path.join(dataset,'meascounts.txt')) as f:
        #get max measure len from csv, ignoring trailing ,

        for c in f.read()[:-1].split(','):
            maxm = max(maxm, int(c))


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
            
            #append empty measures if shorter than longest chart
            difference = (maxm*192 + 1) - len(notedata)
            if difference > 0:
                lastbpm = notedata[-1][1]
                blank = [["0000",lastbpm] for i in range(difference)]

        row = [difficulty, notedata]
        arr.append(np.array(row, dtype=[('a', np.int32), ('b', object)]))

    arr = np.vstack((arr, row))
    
    print(arr['a'])
    print(np.shape(arr))

create_dframe('./data/dataset/')
