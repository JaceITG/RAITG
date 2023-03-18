#import autokeras as ak
import pandas as pd
import numpy as np
import os

# Create a Pandas DataFrame object from the .cht files in dataset.
# Each chart in shape [ beat[ arrows,bpm ], beat[ arrows,bpm ], beat[ arrows,bpm ], ..., difficulty]
# With output being array[-1] (difficulty)
def create_dframe(dataset):
    arr = []

    maxm = 0
    chart_count = 0
    with open(os.path.join(dataset,'meascounts.txt')) as f:
        
        #get max measure len from csv, ignoring trailing ,
        for c in f.read()[:-1].split(','):
            maxm = max(maxm, int(c))
            chart_count += 1

    #arr = np.array(np.zeros(shape=(chart_count,(maxm+1)*192)))

    cnt = 0
    for f in os.listdir(dataset):

        #skip non-cht files
        if not os.path.splitext(f)[1] == '.cht':
            continue

        fp = os.path.join(dataset,f)
        row = []
        difficulty = int( f[ f.rfind('[')+1 : f.rfind(']') ] )

        notedata = []
        with open(fp, 'r') as data:
            datalines = data.readlines()

            for l in datalines:
                #Turn info about a beat into array [arrows, bpm (float)]
                pointinfo = l.strip('\n').split(',')
                beat = (pointinfo[0], float(pointinfo[1]))
                notedata.append(beat)
            
            #append empty measures if shorter than longest chart
            difference = ((maxm+1)*192) - len(notedata)
            if difference > 0:
                lastbpm = notedata[-1][1]
                blank = [("0000",lastbpm) for i in range(difference)]
                notedata += blank


        notedata.append(difficulty)
        new = np.array(notedata, dtype=object)
        arr.append(new)
        cnt += 1

    arr = np.vstack(tuple(c for c in arr))

    print(np.shape(arr))
    

create_dframe('./data/dataset/')
