import autokeras as ak
import pandas as pd
import numpy as np
import os
from progress.bar import Bar

# Create an array of AutoKeras StructuredDataInput nodes from the .cht files in dataset.
# Each input node in shape [ beat[ arrows,bpm ], beat[ arrows,bpm ], ...]
# Each output node as integer difficulty
def create_nodes(dataset):
    inputs = []
    outputs = []

    maxm = 0
    chart_count = 0
    with open(os.path.join(dataset,'meascounts.txt')) as f:
        
        #get max measure len from csv, ignoring trailing ,
        for c in f.read()[:-1].split(','):
            maxm = max(maxm, int(c))
            chart_count += 1

    with Bar('Loading nodes...',suffix='%(percent).1f%% - %(eta)ds',max=len(os.listdir(dataset))) as bar:
        for f in os.listdir(dataset):

            #skip non-cht files
            if not os.path.splitext(f)[1] == '.cht':
                bar.next()
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
                    beat = [pointinfo[0], float(pointinfo[1])]
                    notedata.append(beat)
                
                #append empty measures if shorter than longest chart
                difference = ((maxm+1)*192) - len(notedata)
                if difference > 0:
                    lastbpm = notedata[-1][1]
                    blank = [["0000",lastbpm] for i in range(difference)]
                    notedata += blank

            types = {'Note':'categorical', 'BPM':'Numerical'}
            cname = os.path.basename(fp)
            node = ak.StructuredDataInput(column_names=['Note', 'BPM'],column_types=types, name=cname)
            inputs.append(node)
            outputs.append(difficulty)
            bar.next()

    print(f"{len(inputs)} nodes")
    
    return inputs, outputs

create_nodes('./data/dataset/')
