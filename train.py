import pandas as pd
import numpy as np
import os
from datetime import datetime
from progress.bar import Bar

### Keras Setup ###
import autokeras as ak
import keras_tuner as kt

import tensorflow as tf
from tensorflow import keras
inputshape = (0,0)

# Create an array of AutoKeras StructuredDataInput nodes from the .cht files in dataset.
# Each input node in shape [ beat[ arrows,bpm ], beat[ arrows,bpm ], ...]
# Each output node as integer difficulty
def create_nodes(dataset):
    global inputshape

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
    
    inputshape = (len(inputs),len(inputs[0]))
    return inputs, outputs


def model_builder(hp):
    global inputshape
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=inputshape))

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model


if __name__ == "__main__":
    inputs, outputs = create_nodes('./data/dataset/')
    dt = datetime.now().isoformat(timespec='minutes').replace(":",'-')
    # tuner = kt.Hyperband(model_builder,
    #     objective='val_accuracy',
    #     max_epochs=10,
    #     factor=3,
    #     directory='./data/model',
    #     project_name=dt)

    model = ak.StructuredDataRegressor(
        inputs=inputs,
        outputs=outputs,
        project_name=dt,
        directory='./data/model',
        max_trials=5,
        tuner="hyperband")
    
    print(f"Model {model} created")
    print("~~~~~~~~~~~~~~~")

    print("Predicting Sample")
    samplein, sampleout = create_nodes('./data/testset')
    prediction = model.predict(samplein)
    
    #Display the predicted difficulties alongside actual
    total_error = 0
    for i in range(len(prediction)):
        errorperc = ( abs(prediction[i] - sampleout[i]) / float(sampleout[i]) ) * 100
        print(f"Predicted {prediction[i]:02}\tActual {sampleout[i]:02} (Error: {errorperc:02.01}%)")
        total_error += errorperc

    print()
    print(f"Average error: {total_error/len(prediction)}")

