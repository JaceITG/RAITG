import pandas as pd
import numpy as np
import os, csv
from datetime import datetime
from progress.bar import Bar

### Keras Setup ###
import autokeras as ak
# import keras_tuner as kt

# import tensorflow as tf
# from tensorflow import keras
inputshape = (0,0)

# Create an array of AutoKeras StructuredDataInput nodes from the .cht files in dataset.
# Each input node in shape [ beat[ arrows,bpm ], beat[ arrows,bpm ], ...]
# Each output node as integer difficulty
def create_nodes(dataset):

    maxm = 0
    chart_count = 0
    with open(os.path.join(dataset,'meascounts.txt')) as f:
        
        #get max measure len from csv, ignoring trailing ,
        for c in f.read()[:-1].split(','):
            maxm = max(maxm, int(c))
            chart_count += 1
    
    chts = []
    outputs_df = pd.DataFrame(columns=['difficulty'])
    idx = 0
    with Bar('Loading nodes...',suffix='%(percent).1f%% - %(eta)ds',max=len(os.listdir(dataset))) as bar:
        for f in os.listdir(dataset):

            #skip non-cht files
            if not os.path.splitext(f)[1] == '.cht':
                bar.next()
                continue

            fp = os.path.join(dataset,f)
            difficulty = int( f[ f.rfind('[')+1 : f.rfind(']') ] )

            with open(fp, 'r') as f:
                reader = csv.reader(f)
                notedata = list(map(tuple, reader))
                difference = ((maxm+1)*192) - len(notedata)
                if difference > 0:
                    lastbpm = notedata[-1][1]
                    blank = [["0000",lastbpm] for i in range(difference)]
                    notedata += blank
            
            chts.append(notedata)
            outputs_df.loc[idx] = difficulty
            idx += 1
            bar.next()

    inputs_df = pd.DataFrame(chts, columns=[str(i) for i in range((maxm+1)*192)])
    return inputs_df, outputs_df


if __name__ == "__main__":
    inputs, outputs = create_nodes('./data/dataset/')
    dt = datetime.now().isoformat(timespec='minutes').replace(":",'-')
    # tuner = kt.Hyperband(model_builder,
    #     objective='val_accuracy',
    #     max_epochs=10,
    #     factor=3,
    #     directory='./data/model',
    #     project_name=dt)



    model = ak.StructuredDataClassifier(
        project_name=dt,
        directory='./data/model',
        max_trials=10)
    model.fit(x=inputs,y=outputs)
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

