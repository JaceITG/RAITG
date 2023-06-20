import pandas as pd
import numpy as np
import os
from datetime import datetime
from progress.bar import Bar

### Simfile ###
import simfile
from simfile.timing.engine import TimingEngine, TimingData
from simfile.notes import NoteData, NoteType
from simfile.notes.group import SameBeatNotes
import simfile.notes.count as Count

relevant_notes = [NoteType.TAP, NoteType.HOLD_HEAD, NoteType.ROLL_HEAD, NoteType.TAIL]

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

    maxnotes = 0
    chart_count = 0

    for f in os.listdir(dataset):
        fp = os.path.join(dataset, f)
        song = simfile.open(fp)

        for chart in song.charts:
            numnotes = Count.count_steps(\
                notes=NoteData(chart),\
                include_note_types=relevant_notes,\
                same_beat_notes=SameBeatNotes.KEEP_SEPARATE)
            
            maxnotes = max(maxnotes, numnotes)

    print(f"Maximum note count: {maxnotes}")

    with Bar('Loading nodes...',suffix='%(percent).1f%% - %(eta)ds',max=len(os.listdir(dataset))) as bar:
        for f in os.listdir(dataset):
            fp = os.path.join(dataset, f)
            song = simfile.open(fp)

            for chart in song.charts:
                #skip non-singles
                if chart.stepstype != 'dance-single':
                    continue

                #Instantiate timing and note data for the chart
                timing = TimingEngine(TimingData(song, chart))
                chart_data = NoteData(chart)
                notes = [(timing.time_at(note.beat), note.column, note.note_type, note.beat) for note in chart_data\
                         if note.note_type in relevant_notes]

                # print(f'{song.title} [{chart.difficulty}]')
                # last = None
                # for t in notes[-5:]:
                #     print(f'{t} Diff beat? {t[3]!=last[3] if last else "N/A"}')
                #     last = t


                data = []
                holds = np.zeros(4)
                last_time = notes[0][0] - 5 #default starting rest of 5 seconds 

                for note in notes:
                    #Create an nparray for attributes of the note
                    # time, time since last, 4 columns tap, 4 columns held
                    data += [np.array([0]*10, dtype=np.float32)]

                    #carry over current held notes
                    data[-1][6:] = holds

                    data[-1][0] = note[0]   #time
                    data[-1][1] = min((note[0] - last_time), 5)   #time since last note, cap extended rests at 5
                    last_time = note[0]

                    if note[2] == NoteType.TAP:
                        data[-1][2+note[1]] = 1

                    elif note[2] == NoteType.HOLD_HEAD or note[2] == NoteType.ROLL_HEAD:
                        data[-1][2+note[1]] = 1
                        holds[note[1]] = 1

                    elif note[2] == NoteType.TAIL:
                        holds[note[1]] = 0
                
                #pad chart for maximum note length
                blankend = np.array([0]*10, dtype=np.float32)
                blankend[0] = data[-1][0] + 1
                data += [blankend for i in range(maxnotes - len(data))]

                print(f'{song.title} [{chart.difficulty}] Notes: {len(data)}')
                # for n in data[-10:]:
                #     print("Note:")
                #     print(n)

                
            bar.next()

    return
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
    create_nodes('./data/dataset/')
else:
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

