import matplotlib.pyplot as plt
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

epochs = 10
model_desc = "Averaged Default MHA"

# Create an array of AutoKeras StructuredDataInput nodes from the .cht files in dataset.
# Each input node in shape [ beat[ arrows,bpm ], beat[ arrows,bpm ], ...]
# Each output node as integer difficulty
def create_nodes(dataset, pad=0):
    global inputshape

    inputs = []
    outputs = []

    maxnotes = pad
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
                #blankend[0] = data[-1][0] + 1
                data += [blankend for i in range(maxnotes - len(data))]

                # print(f'{song.title} [{chart.difficulty}] Notes: {len(data)}')
                # for n in data[-10:]:
                #     print("Note:")
                #     print(n)

                inputs.append(np.vstack(data).T)
                outputs.append(int(chart.meter))
                chart_count += 1

                
            bar.next()

    print(f"{len(inputs)} nodes")

    inputshape = (len(inputs),np.shape(inputs[0]))
    outputs = np.array(outputs)
    return inputs, outputs, chart_count

def norm_outputs(output):
    maxl = max(output)
    minl = min(output)

    diff_range = maxl-minl + 0.00001 #scale down values to prevent 1.0 thresholds

    thresholds = np.array([(block-minl)/diff_range for block in output], dtype=np.float32)

    return thresholds

def graph(prediction, sampleout, wape):
    #Graph results
    fig, ax = plt.subplots()
    ax.scatter(sampleout, prediction)

    #find line of best fit
    a, b = np.polyfit(sampleout, prediction, 1)
    ax.plot(sampleout, a*sampleout+b)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, f"WAPE={wape:.2f}%", transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

    plt.xlabel("User-defined Difficulty")
    plt.ylabel("Predicted Difficulty")
    plt.title(f"{model_desc} trained for {epochs} epochs")
    plt.savefig(f"figures/{dt}.png")

if __name__ == "__main__":
    inputs, outputs, samples = create_nodes('./data/dataset/')

    inputs = np.reshape(inputs, (samples, -1, 10))
    trainlength = np.shape(inputs)[1]
    print(np.shape(inputs))

    normalized_outputs = norm_outputs(outputs)

    labels = sorted(np.unique(outputs))

    dt = datetime.now().isoformat(timespec='minutes').replace(":",'-')

    ### MODEL LAYERS ###
    attLayer = keras.layers.MultiHeadAttention(num_heads=2, key_dim=2, output_shape=(1), attention_axes=(1))

    input_seq = keras.Input(shape=(trainlength, 10))
    output_tensor = attLayer(input_seq, input_seq)
    pooling = keras.layers.GlobalAvgPool1D()
    out = pooling(output_tensor)
    
    ### UNUSED
    # normalization = keras.layers.BatchNormalization()
    #masked = keras.layers.Masking()(input_seq)
    #reshaped = keras.layers.Reshape((-1,10), input_shape=(6340,))(input_seq)
    # conv = keras.layers.Conv1D(10, 10, input_shape=(trainlength, 10), padding='causal')
    # input_seq = conv(input_seq)
    #output_tensor = keras.layers.LayerNormalization(axis=1)(output_tensor)


    model = keras.models.Model(inputs=input_seq, outputs=out)

    model.summary()

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)

    model.compile(    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["accuracy"],
    )
    
    model.fit(inputs, normalized_outputs, epochs=epochs)
    #model.train_on_batch(inputs, normalized_outputs)

    print(f"Model {model} created")
    print("~~~~~~~~~~~~~~~")

    print("Predicting Sample")
    samplein, sampleout, samples = create_nodes('./data/testset', pad=trainlength)

    for i in range(len(samplein)):
        samplein[i] = samplein[i][:,:trainlength]

    samplein = np.reshape(samplein, (samples, -1, 10))

    prediction = []
    for i in range(samples):
        pred = model.predict(np.array([samplein[i]]))
        prediction.append(pred[0])
    
    #prediction = model.predict_on_batch(samplein)

    print(f"\n\n###############PREDICTION################")
    print(prediction)

    #shift-positive if range of predictions spans pos and neg vals
    if min(prediction)>0 and max(prediction)>0:
        m = abs(min(prediction))
        prediction = [p+m for p in prediction]

    prediction = [abs(tensor[0]) for tensor in prediction]
    prediction = norm_outputs(prediction)
    diff_range = max(outputs) - min(outputs)
    prediction = np.array([(tensor*diff_range)+min(outputs) for tensor in prediction])

    #Calculate Weighted Average Percent Error
    wape = sum([abs(a - p) for a, p in zip(sampleout, prediction)]) / sum(sampleout)
    wape*=100

    graph(prediction, sampleout, wape)

    #Display the predicted difficulties alongside actual
    total_error = 0
    for i in range(len(prediction)):
        errorperc = ( abs(prediction[i] - sampleout[i]) / float(sampleout[i]) ) * 100
        print(f"Predicted {prediction[i]:02.1f}\tActual {sampleout[i]:02} (Error: {errorperc:.1f}%)")

        total_error += errorperc


    print()
    print(f"WAPE: {wape:.2f}%")

