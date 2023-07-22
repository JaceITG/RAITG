import matplotlib.pyplot as plt
import numpy as np
import os, json
from datetime import datetime

from utils.process_charts import create_nodes

### Keras Setup ###
from tensorflow import keras
inputshape = (0,0)

dt = datetime.now().isoformat(timespec='minutes').replace(":",'-')

model_desc = ""

def norm_outputs(output):
    maxl = max(output)
    minl = min(output)

    diff_range = maxl-minl + 0.00001 #scale down values to prevent 1.0 thresholds

    thresholds = np.array([(block-minl)/diff_range for block in output], dtype=np.float32)

    return thresholds

def graph(prediction, sampleout, wape, meta):
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
    plt.title(f"{meta['model_desc']}")
    plt.savefig(f"model/{meta['name']}/graph.png")

def compile(nodes):
    global model_desc

    ### MODEL LAYERS ###
    model_desc = "Averaged MHA"
    nonneg = keras.constraints.NonNeg()
    attLayer = keras.layers.MultiHeadAttention(
        num_heads=2,
        key_dim=2,
        value_dim=4,
        output_shape=(1),
        attention_axes=(2),
        kernel_constraint=nonneg)

    input_seq = keras.Input(shape=(nodes, 10))
    output_tensor = attLayer(input_seq, input_seq)
    pooling = keras.layers.GlobalAvgPool1D()
    out = pooling(output_tensor)
    
    ### UNUSED
    # conv = keras.layers.Conv1DTranspose(1, 3, input_shape=(trainlength, 10), data_format='channels_last')
    # convoluted = conv(input_seq)
    # normalization = keras.layers.BatchNormalization()
    #reshaped = keras.layers.Reshape((-1,10), input_shape=(6340,))(input_seq)
    #output_tensor = keras.layers.LayerNormalization(axis=1)(output_tensor)


    model = keras.models.Model(inputs=input_seq, outputs=out)

    model.summary()

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)

    model.compile(    loss=keras.losses.Huber(),
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["accuracy"],
    )

    return model

def train(dataset, save=False, name=None, epochs=10):
    if not name:
        name = dt

    inputs, outputs, samples, names = create_nodes(dataset)

    inputs = np.reshape(inputs, (samples, -1, 10))
    trainlength = np.shape(inputs)[1]
    print(np.shape(inputs))

    normalized_outputs = norm_outputs(outputs)

    model = compile(trainlength)
    outputs = [str(o) for o in outputs]
    data = {'trainlength': trainlength,
            'outputs': ','.join(outputs),
            'model_desc': model_desc,
            'name': name}
    
    model.fit(inputs, normalized_outputs, epochs=epochs)

    os.makedirs(f"./model/{name}/")
    if save:
        model.save(f"./model/{name}/model.keras")
        
        with open(f"./model/{name}/dataset.json", 'w') as f:
            json.dump(data, f)

    print(f"Model {model} created")
    print("~~~~~~~~~~~~~~~")
    return model, data

def predict(model, data, testset, make_graph=True):
    #Load saved model
    if type(model) == str:
        model = keras.models.load_model(model)

    print("Predicting Sample")
    trainlength = data['trainlength']
    outputs = [int(o) for o in data['outputs'].split(',')]
    samplein, sampleout, samples, names = create_nodes(testset, pad=trainlength)

    for i in range(len(samplein)):
        samplein[i] = samplein[i][:,:trainlength]

    samplein = np.reshape(samplein, (samples, -1, 10))

    prediction = []
    for i in range(samples):
        pred = model.predict(np.array([samplein[i]]))
        prediction.append(pred[0])
    
    #prediction = model.predict_on_batch(samplein)

    print(f"\n\n###############PREDICTION################")

    #shift-positive if range of predictions spans pos and neg vals
    if min(prediction)>0 and max(prediction)>0:
        m = abs(min(prediction))
        prediction = [p+m for p in prediction]

    prediction = [abs(tensor[0]) for tensor in prediction]
    prediction = norm_outputs(prediction)
    diff_range = max(outputs) - min(outputs)
    prediction = np.array([(tensor*diff_range)+min(outputs) for tensor in prediction])

    #Calculate Weighted Average Percent Error
    # WAPE = ( Σ|Actual-Predicted| / Σ|Actual| ) * 100
    wape = sum([abs(a - p) for a, p in zip(sampleout, prediction)]) / sum(sampleout)
    wape*=100

    if make_graph:
        graph(prediction, sampleout, wape, data)

    #Display the predicted difficulties alongside actual
    total_error = 0
    for i in range(len(prediction)):
        errorperc = ( abs(prediction[i] - sampleout[i]) / float(sampleout[i]) ) * 100
        print(f"Predicted {prediction[i]:03.1f}\tActual {sampleout[i]:02} (Error: {errorperc:.1f}%) for {names[i]}")

        total_error += errorperc

    print()
    print(f"WAPE: {wape:.2f}%")

