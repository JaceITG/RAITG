# Notes

## Chart Parsing

- Convert all patterns to 192nd quantization by inserting 0000 lines
- Could also denote 0000 lines as a sentinel integer denoting N 192nds of rest between each note
- Strip leading and trailing empty measures?
- May have to adjust how holds/rolls and mines are represented

## Type of model

- DNN? CNN?
- Deep Neural Network with many hidden layers between input and output data
- From Pandas DataFrame: chart diff, notedata[ beat[ arrows,bpm ], beat[ arrows,bpm ], beat[ arrows,bpm ], ...]
- in which: arrows are categorical, bpm numerical

## Dataset

- Dimocracy, Notice Me Benpai
- Preprocessing: Need identification of beat points as (categorical, numerical) data
- Must be context-dependent with respect to column index in row
- Use string lookup for categorical beat type

```vocab = ["a", "b", "c", "d"]
data = tf.constant([["a", "c", "d"], ["d", "z", "b"]])
layer = layers.StringLookup(vocabulary=vocab)
vectorized_data = layer(data)
print(vectorized_data)

tf.Tensor(
[[1 3 4]
 [4 0 2]], shape=(2, 3), dtype=int64)```
