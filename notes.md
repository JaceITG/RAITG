# Notes

## CHART PARSING

- Convert all patterns to 192nd quantization by inserting 0000 lines
- Could also denote 0000 lines as a sentinel integer denoting N 192nds of rest between each note
- Strip leading and trailing empty measures?
- May have to adjust how holds/rolls and mines are represented

## Type of model

- DNN? CNN?
- Deep Neural Network with many hidden layers between input and output data
- From Pandas DataFrame: chart diff, notedata[ beat[ arrows,bpm ], beat[ arrows,bpm ], beat[ arrows,bpm ], ...]
