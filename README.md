# RAITG
AI Difficulty estimation for In The Groove (ITG) technical charts as investigated by [AITG: A Machine Learning Approach to ITG Difficulty Analysis](https://github.com/JaceITG/RAITG/blob/main/Report.pdf)

# Usage
After cloning/downloading the project repository and installing the required libraries as listed in `requirements.txt`;
Run
`python main.py`
for a list of available commands and arguments.

## Example
To execute a full training and prediction run of a new model named `MyModel`, while saving a copy of the model, run:
`python main.py full --name=MyModel -s`

This will train the model for a default 10 epochs (configurable via the `--epochs=` argument) on any charts located in `./data/dataset/`, and start a prediction run using charts in `./data/testset/`.
The per-chart prediction results will be printed to the terminal and a csv in the model directory, along with a graph comparing the user and machine generated values for the run.

