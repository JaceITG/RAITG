from progress.bar import Bar
import os
import numpy as np

### Simfile ###
import simfile
from simfile.timing.engine import TimingEngine, TimingData
from simfile.notes import NoteData, NoteType
from simfile.notes.group import SameBeatNotes
import simfile.notes.count as Count

relevant_notes = [NoteType.TAP, NoteType.HOLD_HEAD, NoteType.ROLL_HEAD, NoteType.TAIL]

# Create an array of AutoKeras StructuredDataInput nodes from the .cht files in dataset.
# Each input node in shape [ beat[ arrows,bpm ], beat[ arrows,bpm ], ...]
# Each output node as integer difficulty
def create_nodes(dataset, pad=0):

    inputs = []
    outputs = []
    names = []

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
                names.append(song.title)

                
            bar.next()

    print(f"{len(inputs)} nodes")

    outputs = np.array(outputs)
    return inputs, outputs, chart_count, names