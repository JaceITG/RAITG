import os, sys

# Take measure in a lower snap and return array of notes separated by
# counts of how many 192nd rest beats are between each
def conv_measure(measure, bpms, meas_number):
    snap = len(measure)

    scalar = int(192/snap)

    newlines = []
    beat = 0.0
    bpm = 0
    relevant_changes = []

    #find starting bpm of measure
    for change in bpms:
        if change[0] < meas_number*4:
            #if bpm change happened before this measure
            bpm = change[1]
        elif change[0] - meas_number*4 <= 4:
            #if bpm will change in this measure
            relevant_changes.append(change)
            break

    #write notes from measure
    for m in measure:
        newlines.append(f"{m},{bpm}")

        #if bpm changed on/before current beat
        if len(relevant_changes)>0 and relevant_changes[0][0] <= (meas_number*4 + beat):
            change = relevant_changes.pop(0)
            bpm = change[1]

        #Progress beat by scalar/48 for appropriate num 192nd notes
        beat += scalar/48
        #Append an empty note for each 192nd rest
        for i in range(scalar - 1):
            newlines.append(f"0000,{bpm}")
    
    return newlines

# For a chart (start_ind, name, difficulty) in lines, write a file
# containing metadata header followed by note data
# converted to 192nd notes
def format_diff(songname, lines, chart, bpms):
    with open(f"../data/dataset/{songname} [{chart[2]}].cht", 'w') as f:

        #Get first note line
        index = chart[0] + 1
        note = lines[index].strip('\n')
        meas_number = 0

        #While end of chart (;) not found
        while not note.startswith(';'):
            #Collect measure
            measure = []
            while not note.startswith(','):
                if note.startswith(";"):
                    #Why would you end a chart before closing a measure, you're killing me .ssc
                    break

                if note.startswith('//'):
                    #ignore comments
                    index += 1
                    note = lines[index].strip('\n')
                    continue

                measure.append(note)
                index += 1
                note = lines[index].strip('\n')
            
            #Check for bpm changes
            # relevant_changes = []
            # for change in bpms:
            #     #If beat index is within 4 beats after this measure
            #     change_bpm = change[0] - meas_number*4
            #     if change_bpm >= 4 or change_bpm < 0:
            #         change_bpm = None
            #     else:
            #         #Add (beats after start measure, new bpm)
            #         relevant_changes.append((change_bpm, change[1]))
            # if len(relevant_changes) < 1:
            #     relevant_changes = None

            #Convert to 192nd
            converted = conv_measure(measure, bpms, meas_number)
            #Write to file
            for l in converted:
                f.write(f"{l}\n")
            
            #Pain
            if note.startswith(';'):
                break

            #Move to next measure
            index += 1
            note = lines[index].strip('\n')
            meas_number += 1
    
    with open("../data/dataset/meascounts.txt", 'a') as f:
        f.write(f"{meas_number},")

# Read a file and extract information about bpm and difficulties
def main(fp):
    if os.path.splitext(fp)[1] != '.ssc':
        raise Exception("Stepfile must be an .ssc")
    
    #Read file
    lines = []
    with open(fp) as f:
        lines = f.readlines()

    bpmstr = ""
    readingbpm = False

    # List of difficulties in file [(start_index, name, difficulty)]
    charts = []
    for i in range(len(lines)):
        if readingbpm:
            clean = lines[i].strip(';\n')
            if clean.startswith('#') or len(clean) < 1:
                readingbpm = False
            else:
                bpmstr += clean
                continue

        if lines[i].startswith("#BPMS"):
            #add line value to bpm string
            clean = lines[i].strip(';\n')
            end = clean.index(":")+1
            bpmstr += clean[end:]
            readingbpm = True
            continue
        
        if lines[i].startswith("#STEPSTYPE:dance-single"):
            while not lines[i].startswith("#DIFFICULTY"):
                i += 1
            clean = lines[i].strip(";\n")
            name = clean[clean.index(":")+1:]

            #Seek to difficulty meter declaration
            while not lines[i].startswith("#METER"):
                i += 1
            clean = lines[i].strip(";\n")
            difficulty = int(clean[clean.index(":")+1:])

            while not lines[i].startswith("#NOTES"):
                i += 1
            
            charts.append((i, name, difficulty))
            continue

    

    #print(bpmstr)
    bpms = []
    # Convert BPM string to floating point list of timing points (beat#, bpm)
    for timingpoint in bpmstr.split(','):
        values = tuple(float(v) for v in timingpoint.split('='))
        bpms.append(values)

    # bpmstr = bpmstr.replace(',','\n')
    # print(f"BPM breakdown:\n{bpmstr}")
    # print()
    # for c in charts:
    #     print(f"Chart at line {c[0]}: {c[1]} {c[2]}")

    for c in charts:
        fname = os.path.basename(fp)
        format_diff(os.path.splitext(fname)[0], lines, c, bpms)



if __name__ == "__main__":
    args = sys.argv

    if '-h' in args or '--help' in args:
        print("""ITG Chart Parsing for AI analysis
        \tUsage: python3 parse_chart.py [filename]
        """)
        sys.exit(0)
    
    if len(args) < 2 or not os.path.isfile(args[1]):
        print("First argument must be a valid path to a stepfile")
        sys.exit(1)
    
    main(args[1])
