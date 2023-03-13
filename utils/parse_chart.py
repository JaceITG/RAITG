import os, sys

# Take measure in a lower snap and return array of notes separated by
# counts of how many 192nd rest beats are between each
def conv_measure(measure):
    snap = len(measure)

    scalar = int(192/snap)

    rest = 0
    newlines = []
    for m in measure:
        if m == "0000":
            rest += scalar
            continue
        
        if rest>0:
            newlines.append(f"r{rest}")
            rest = 0

        newlines.append(m)
        rest += scalar - 1

    #Add remaining rest
    if rest>0:
            newlines.append(f"r{rest}")
            rest = 0
    
    return newlines

# For a chart (start_ind, name, difficulty) in lines, write a file
# containing metadata header followed by note data
# converted to 192nd notes
def format_diff(songname, lines, chart):
    with open(f"../data/{songname} [{chart[2]}].cht", 'w') as f:

        #Get first note line
        index = chart[0] + 1
        note = lines[index].strip('\n')

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
            
            #Convert to 192nd
            converted = conv_measure(measure)
            #Write to file
            for l in converted:
                f.write(f"{l}\n")
            
            #Pain
            if note.startswith(';'):
                break

            #Move to next measure
            index += 1
            note = lines[index].strip('\n')

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
        
        if lines[i].startswith("#DIFFICULTY"):
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

    
    bpmstr = bpmstr.replace(',','\n')
    print(f"BPM breakdown:\n{bpmstr}")
    print()
    for c in charts:
        print(f"Chart at line {c[0]}: {c[1]} {c[2]}")

    bpms = []
    # Convert BPM string to floating point list of timing points (beat#, bpm)
    for timingpoint in bpmstr.split(','):
        values = (float(v) for v in timingpoint.split('='))
        bpms.append(values)
    
    for c in charts:
        format_diff(os.path.splitext(fp)[0], lines, c)



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
