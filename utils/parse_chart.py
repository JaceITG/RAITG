import os, sys

# For a chart (start_ind, name, difficulty) in lines, write a file
# containing metadata header followed by note data
# converted to 192nd notes
def format_diff(lines, chart):
    pass

# Read a file and extract information about bpm and difficulties
def main(fp):
    if os.path.splitext(fp)[1] != '.ssc':
        return ValueError("Stepfile must be an .ssc")
    
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
    # Convert BPM string to floating point list of timing points (time, bpm)
    for timingpoint in bpmstr.split(','):
        values = (float(v) for v in timingpoint.split('='))
        bpms.append(values)
            


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
