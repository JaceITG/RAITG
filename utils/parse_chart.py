import os, sys

def main(fp):
    if os.path.splitext(fp)[1] != '.ssc':
        return ValueError("Stepfile must be an .ssc")
    
    #Read file
    lines = []
    with open(fp) as f:
        lines = f.readlines()

    bpmstr = ""
    readingbpm = False

    # List of difficulties in file [(start_index, difficulty)]
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
        
        if lines[i].startswith("#NOTEDATA"):
            startind = i
            
            #Seek to difficulty meter declaration
            while not lines[i].startswith("#METER"):
                i += 1
            
            clean = lines[i].strip(";\n")
            difficulty = int(clean[clean.index(":")+1:])
            charts.append((startind, difficulty))
            continue
    
    bpmstr = bpmstr.replace(',','\n')
    print(f"BPM breakdown:\n{bpmstr}")
    print()
    for c in charts:
        print(f"Chart at line {c[0]}: Difficulty {c[1]}")
            


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
