import parse_chart, os, sys, shutil
from progress.bar import Bar


def batch(fp, output):
    sscs = []
    count = 0 

    #clear output directory
    # for f in os.listdir(output):
    #     os.remove(os.path.join(output,f))

    #iterate through pack folders
    for pack in os.listdir(fp):
        packfp = os.path.join(fp, pack)
        for song in os.listdir(packfp):
            songfp = os.path.join(packfp, song)
            if os.path.isdir(songfp):
                songfp = os.path.join(packfp, song)
                for f in os.listdir(songfp):
                    file = os.path.join(songfp,f)
                    if os.path.isfile(file) and os.path.splitext(f)[1] in ['.ssc', '.sm']:
                        if os.path.splitext(f)[0] in sscs:
                            #Already collected other simfile of song
                            continue
                        sscs.append(file)
                        count += 1
                        shutil.copyfile(file, f'../data/dataset/{f}')


    ###OLD PARSER###
    return

    with Bar('Parsing...',suffix='%(percent).1f%% - %(eta)ds',max=count) as bar:
        for i in range(len(sscs)):
            parse_chart.main(sscs[i], output_dir=output)
            bar.next()


if __name__ == "__main__":
        
    fp = "C:/Users/Jace/Desktop/Code/RAITG/data/Training Songs"
    output = None

    #Parse args
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('-i'):
            i+=1
            fp = sys.argv[i].strip("\'\"")
        
        if sys.argv[i].startswith('-o'):
            i+=1
            output = sys.argv[i].strip("\'\"")

    print(output)
    print(fp)
    
    batch(fp, output)
    

