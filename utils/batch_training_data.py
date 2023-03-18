import parse_chart, os
from progress.bar import Bar

fp = os.path.abspath("../data/Songs/")

sscs = []
count = 0

#clear dataset directory
for f in os.listdir("../data/dataset/"):
    os.remove(os.path.join("../data/dataset/",f))

#iterate through pack folders
for pack in os.listdir(fp):
    packfp = os.path.join(fp, pack)
    for song in os.listdir(packfp):
        songfp = os.path.join(packfp, song)
        if os.path.isdir(songfp):
            songfp = os.path.join(packfp, song)
            for f in os.listdir(songfp):
                file = os.path.join(songfp,f)
                if os.path.isfile(file) and os.path.splitext(f)[1] == '.ssc':
                    sscs.append(file)
                    count += 1

with Bar('Parsing...',suffix='%(percent).1f%% - %(eta)ds',max=count) as bar:
    for i in range(len(sscs)):
        parse_chart.main(sscs[i])
        bar.next()
