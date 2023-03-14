import parse_chart, os

fp = os.path.abspath("../data/Songs/")

sscs = []

#iterate through pack folders
for pack in os.listdir(fp):
    packfp = os.path.join(fp, pack)
    print(f"in pack {pack}")
    for song in os.listdir(packfp):
        songfp = os.path.join(packfp, song)
        if os.path.isdir(songfp):
            print(f"in song {song}")
            songfp = os.path.join(packfp, song)
            for f in os.listdir(songfp):
                file = os.path.join(songfp,f)
                if os.path.isfile(file) and os.path.splitext(f)[1] == '.ssc':
                    sscs.append(file)


for sscpath in sscs:
    parse_chart.main(sscpath)
