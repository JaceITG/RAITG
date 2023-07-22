from automodel import train, predict
import sys, os, json

def help(module=None):
    if not module:
        print("First argument should be one of:")
        print("\ttrain\tCreate and fit a model to the dataset (saves model to folder)")
        print("\tpredict\tEvaluate chart(s) based on existing model")
        print("\tfull\tRun training and prediction phases sequentially")
    else:
        pass

if __name__ == "__main__":
    args = sys.argv[1:]

    dataset = "./data/dataset"
    testset = "./data/testset"
    save_model = False
    graph_results = True

    for a in args:
        if a.startswith('-'):
            if 's' in a:
                save_model = True
            if 'n' in a:
                graph_results = False

    if len(args)<1:
        help()
        sys.exit(0)

    if args[0].lower() == "train":
        name = None
        for a in args:
            if a.startswith("--name="):
                name = a[a.index('=')+1:]
        
        train(dataset, save=True, name=name)

    elif args[0].lower() == "predict":
        name = None
        for a in args:
            if a.startswith("--model="):
                name = a[a.index('=')+1:]

        #Find model to make prediction with
        models = os.listdir("./model/")
        if not name or name not in models:
            #Prefer most recent dated model (starts with digit)
            dated = [m for m in models if m[0].isdigit()]
            if len(dated) < 1 and len(models) > 0:
                #Select first names
                model = models[0]
            else:
                model = dated[-1]
        else:
            model = name

        if not model:
            raise ValueError("Model could not be found in \"models\" folder")
        
        #Obtain input metadata for the model
        with open(f"model/{model}/dataset.json", 'r') as f:
            data = json.load(f)
            
        print(f"Predicting charts in {testset} using model {model}...")
        predict(f"model/{model}/model.keras", data, testset, make_graph=graph_results)

    elif args[0].lower() == "full":
        pass
    else:
        help()