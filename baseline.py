import pickle, argparse, os, pyconll, dataloader, utils
import numpy  as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="./")
parser.add_argument("--file", type=str, default="./decision_tree_files.txt")
parser.add_argument("--features", type=str, default="Gender+Person+Number+Mood+Case+Tense", nargs='+')
parser.add_argument("--relation_map", type=str, default="./relation_map")
parser.add_argument("--output", type=str, default="./baseline.out")
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

def automated_metric(leaves, test_path, feature, feature_distribution,  test_samples, threshold, hard, traindata):
    f = test_path.strip()
    data = pyconll.load_from_file(f"{f}")
    test_agree_tuple, test_freq_tuple = utils.getTestData(data, feature)
    automated_evaluation_score = {}
    test_leaves_distr = {}
    #ITerating the leaves constructed from the training data

    #Assume all leaves have chance-agreement baseline
    for (class_, rules) in leaves:
        class_ = 'chance-agreement'
        test_leaf_agreement, test_leaf_disagree = 0, 0
        #(info, agree, disagree) = leafinfo
        tuples, train_total, freq_tuple = utils.retrivePossibleTuples(feature, rules, traindata)
        for tuple in tuples:
            tupledata = {'relation': None, 'head': None, 'child': None}
            tupledata['relation'], tupledata['head'], tupledata['child'] = tuple[0], tuple[1], tuple[2]
            #Check the test_percent
            if tuple in test_freq_tuple:
                test_percent  = test_agree_tuple[tuple] * 1.0 / test_freq_tuple[tuple]
                test_leaf_agreement += test_agree_tuple[tuple]
                test_leaf_disagree += test_freq_tuple[tuple] - test_agree_tuple[tuple]
                if class_ == 'agreement' and test_percent >= 0.95:
                    automated_evaluation_score[tuple] = 1
                elif class_ == 'chance-agreement' and test_percent < 0.95:
                    automated_evaluation_score[tuple] = 1
                else:
                    automated_evaluation_score[tuple] = 0



    total = 0
    correct = 0
    for tuple, count in automated_evaluation_score.items():
        correct += count
        total += 1
    metric = correct * 1.0/total
    return metric, test_leaves_distr

if __name__ == "__main__":
    relation_map = {}
    with open(args.relation_map, "r") as inp:
        for line in inp.readlines():
            relation_map[line.split(";")[0]] = (
                line.split(";")[1].lstrip().rstrip(), line.split(";")[-1].lstrip().rstrip())

    with open(args.file, "r") as inp:
        files = []
        for file in inp.readlines():
            if file.startswith("#"):
                continue
            files.append(file)
    args.features = args.features.split("+")


    with open(args.output, "w") as fout:
        fnum =0
        while fnum < len(files):

            treebank = files[fnum].strip()
            fnum += 1
            train_path, dev_path, test_path = None, None, None
            for [path, dir, inputfiles] in os.walk(treebank):
                for file in inputfiles:
                    if "-train.conllu" in file:
                        train_path = treebank + "/" + file
                        treebank_lang = train_path.strip().split('/')[-1].split("-")[ 0]

                        percent = 'all'
                        lang = train_path.strip().split('/')[-1].split("-")[ 0]  # + "_"+ train_path.strip().split("/")[-1].split("_")[0].split("-")[0]
                    if "dev.conllu" in file:
                        dev_path = treebank + "/" + file

                    if "test.conllu" in file:
                        test_path = treebank + "/" + file

            if train_path is None:
                continue
            lang_full = lang
            f = train_path.strip()
            data = pyconll.load_from_file(f"{f}")
            traindata = data
            data_loader = dataloader.DataLoader(args, relation_map)

            for feature in args.features:
                # Creating the vocabulary
                inputFiles = [train_path, dev_path, test_path]
                data_loader.readData(inputFiles)

                # creating the features for training
                # creating the features for training
                train_features, train_output_labels = data_loader.getBinaryFeatures(train_path, type="train",
                                                                                    p=1.0, shuffle=True)
                test_features, test_output_labels = data_loader.getBinaryFeatures(test_path, type="test", p=1.0,
                                                                                  shuffle=False)

                if feature not in test_features:
                    continue
                leaffile = f'{args.input}/1_{treebank_lang}_{feature}_all_9.pkl'
                if os.path.exists(leaffile):
                    test_samples = len(test_features[feature])
                    leaves = pickle.load(open(leaffile,'rb'))

                    ARM, _ = automated_metric(leaves, test_path, feature,
                                                        data_loader.feature_distribution[feature],
                                                        test_samples, threshold=0.01,
                                                        hard=False, traindata=traindata)
                    print(f"{treebank_lang},{feature},{ARM}")

