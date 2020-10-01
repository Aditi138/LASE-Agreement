from collections import defaultdict
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="transfer.log")
parser.add_argument("--type", type=str, default="test")
args=parser.parse_args()

if __name__ == "__main__":
    with open(args.input, "r") as fin:
        lines = fin.readlines()
        lang_feature = {}
        ARM, p, lengths = {}, '50', {}
        var_ARM = {}
        for line in lines:
            if not line.startswith(args.type): #For getting test results only
                continue
            if line == "" or line == "\n" or line.startswith("error"):
                continue
            line = line.replace(args.type,"")
            info = line.strip().split(", ")
            prefix = info[0].split('-')
            length = int(info[1])

            if prefix[0] != p:  # print the average values
                for lang, _ in ARM.items():
                    print(lang, p)
                    for feature, _ in ARM[lang].items():
                        denom = lang_feature[lang][feature]
                        arm = np.round(ARM[lang][feature] * 1.0 / denom, 3)
                        if arm > 1:
                            arm = 'NA'
                        l = np.round(lengths[lang][feature] * 1.0 / denom, 3)
                        print(feature + "," + str(arm) + "+/-" + str(np.round(np.var(var_ARM[lang][feature]),3)))
                WDS, ARM, ADS, p, lengths = {}, {}, {}, '0', {}
                var_ARM = {}

            seed = prefix[1]

            lang = info[2].split("-")[0]
            feature = info[3]

            arm = float(info[4].replace("NA", "1.0"))
            if lang not in ARM:
                ARM[lang] = defaultdict(lambda: 0)
                lang_feature[lang] = defaultdict(lambda :0)
                lengths[lang] = defaultdict(lambda : 0)
                var_ARM[lang] = defaultdict(list)

            ARM[lang][feature] += arm
            lang_feature[lang][feature] += 1
            lengths[lang][feature] += length
            p = prefix[0]
            var_ARM[lang][feature].append(arm)

    p='0'
    if prefix[0] != p:  # print the average values
        for lang, info in ARM.items():
            print(lang, prefix[0])
            for feature, _ in ARM[lang].items():
                denom=lang_feature[lang][feature]
                arm = np.round(ARM[lang][feature] * 1.0 / denom, 3)
                l = np.round(lengths[lang][feature] * 1.0 / denom, 3)

                if arm > 1:
                    arm = 'NA'

                print(feature + "," + str(arm) + "+/-" + str(np.round(np.var(var_ARM[lang][feature]),3)))
