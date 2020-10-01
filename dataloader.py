import pyconll
import utils
import numpy as np
np.random.seed(1)
from collections import defaultdict
import codecs
import os
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import json


class DataLoader(object):
    def __init__(self, args, relation_map):
        self.args = args
        self.pos_dictionary = {}
        self.feature_dictionary = {}
        self.relation_dictionary = {}
        self.used_relations = set()
        self.used_head_pos = set()
        self.used_child_pos = set()
        self.used_tuples = defaultdict(set)
        self.class_relations = defaultdict(set)
        self.class_headpos = defaultdict(set)
        self.class_childpos = defaultdict(set)
        self.relation_map = relation_map
        random.seed(args.seed)

    def unison_shuffled_copies(self,a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def readData(self, inputFiles):
        na = set()
        for inputFile in inputFiles:
            if inputFile == None:
                continue
            lang = inputFile.strip().split('/')[-1].split('_')[0]
            self.lang_full = inputFile.strip().split('/')[-2].split('-')[0][3:]
            f = inputFile.strip()
            data = pyconll.load_from_file(f"{f}")

            for sentence in data:
                for token in sentence:
                    token_id = token.id
                    if "-" in token_id:
                        continue
                    if token.deprel is None:
                        continue
                    relation = token.deprel.lower()
                    pos = token.upos
                    feats = token.feats
                    if pos is None:
                        pos = "None"
                    if relation is None:
                        relation  = "None"

                    if relation not in self.relation_map:
                        na.add(relation)

                    if pos not in self.pos_dictionary:

                        self.pos_dictionary[pos] = len(self.pos_dictionary)
                    for feat in feats:
                        if feat not in self.feature_dictionary:
                            self.feature_dictionary[feat] = len(self.feature_dictionary)
                    if relation not in self.relation_dictionary:
                        self.relation_dictionary[relation] = len(self.relation_dictionary)

        self.relation_id2tag = {v: k for k, v in self.relation_dictionary.items()}
        self.pos_id2tag = {v: k for k, v in self.pos_dictionary.items()}
        self.feature_id2tag = {v: k for k, v in self.feature_dictionary.items()}

    def getBinaryFeatures(self, inputFile, type, p,  shuffle ):
        f = inputFile.strip()
        data = pyconll.load_from_file(f"{f}")
        all_features = defaultdict(list)
        output_labels = defaultdict(list)
        index = [i for i in range(len(data))]

        if type == "train":
            self.feature_distribution = {}
            self.train_random_samples =  random.sample(index, k= int(len(data) * p))
            random_samples = deepcopy(self.train_random_samples)
        else:
            random_samples = [i for i in range(len(data))]

        for sentence_num in random_samples:
            sentence = data[sentence_num]
            for token in sentence:
                token_id = token.id
                if "-"in token_id:
                    continue
                if token.deprel is None:
                    continue
                relation = token.deprel.lower()
                pos = token.upos
                feats = token.feats

                if token.head and token.head != "0":
                    head_pos = sentence[token.head].upos
                    head_feats = sentence[token.head].feats
                    shared, agreed = utils.find_agreement(feats, head_feats)

                    one_hpos_feature = np.zeros((len(self.pos_dictionary),), dtype=int)
                    one_cpos_feature = np.zeros((len(self.pos_dictionary),), dtype=int)
                    one_rel_feature = np.zeros((len(self.relation_dictionary),), dtype=int)

                    one_hpos_feature[self.pos_dictionary[head_pos]] = 1
                    one_cpos_feature[self.pos_dictionary[pos]] = 1
                    one_rel_feature[self.relation_dictionary[relation]] = 1

                    for f in shared:
                        attribute_present = np.zeros((2,), dtype=int)
                        attribute_present[0] = 1 #cpos
                        attribute_present[1] = 1 #hpos
                        one_feature = np.concatenate(
                            (one_hpos_feature, one_cpos_feature, one_rel_feature, attribute_present), axis=None)

                        if f in self.args.features:
                            self.used_relations.add(relation)
                            self.used_head_pos.add(head_pos)
                            self.used_child_pos.add(pos)
                            all_features[f].append(one_feature)
                            if type == "train":
                                if f not in self.feature_distribution:
                                    self.feature_distribution[f] = defaultdict(lambda: 0)
                                self.feature_distribution[f][",".join(list(feats[f]))] += 1
                            if f in agreed:
                                output_labels[f].append(1)
                                if type == "train":
                                    self.class_relations[1].add(relation)
                                    self.class_headpos[1].add(head_pos)
                                    self.class_childpos[1].add(pos)
                                    #self.used_tuples[1].add(relation,head_pos,pos)
                            else:
                                output_labels[f].append(0)
                                if type == "train":
                                    self.class_relations[0].add(relation)
                                    self.class_headpos[0].add(head_pos)
                                    self.class_childpos[0].add(pos)
                                    #self.used_tuples[0].add(relation, head_pos, pos)

        if shuffle:
            shuffled_features = {}
            shuffled_output_labels = {}

            for feature, data in all_features.items():
                x,y = self.unison_shuffled_copies(np.array(data), np.array(output_labels[feature]))
                assert len(x) == len(y)
                shuffled_features[feature] = x
                shuffled_output_labels[feature] = y
            return shuffled_features, shuffled_output_labels

        shuffled_features = {}
        shuffled_output_labels = {}

        for feature in all_features.keys():
            assert len(all_features[feature]) == len(output_labels[feature])
            shuffled_features[feature] = np.array(all_features[feature])
            shuffled_output_labels[feature] = np.array(output_labels[feature])

        return shuffled_features, shuffled_output_labels

    def getFeatures(self, inputFile, lang, type, shuffle=False):
        f = inputFile.strip()
        data = pyconll.load_from_file(f"{f}")
        all_features = defaultdict(list)
        output_labels = defaultdict(list)
        name = os.path.basename(f)
        filepointers = {}
        for feature in self.args.features:
            #print(self.args.debug_folder + "/" + lang + "/" + type +"_" + feature  + ".txt")
            filepointers[feature] = codecs.open(self.args.debug_folder + "/" + lang + "/" + type +"_" + feature  + ".txt", "w", encoding='utf-8')


        for sentence in data:
            for token in sentence:
                token_id = token.id
                if token.deprel is None:
                    continue
                relation = token.deprel.lower()
                pos = token.upos
                feats = token.feats

                if token.head and token.head != "0":
                    head_pos = sentence[token.head].upos
                    head_feats = sentence[token.head].feats
                    shared, agreed = utils.find_agreement(feats, head_feats)

                    one_feature = np.zeros((3,), dtype=int)
                    one_feature[0] = self.pos_dictionary[pos]
                    one_feature[1] = self.pos_dictionary[head_pos]
                    one_feature[2] = self.relation_dictionary[relation]

                    for f in shared:
                        if f not in self.args.features:
                            continue

                        self.used_relations.add(relation)
                        self.used_head_pos.add(head_pos)
                        self.used_child_pos.add(pos)

                        all_features[f].append(one_feature)
                        feats_text, head_feats_text = "", ""
                        for key, valye in feats.items():
                            feats_text += key + "=" + str(list(valye)) + " , "
                        for key, valye in head_feats.items():
                            head_feats_text += key + "=" + str(list(valye)) + " , "

                        if f in agreed:
                            output_labels[f].append(1)
                            filepointers[f].write(str(1) + "\t" + relation + "\t" + pos + "\t" +feats_text + "\t" +   head_pos + "\t" + head_feats_text + "\n")
                            if type == "train":
                                self.class_relations[1].add(relation)
                                self.class_headpos[1].add(head_pos)
                                self.class_childpos[1].add(pos)

                        else:
                            output_labels[f].append(0)
                            filepointers[f].write(
                                str(0) + "\t" + relation + "\t" + pos + "\t" + feats_text + "\t" +   head_pos + "\t" + head_feats_text + "\n")
                            if type == "train":
                                self.class_relations[0].add(relation)
                                self.class_headpos[0].add(head_pos)
                                self.class_childpos[0].add(pos)

                    for f in self.args.features:
                        if f not in shared:
                            self.used_relations.add(relation)
                            self.used_head_pos.add(head_pos)
                            self.used_child_pos.add(pos)

                            all_features[f].append(one_feature)
                            output_labels[f].append(2)
                            feats_text, head_feats_text = "", ""
                            for key, valye in feats.items():
                                feats_text += key + "=" +  str(list(valye)) + " , "
                            for key, valye in head_feats.items():
                                head_feats_text += key + "=" + str(list(valye)) + " , "
                            filepointers[f].write(
                                str(2) + "\t" + relation + "\t" + pos + "\t" + feats_text + "\t" + head_pos   + "\t" + head_feats_text + "\n")

                            if type == "train":
                                self.class_relations[2].add(relation)
                                self.class_headpos[2].add(head_pos)
                                self.class_childpos[2].add(pos)



        if shuffle:
            shuffled_features = {}
            shuffled_output_labels = {}
            for feature, data in all_features.items():
                x,y = self.unison_shuffled_copies(np.array(data), np.array(output_labels[feature]))
                shuffled_features[feature] = x
                shuffled_output_labels[feature] = y
            return shuffled_features, shuffled_output_labels

        shuffled_features = {}
        shuffled_output_labels = {}
        for feature in all_features.keys():
            shuffled_features[feature] = np.array(all_features[feature])
            shuffled_output_labels[feature] = np.array(output_labels[feature])

        return shuffled_features, shuffled_output_labels

    def printData(self, language, data, output_label, feature, type):
        dir = self.args.debug_folder + "/" + language + "/"
        try:
            os.system("mkdir -p %s" % dir)
        except:
            print("Unable to create: ", dir)

        with codecs.open( dir +  type + "_" + feature + ".txt", "w", encoding='utf-8') as fout:
            for d, o in zip(data, output_label):
                fout.write(self.pos_id2tag[d[1]] + "\t" + self.pos_id2tag[d[0]] + "\t" + self.relation_id2tag[d[2]] + "\t" + str(o) + "\n")

    def getHistogram(self, folder_name, lang, input_path, feature):
        f = input_path.strip()
        data = pyconll.load_from_file(f"{f}")
        self.feature_tokens, self.feature_forms = {}, {}
        self.feature_tokens[feature] = defaultdict(lambda: 0)
        self.feature_forms = {}
        self.feature_forms_num = {}
        tokens, feature_values, pos_values= [], [], []
        self.lemma, self.lemmaGroups, self.lemma_freq, self.lemma_inflection = {}, defaultdict(set), {}, {}
        pos_barplots = {}
        features_set, pos_count = set(), defaultdict(lambda : 0)

        for sentence_num in self.train_random_samples:
            sentence = data[sentence_num]
            for token in sentence:
                if token.form == None or "-" in token.id:
                    continue

                token_id = token.id
                relation = token.deprel
                pos = token.upos
                if pos == None:
                    pos = 'None'
                feats = token.feats
                lemma = token.lemma

                tokens.append(token.form)
                pos_values.append(pos)

                pos_count[pos] += 1
                self.lemma[token.form.lower()] = lemma
                self.lemmaGroups[lemma].add(token.form.lower())
                if pos not in self.feature_forms_num:
                    self.feature_forms_num[pos] = {}
                    pos_barplots[pos] = defaultdict(lambda : 0)

                if pos not in self.lemma_inflection:
                    self.lemma_freq[pos] = defaultdict(lambda: 0)
                    self.lemma_inflection[pos] = {}

                if lemma:
                    self.lemma_freq[pos][lemma.lower()] += 1
                if lemma and lemma.lower() not in self.lemma_inflection[pos]:
                    self.lemma_inflection[pos][lemma.lower()] = {}
                # Aggregae morphology properties of required-properties - feature
                morphology_props = set(self.args.features) - set([feature])
                morphology_prop_values = []
                for morphology_prop in morphology_props:
                    if morphology_prop in feats:
                        morphology_prop_values.append(",".join(feats[morphology_prop]))
                morphology_prop_values.sort()
                inflection = ";".join(morphology_prop_values)
                if lemma and inflection not in self.lemma_inflection[pos][lemma.lower()]:
                    self.lemma_inflection[pos][lemma.lower()][inflection] = {}
                if feature in feats:
                    values = list(feats[feature])
                    values.sort()
                    feature_values.append(",".join(values))
                    #for feat in values:
                else:
                    values = ['NA']
                    feature_values.append("NA")

                for feat in values:
                    features_set.add(feat)
                    pos_barplots[pos][feat] += 1
                    if feat not in self.feature_forms_num[pos]:
                        self.feature_forms_num[pos][feat] = defaultdict(lambda : 0)

                    self.feature_forms_num[pos][feat][token.form.lower()] += 1
                    if lemma:
                        self.lemma_inflection[pos][lemma.lower()][inflection][feat] = token.form.lower()

        #sort the pos by count
        sorted_pos = sorted(pos_count.items(), key= lambda kv: kv[1], reverse=True)
        pos_to_id, pos_order = {}, []
        for (pos, _) in sorted_pos:
            pos_to_id[pos] = len(pos_to_id)
            pos_order.append(pos)

        #Stacked histogram
        #sns.set()
        fig, ax = plt.subplots()
        bars_num = np.zeros((len(features_set), len(pos_barplots)))
        x_axis = []
        feat_to_id, id_to_feat = utils.get_vocab_from_set(features_set)

        for pos in pos_order:
            feats = pos_barplots[pos]
            x_axis.append(pos)
            pos_id = pos_to_id[pos]
            for feat, num in feats.items():
                feat_id = feat_to_id[feat]
                bars_num[feat_id][pos_id] = num

        r = [i for i in range(len(pos_to_id))]
        handles, color = [], ['steelblue', 'orange', 'olivedrab', 'peru', 'seagreen', 'chocolate',
                              'tan', 'lightseagreen', 'green', 'teal','tomato','lightgreen','yellow','lightblue','azure','red',
                              'aqua', 'darkgreen', 'tomato', 'firebrick', 'khaki', 'gold', 'powderblue',  'navy', 'plum' ]
        bars = np.zeros((len(pos_barplots)))
        for barnum in range(len(features_set)):
            plt.bar(r, bars_num[barnum], bottom=bars, color=color[barnum], edgecolor='white', width=1)
            handles.append(mpatches.Patch(color=color[barnum], label=id_to_feat[barnum]))
            bars += bars_num[barnum]

        handles.reverse()
        # Custom X axis
        plt.xticks(r, x_axis,rotation=45, fontsize=9)
        #plt.xlabel("pos")
        plt.ylabel("Number of Tokens")
        plt.legend(handles=handles)

        right_side = ax.spines["right"]
        right_side.set_visible(False)

        top_side = ax.spines["top"]
        top_side.set_visible(False)

        plt.savefig(f"./{folder_name}/" + lang + "/" + feature + "/pos.png", transparent=True)
        plt.close()
        return pos_order















