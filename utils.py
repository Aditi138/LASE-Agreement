import numpy as np
import codecs

np.random.seed(1)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from collections import defaultdict
import pyconll
from copy import deepcopy
from scipy.stats import chisquare

def convertStringToset(data):
    feats = {}
    if data == "_":
        return {}
    for f in data.split("|"):
        k = f.split("=")[0]
        v = f.split("=")[1]
        feats[k]=v
    return feats

def find_agreement(feats1, feats2):
    shared = set()
    agreed = set()
    for feat in feats1:
        if feat in feats2:
            shared.add(feat)
            if feats1[feat] == feats2[feat]:
                agreed.add(feat)
    return shared, agreed

def print_examples(data, rel, pos1, pos2, agreement, ag_examples, dis_examples):
    print(f"Relation: {rel} and POS: {pos1}--{pos2} on {','.join(agreement)}")
    print("\tAgreeing examples:")
    for ex in ag_examples:
        print('\t\t', data[ex[0]].text)
        sentid = int(ex[0])
        tokid = ex[1]
        print(
            f"\t\t\tID: {tokid} token: {data[sentid][tokid].form}\t{data[sentid][tokid].upos}\t{data[sentid][tokid].feats}")
        print(data[sentid][tokid].conll())
        headid = data[sentid][tokid].head
        print(
            f"\t\t\tID: {headid} token: {data[sentid][headid].form}\t{data[sentid][headid].upos}\t{data[sentid][headid].feats}")
        print(data[sentid][headid].conll())
    if not dis_examples:
        print("\tNo disagreeing examples")
    else:
        print("\tDisagreeing examples:")
        for ex in dis_examples:
            print('\t\t', data[ex[0]].text)
            sentid = int(ex[0])
            tokid = ex[1]
            print(
                f"\t\t\tID: {tokid} token: {data[sentid][tokid].form}\t{data[sentid][tokid].upos}\t{data[sentid][tokid].feats}")
            headid = data[sentid][tokid].head
            print(
                f"\t\t\tID: {headid} token: {data[sentid][headid].form}\t{data[sentid][headid].upos}\t{data[sentid][headid].feats}")
    print("****************************************")

def printTreeForBinaryFeatures(tree_rules, pos_id2tag, relation_id2tag, used_relations, used_headpos, used_childpos, feature):
    # print(tree_rules)
    new_lines = []
    lines = tree_rules.split("\n")
    relation_stack, head_pos_stack, child_pos_stack = [], [], []
    rel_stack, headpos_stack, childpos_stack = [], [], []
    relation_stack.append((0, used_relations))
    head_pos_stack.append((0, used_headpos))
    child_pos_stack.append((0, used_childpos))  # = used_relations, used_headpos, used_childpos
    relation_num, headpos_num, childpos_num = 1, 1, 1
    prev = None
    for line in lines:
        if "<=" in line:
            info = line.split("<=")

            if "@head" in info[0]:
                depth = info[0].count("|")
                new_line = "   ".join(["|" * depth]) + " {0} not in head".format(feature)
                new_lines.append(new_line)

            elif "@child" in info[0]:
                depth = info[0].count("|")
                new_line = "   ".join(["|" * depth]) + "  {0} not in child".format(feature)
                new_lines.append(new_line)

            elif "relation" in info[0]:
                tags = info[0].split("@")[1].lstrip().rstrip()
                rel = set()
                tags = used_relations - set([tags])
                for t in tags:
                    rel.add(t)
                depth = info[0].count("|")
                (lookup_depth, lookup_all) = relation_stack[-1]
                if depth == lookup_depth:
                    relation_stack.pop()
                (lookup_depth, lookup_all) = relation_stack[-1]

                rel_stack.append((depth, rel & lookup_all))
                relation_stack.append((depth, rel & lookup_all))
                tags = "[" + ",".join(list(rel & lookup_all)) + "]"
                new_line = "   ".join(["|" * depth]) + "--- relation in " + tags
                relation_num += 1
                new_lines.append(new_line)
                prev = "if-relation"

            elif "head" in info[0]:
                tags = info[0].split("@")[1].lstrip().rstrip()
                tags = used_headpos - set([tags])
                pos = set()
                for t in tags:
                    pos.add(t)
                depth = info[0].count("|")
                (lookup_depth, lookup_all) = head_pos_stack[-1]

                if depth == lookup_depth:
                    headpos_stack.pop()
                (lookup_depth, lookup_all) = head_pos_stack[-1]

                headpos_stack.append((depth, pos & lookup_all))
                head_pos_stack.append((depth, pos & lookup_all))
                tags = "[" + ",".join(list(pos & lookup_all)) + "]"
                new_line = "   ".join(["|" * depth]) + "--- head-pos in " + tags
                relation_num += 1
                new_lines.append(new_line)
                prev = "if-head"


            elif "child" in info[0]:
                tags = info[0].split("@")[1].lstrip().rstrip()
                tags = used_childpos - set([tags])
                pos = set()
                for t in tags:
                    pos.add(t)
                depth = info[0].count("|")
                (lookup_depth, lookup_all) = child_pos_stack[-1]

                if depth == lookup_depth:
                    childpos_stack.pop()
                (lookup_depth, lookup_all) = child_pos_stack[-1]
                childpos_stack.append((depth, pos & lookup_all))
                child_pos_stack.append((depth, pos & lookup_all))
                tags = "[" + ",".join(list(pos & lookup_all)) + "]"
                new_line = "   ".join(["|" * depth]) + "--- child-pos in " + tags
                new_lines.append(new_line)
                prev = "if"

            else:
                new_line = line
                new_lines.append(new_line)

        elif ">" in line:
            info = line.split(">")
            depth = info[0].count("|")

            while depth <= head_pos_stack[-1][0]:
                head_pos_stack.pop()

            while depth <= relation_stack[-1][0]:
                relation_stack.pop()

            while depth <= child_pos_stack[-1][0]:
                child_pos_stack.pop()

            if feature in info[0] and "head" in line:
                new_line = "   ".join(["|" * depth]) + "--- else-{0} in head".format(feature)
                new_lines.append(new_line)

            elif feature in info[0] and "child" in line:
                new_line = "   ".join(["|" * depth]) + "--- else-{0} in child".format(feature)
                new_lines.append(new_line)

            elif "relation" in info[0]:
                tags = line.split("@")[1].split(">")[0].lstrip().rstrip()
                (prev_depth, prev_rel) = rel_stack.pop()
                cur_rel = tags
                relation_stack.append((depth, cur_rel))
                new_line = "   ".join(["|" * depth]) + "--- else-relation" + " in " + "[" + tags + "]"
                new_lines.append(new_line)
                prev = "else-relation"


            elif "head" in info[0] or "child" in info[0]:
                tags = line.split("@")[1].split(">")[0].lstrip().rstrip()

                if "head" in info[0]:
                    (prev_depth, prev_pos) = headpos_stack.pop()
                    cur_pos = tags
                    head_pos_stack.append((depth, cur_pos))
                    new_line = "   ".join(["|" * depth]) + "--- else-head-pos" + " in " + "[" + tags + "]"

                elif "child" in info[0]:
                    cur_pos = tags
                    child_pos_stack.append((depth, cur_pos))
                    new_line = "   ".join(["|" * depth]) + "--- else-child-pos" + " in " + "[" + tags + "]"

                new_lines.append(new_line)

            else:
                new_line = line
                new_lines.append(new_line)


        else:
            new_lines.append(line)

    # leaves = constructTree(new_lines)
    return "\n".join(new_lines)

def isAgreement(feature_distribution, agree, disagree, threshold, hard=False):
    leaftotal = agree + disagree
    if agree < disagree:
        return False

    if hard:
        t = agree * 1.0 / leaftotal
        if t > threshold:
            return True
        else:
            return False


    chance_agreement = 0.0
    total = 0.0
    for type,val in feature_distribution.items():
        total += val
    for type, val in feature_distribution.items():
        p = val * 1.0 / total
        chance_agreement += (p * p)
    empirical_distr = [1 - chance_agreement, chance_agreement]


    expected_agree = empirical_distr[1] * leaftotal
    expected_disagree = empirical_distr[0] * leaftotal
    t = agree * 1.0 / leaftotal

    if min(expected_disagree, expected_agree) < 5: #cannot apply the chi-squared test, return chance agreement
        return False

    T,p = chisquare([disagree, agree], [expected_disagree, expected_agree])
    w = np.sqrt(T * 1.0 / leaftotal)
    #print(T,p)

    if p < threshold and w > 0.5 : #reject the null
        return True
    else:
        return False

def constructTree(lines, feature):
    lines = lines.split("\n")
    leaves = []
    tree_depth = {}
    for line in lines:
        if line == "" or line == "\n":
            break
        type = ""
        depth = line.count("|")
        if "class" in line:
            agree = int(line.strip().split("class:")[1])
            data = {"relation": None, "head": None, "child": None, "child_feature":None, "head_feature":None}
            i = depth - 1
            while i > 0:
                if i in tree_depth:
                    (type, tags) = tree_depth[i]
                    if data[type] == None:
                        data[type] = tags
                i -= 1
            leaves.append((agree, data))
        else:

            if feature in line:
                if "not" in line:
                    tag = feature + " not in "
                else:
                    tag = feature + " in "
                if "head" in line:
                    type = "head_feature"
                    tag += "head"
                elif "child" in line:
                    type = "child_feature"
                    tag += "child"
                tree_depth[depth] = (type,tag)

            else:
                info = line.strip().split("[")
                if "relation" in info[0]:
                    type = "relation"
                elif "head" in info[0]:
                    type = "head"
                elif "child" in info[0]:
                    type = "child"

                tree_depth[depth] = (type, "[" + info[1].lstrip().rstrip())

    return leaves

def example_web_print(ex, outp2, data):
    try:
        # print('\t\t',data[ex[0]].text)
        sentid = int(ex[0])
        tokid = ex[1]
        headid = data[sentid][tokid].head
        outp2.write('<pre><code class="language-conllu">\n')
        for token in data[sentid]:
            if token.id == tokid:
                outp2.write(token.conll() + "\n")
            elif token.id == headid:
                temp = token.conll().split('\t')
                temp2 = '\t'.join(temp[:6])
                outp2.write(f"{temp2}\t0\t_\t_\t_\n")
            elif '-' not in token.id:
                outp2.write(f"{token.id}\t{token.form}\t_\t_\t_\t_\t0\t_\t_\t_\n")
        outp2.write('\n</code></pre>\n\n')
    # print(f"\t\t\tID: {tokid} token: {data[sentid][tokid].form}\t{data[sentid][tokid].upos}\t{data[sentid][tokid].feats}")
    # print(data[sentid][tokid].conll())
    # headid = data[sentid][tokid].head
    # print(f"\t\t\tID: {headid} token: {data[sentid][headid].form}\t{data[sentid][headid].upos}\t{data[sentid][headid].feats}")
    # print(data[sentid][headid].conll())
    except:
        pass

def getAggreeingExamples(leaf_node, feature, data, leaf_values, random_samples, M=5):
    found_agree = []
    found_disagree = []
    found_na = []
    relation_dict, head_pos_dict, child_pos_dict = defaultdict(lambda : 0), defaultdict(lambda :0), defaultdict(lambda :0)
    (agree, data_) = leaf_node
    found = False
    #class_disagree, class_agree, class_na = int(leaf_values[0]), int(leaf_values[1]), int(leaf_values[2])
    #class_disagree, class_agree = int(leaf_values[0]), int(leaf_values[1])
    required_relation, required_head, required_child, required_head_feature, required_child_feature = parseLeafInformation(data_)
    example = {}
    example_count = defaultdict(lambda:0)

    for j, sentence in enumerate(data):
        if j not in random_samples:
            continue
        for token in sentence:
            token_id = token.id
            if "-" in token_id:
                continue
            if token.deprel is None:
                continue
            relation = token.deprel.lower()
            pos = token.upos
            feats = token.feats

            if token.head and token.head != "0":
                head_pos = sentence[token.head].upos
                head_feats = sentence[token.head].feats
                shared, agreed = find_agreement(feats, head_feats)
                yes_child, yes_head, yes_rel, yes_child_feature, yes_head_feature = assert_path(feature, head_pos, pos, relation, feats, head_feats, required_child,
                                                           required_head,
                                                           required_relation, required_head_feature, required_child_feature)

                if feature in shared:
                    if yes_rel and yes_child and yes_head and yes_child_feature and yes_head_feature:
                        relation_dict[relation] += 1
                        head_pos_dict[head_pos] += 1
                        child_pos_dict[pos] +=1
                        if feats[feature] == head_feats[feature]:
                            if required_relation is None:
                                relation = None
                            if required_head is None:
                                head_pos = None
                            if required_child is None:
                                pos = None
                            example[(relation, head_pos, pos)] = (j, token_id)
                            example_count[(relation, head_pos, pos)] += 1
                            if len(found_agree) < M:
                                found_agree.append((j, token_id))
                        else:
                            if len(found_disagree) < M:
                                found_disagree.append((j, token_id))

    #sort example by frequence
    sorted_examplecount = sorted(example_count.items(), key=lambda kv:kv[1], reverse=True)
    return found_agree[:M], found_disagree[:M], relation_dict, head_pos_dict, child_pos_dict, example, sorted_examplecount

def parseLeafInformation(data_):
    required_relation = data_["relation"].replace("[", "").replace("]", "").split(",") if data_[
                                                                                              'relation'] is not None  and data_['relation'] != '' else None
    required_head = data_["head"].replace("[", "").replace("]", "").split(",") if data_[
                                                                                      "head"] is not None and data_["head"] != '' else None
    required_child = data_["child"].replace("[", "").replace("]", "").split(",") if data_[
                                                                                        "child"] is not None and data_['child'] != '' else None
    # if data_["head_feature"] is not None:
    #     if "not" in data_["head_feature"]:
    #         required_head_feature = False
    #     else:
    #         required_head_feature = True
    # else:
    #     required_head_feature = None
    #
    # if data_["child_feature"] is not None:
    #     if "not" in data_["child_feature"]:
    #         required_child_feature = False
    #     else:
    #         required_child_feature = True
    # else:
    #     required_child_feature = None

    return required_relation, required_head, required_child, None, None

def assert_path(feature, head_pos, pos, relation, feats, head_feats, required_child, required_head, required_relation, required_head_feature, required_child_feature):
    yes_child, yes_head,yes_rel, yes_head_feature, yes_child_feature = False, False, False, False, False

    if required_child_feature is not None:
        if feature in feats:
            if required_child_feature:
                yes_child_feature = True
    else:
        yes_child_feature = True

    if required_head_feature is not None:
        if feature in head_feats:
            if required_head_feature:
                yes_head_feature = True
    else:
        yes_head_feature = True

    if required_relation is not None:
        if relation in required_relation:
            yes_rel = True
    else:
        yes_rel = True
    if required_head is not None:
        if head_pos in required_head:
            yes_head = True
    else:
        yes_head = True
    if required_child is not None:
        if pos in required_child:
            yes_child = True
    else:
        yes_child = True
    return yes_child, yes_head, yes_rel, yes_child_feature, yes_head_feature

def printDataStat(train, dev, test, features):
    print("Percent of examples agreeing\n")
    for feature in features:
        if feature in train:
            print("Training Data: {0} {1} Agree: {2} ".format(len(train[feature]), feature,
                                                              np.sum(train[feature] == 1) / len(train[feature])))
        if feature in dev:
            print("Validation Data: {0} {1} Agree: {2} ".format(len(dev[feature]), feature,
                                                                np.sum(dev[feature] == 1) / len(dev[feature])))
        if feature in test:
            print("Testing Data: {0} {1} Agree: {2} ".format(len(test[feature]), feature,
                                                             np.sum(test[feature] == 1) / len(test[feature])))
        print()

def printMultipleLines(input, t=5):
    input = list(input)
    new_list = []
    if len(input) > t:
        num = len(input) % t
        i = 0
        j = i + t
        while i < len(input):
            new_list.append(",".join(str(a) for a in input[i:j]))
            i = j
            j = i + t

        return "\\n".join(new_list) + "\\n"
    else:
        new_list.append(",".join(str(a) for a in input))
        return ",".join(str(a) for a in input) + "\\l"

def debug(feature, data):
    found_na = []
    with codecs.open(feature + "_debug.txt", "w", encoding='utf-8') as fout:
        for j, sentence in enumerate(data):
            for token in sentence:
                token_id = token.id
                relation = token.deprel.lower()
                pos = token.upos
                feats = token.feats
                if token.head and token.head != "0":
                    head_pos = sentence[token.head].upos
                    head_feats = sentence[token.head].feats
                    shared, agreed = find_agreement(feats, head_feats)
                    if feature not in shared:
                        found_na.append((j, token_id))

        total = 0
        nongendered_noun_cpos =0
        nongendered_noun_hpos = 0
        for (sent_id, token_id) in found_na:
            sentid = int(sent_id)
            total += 1
            tokid = token_id
            headid = data[sentid][tokid].head
            sentence = []
            cpos, hpos = "", ""
            cpos_feats = ""
            hpos_feats = ""
            for token in data[sentid]:
                sentence.append(token.form)

                if token.id == tokid:
                    cpos_feats = ""
                    cpos = token.form  + "-" + token.upos
                    for key, valye in token.feats.items():
                        cpos_feats += key + "=" + str(list(valye)) + ","

                elif token.id == headid:
                    hpos_feats = ""
                    hpos = token.form + "-" + token.upos
                    for key, valye in token.feats.items():
                        hpos_feats += key + "=" + str(list(valye)) + ","

            if'NOUN' in cpos:
                if "Gender" not in cpos_feats:
                    nongendered_noun_cpos +=1
            if "NOUN" in hpos:
                if "Gender" not in hpos_feats:
                    nongendered_noun_hpos += 1

            fout.write(" ".join(sentence) + "\n")
            fout.write(cpos + "\t" + cpos_feats + "\n")
            fout.write(hpos + "\t" + hpos_feats + "\n")
            fout.write("\n")

    print("cpos: {0}".format(nongendered_noun_cpos * 100.0/ total))
    print("hpos: {0}".format(nongendered_noun_hpos * 100.0 / total))

def colorRetrival(agree, disagree, feature_distribution, threshold, hard):
    agreement_color_schemes = {0.1: '#eff3ff', 0.5: '#bdd7e7', 0.9: '#2171b5'}
    chanceagreement_color_schemes = {0.1: '#fee8c8', 0.5: '#fdbb84', 0.9: '#e34a33'}
    t = agree * 1.0 / (disagree + agree)
    if isAgreement(feature_distribution, agree, disagree, threshold, hard):#t >= threshold:
        if t >= 0.9:
            color = agreement_color_schemes[0.9]
        elif t >= 0.5:
            color = agreement_color_schemes[0.5]
        else:
            color = agreement_color_schemes[0.1]
    else:
        if (1 - t) >= 0.9:
            color = chanceagreement_color_schemes[0.9]
        elif (1 - t) >= 0.5:
            color = chanceagreement_color_schemes[0.5]
        else:
            color = chanceagreement_color_schemes[0.1]
    return color

def collateTree(feature_distribution, leafedges, editedgraph, topleafnodes, tree_dictionary, treeleaves, threshold, topnodes, removednodes, hard):
    collatedGraph, collate_leafedges, leaves, relabeled_leaves = [], deepcopy(leafedges), deepcopy(treeleaves), {}
    editedgraph = editedgraph.split("\n")
    collatedGraph.append(editedgraph[0])
    collatedGraph.append(editedgraph[1])
    collatedGraph.append(editedgraph[2])
    lableindexmap = {}
    i=0
    removed_leaves = set()

    if len(tree_dictionary) == 1 and len(treeleaves) == 1:  #There are no root nodes, possibly one one leaf
        for leaf, value in tree_dictionary.items():
            disagree = int(tree_dictionary[leaf]["lvalue"][0])
            agree = int(tree_dictionary[leaf]["lvalue"][1])
            t = agree * 1.0 / (agree + disagree)
            class_ = 'chance-agreement'
            if isAgreement(feature_distribution, agree, disagree, threshold, hard):  # t >= threshold:
                class_ = "agreement"
            leaves = [(class_, treeleaves[0][1])]
            relabeled_leaves[0] = (treeleaves[0][1], agree, disagree)
            return editedgraph, leaves, relabeled_leaves

    classm_leaf = {}
    while True:
        #print(i)
        i += 1
        num_changes = 0
        revisedtopleafnodes = defaultdict(set)

        for topnode in topleafnodes: #Traversing only leaves
            children = topleafnodes[topnode]
            classes = {"agreement": [], "chance-agreement": []}
            for child in children:
                class_ = tree_dictionary[child]["info"].split("class = ")[-1].split("\l")[0]
                classes[class_].append(child)
            for class_, children in classes.items():
                if len(children) > 1:  # Merge the children
                    num_changes += 1

                    labels, nsamples, agree, disagree, index = [], 0, 0, 0, []
                    new_leaf_features = {'relation': set(), 'head': set(), 'child': set()}
                    edge = defaultdict(set)

                    for child in children:
                        labels.append(tree_dictionary[child]["label"])
                        nsamples += tree_dictionary[child]["nsamples"]
                        disagree += int(tree_dictionary[child]["lvalue"][0])
                        agree += int(tree_dictionary[child]["lvalue"][1])
                        index.append(tree_dictionary[child]["index"])
                        e = leafedges[child].split(" in ")
                        type, val = e[0], e[1].replace("\\l", "").split(",")
                        for v in val:
                            edge[type].add(v)
                        (_, data) = leaves[tree_dictionary[child]["label"]]
                        relation = set(data['relation'].replace("[","").replace("]","").split(",")) if data['relation'] is not None else set()
                        head = set(data['head'].replace("[","").replace("]","").split(",")) if data['head'] is not None else set()
                        child = set(data['child'].replace("[","").replace("]","").split(",")) if data['child'] is not None else set()
                        new_leaf_features['relation'] = new_leaf_features['relation'].union(relation)
                        new_leaf_features['head'] = new_leaf_features['head'].union(head)
                        new_leaf_features['child'] = new_leaf_features['child'].union(child)

                    new_leaf_features['relation'] = ",".join(new_leaf_features['relation']) if new_leaf_features['relation'] is not None else None
                    new_leaf_features['head'] = ",".join(new_leaf_features['head']) if new_leaf_features['head'] is not None else None
                    new_leaf_features['child'] = ",".join(new_leaf_features['child']) if new_leaf_features['child'] is not None else None
                    edgeinfo = ""
                    for type, val in edge.items():
                        edgeinfo += type + " in " + printMultipleLines(val) + "\\l"


                    leaf_labels, leaf_indices = zip(*sorted(zip(labels, index)))

                    #update the tree_dictionary
                    leaf_label, leaf_index = leaf_labels[0], leaf_indices[0]
                    tree_dictionary[leaf_index]["label"] = leaf_label
                    tree_dictionary[leaf_index]["nsamples"] = nsamples
                    tree_dictionary[leaf_index]["lvalue"] = [disagree, agree]
                    tree_dictionary[leaf_index]["index"] = leaf_index
                    leafedges[leaf_index] = edgeinfo
                    leaves[leaf_label] = ("", new_leaf_features)
                    lableindexmap[leaf_label] = leaf_index
                    classm_leaf[leaf_label] = class_


                    for llab, llind in zip(leaf_labels[1:], leaf_indices[1:]):
                        tree_dictionary[topnode]['children'].remove(llind)
                        removednodes.add(llind)
                        removed_leaves.add(llab)

                    if 'top' in tree_dictionary[topnode] and len(tree_dictionary[topnode]['children']) == 1:
                        child = tree_dictionary[topnode]["children"][0]
                        toptopnode = tree_dictionary[topnode]['top']
                        tree_dictionary[toptopnode]['children'].append(child)
                        leafedges[leaf_index] = leafedges[topnode]
                        del tree_dictionary[topnode]
                        tree_dictionary[toptopnode]['children'].remove(topnode)
                        removednodes.add(topnode)
                        revisedtopleafnodes[toptopnode].add(child)

                    else:
                        revisedtopleafnodes[topnode].add(leaf_index)



                elif len(children) == 1:
                    revisedtopleafnodes[topnode].add(children[0])
                    lableindexmap[tree_dictionary[children[0]]["label"]] = children[0]
                    classm_leaf[tree_dictionary[children[0]]["label"]] = class_
                    #collatedGraph.append(tree_dictionary[list(children)[0]]["info"])

        topleafnodes = revisedtopleafnodes
        if num_changes == 0 or i > 5:
            break

    # for node in topnodes:
    #     #Check if the subtree
    labeltext = "[headlabel=\"{0}\",labelfontsize=10];"
    topnodes = set(topnodes) - removednodes
    for node in topnodes:
        #if node not in removednodes:
        collatedGraph.append(tree_dictionary[node]["info"])
        for num, children in enumerate(tree_dictionary[node]["children"]):
            textinfo = str(node) + " -> " + str(children)
            angle = 50
            if num > len(tree_dictionary[node]["children"]) / 2:
                angle *= -1
            edge = leafedges[children]
            textinfo += " " + labeltext.format(edge.replace("\\l", ""))
            collatedGraph.append(textinfo)

    new_leaf_info = "{0} [label=\"Leaf- {1}\\n{2}\\lsamples={3}\\lvalue = {4}\\lclass = {5}\\l\", fillcolor=\"{6}\"]; "
    new_leaves, new_leaf_num = [], 0
    for leaf,(_,leafnode) in enumerate(leaves):
        if leaf in removed_leaves:
            continue

        data_info = ""
        if leafnode["relation"] is None or len(leafnode["relation"]) == 0:
            data_info += "relation = *" + "\\l\\l"
        else:
            data_info += "relation = " + printMultipleLines(leafnode["relation"].split(",")) + "\\l"

        if leafnode["head"] is None or len(leafnode["head"]) == 0:
            data_info += "head-pos = *" + "\\l\\l"
        else:
            data_info += "head-pos = " + printMultipleLines(leafnode["head"].split(",")) + "\\l"

        if leafnode["child"] is None or len(leafnode["child"]) == 0:
            data_info += "child-pos = *" + "\\l\\l"
        else:
            data_info += "child-pos = " + printMultipleLines(leafnode["child"].split(",")) + "\\l"

        index = lableindexmap[leaf]
        classm = classm_leaf[leaf]

        disagree,agree = int(tree_dictionary[index]['lvalue'][0]), int(tree_dictionary[index]['lvalue'][1])
        t = agree * 1.0 / (agree + disagree)
        relabeled_leaves[new_leaf_num] = (leafnode, agree, disagree)
        class_="chance-agreement"
        if isAgreement(feature_distribution, agree, disagree, threshold, hard):#t >= threshold:
            class_ ="agreement"
        #assert classm == class_ #Making sure that after mergign the leaves the threshold gives the same class value.
        new_leaves.append((class_, leafnode))
        lvalue = "[" + str(disagree) + "," + str(agree) + "]"
        leafinfo = new_leaf_info.format(tree_dictionary[index]['index'], new_leaf_num, data_info.lower(), tree_dictionary[index]['nsamples'], lvalue,
                                        class_, colorRetrival(agree, disagree, feature_distribution, threshold, hard))
        collatedGraph.append(leafinfo)
        new_leaf_num += 1

    collatedGraph.append(editedgraph[-1])
    return collatedGraph, new_leaves, relabeled_leaves

def pruneTree(editedgraph, tree_dictionary, topnodes, leafnodes, leafedges, feature):
    prunedGraph = []
    removednodes = set()
    topleafnodes = defaultdict(set)
    i = 0
    while True:
        #print("Pruning iteration: ", i)
        i +=1
        num_changes = 0
        for node in topnodes:
            removed = False
            if node not in removednodes:
                current = tree_dictionary[node]
                current_edge = current["edge"].split("in")[0].lstrip().rstrip()
                for child in current["children"]:
                    if child not in leafnodes:
                        child_node = tree_dictionary[child]
                        child_edge = child_node["edge"].split("in")[0].lstrip().rstrip()
                        if current_edge == child_edge: #current edge is same as the child edge (so all child edges are same)
                            if feature in current_edge:
                                x = current["edge"].split("in")[1].lstrip().rstrip()
                                y = child_node["edge"].split("in")[1].lstrip().rstrip()
                                if x != y:
                                    continue

                            removednodes.add(child)
                            for n in child_node["children"]:
                                tree_dictionary[node]["children"].append(n)
                            num_changes += 1
                            tree_dictionary[node]["children"].remove(child)
                            break
                    else:
                        topleafnodes[node].add(child)

        if num_changes == 0 or i > 5:
            break

    prunedGraph.append(editedgraph[0])
    prunedGraph.append(editedgraph[1])
    prunedGraph.append(editedgraph[2])

    for node in topnodes:
        if node not in removednodes:
            prunedGraph.append(tree_dictionary[node]["info"])
            info = tree_dictionary[node]["info"]
            tree_dictionary[node]["label"] = int(info.split("node - ")[1].split("\\n")[0].lstrip().rstrip())
            tree_dictionary[node]["nsamples"] = int(info.split("nsamples = ")[1].split("\\l")[0].lstrip().rstrip())
            tree_dictionary[node]["lvalue"] = info.split("value = [")[1].split("\\l")[0].replace("]", "").split(",")
            tree_dictionary[node]["index"] = int(info.split("[")[0])


    for node in leafnodes:
        prunedGraph.append(tree_dictionary[node]["info"])
        info = tree_dictionary[node]["info"]
        tree_dictionary[node]["label"]  = int(info.split("Leaf- ")[1].split("\\n")[0].lstrip().rstrip())
        tree_dictionary[node]["nsamples"]  = int(info.split("samples = ")[1].split("\\l")[0].lstrip().rstrip())
        tree_dictionary[node]["lvalue"]  = info.split("value = [")[1].split("\\l")[0].replace("]","").split(",")
        tree_dictionary[node]["index"] = int(info.split("[")[0])

    #labeltext = "[labeldistance={0},labelangle={1}, headlabel=\"{2}\",labelfontsize=10];"
    labeltext = "[headlabel=\"{0}\",labelfontsize=10];"

    for node in topnodes:
        if node not in removednodes:
            for num, children in enumerate(tree_dictionary[node]["children"]):
                textinfo =   str(node) + " -> " + str(children)
                tree_dictionary[children]["top"] = node
                angle = 50
                if num > len(tree_dictionary[node]["children"]) / 2:
                    angle *= -1
                edge = leafedges[children]
                textinfo += " " + labeltext.format(edge.replace("\\l",""))
                prunedGraph.append(textinfo)
    prunedGraph.append(editedgraph[-1])
    return "\n".join(prunedGraph), tree_dictionary, leafnodes, topleafnodes, removednodes, topnodes

def get_vocab_from_set(input):
    word_to_id = {}
    if "NA" in input:
        word_to_id['NA'] = 0
    for i in input:
        if i == "NA":
            continue
        word_to_id[i] = len(word_to_id)
    id_to_word = {v:k for k,v in word_to_id.items()}
    return word_to_id, id_to_word

def plot_histogram(input, color, type, file):
    #sns.set()
    r = [i for i in range(len(input))]
    x,y = [],[]
    for k,v in input.items():
        x.append(k)
        y.append(v)
    y,x = zip(*sorted(zip(y,x), reverse=True))
    plt.bar(r, y, color=color, edgecolor='white', width=1)
    plt.xticks(r, x, rotation=45, fontsize=9)
    plt.ylabel("count")
    plt.title(type)
    plt.legend(handles=[mpatches.Patch(color=color, label=type)])
    plt.savefig(file + type + ".png", transparent=True)
    plt.close()

def distributional_metric(leaves, test_path, feature, feature_distribution,  test_samples, threshold, hard):
    f = test_path.strip()
    data = pyconll.load_from_file(f"{f}")
    testleaves = {}
    wd, d = 0.0, 0.0
    total = 0.0
    #Calculating the chance distribution for the feature values
    for type,val in feature_distribution.items():
        total += val
    chance_agreement = 0.0
    for type, val in feature_distribution.items():
        p = val * 1.0/total
        chance_agreement += (p * p)
    chance_distribution = [1-chance_agreement, chance_agreement]


    agreement_distribution = [0.0, 1.0]
    only_agreement, only_agreement_count, found_agreement = 0, 0, 0
    atleastagree = False

    leaves_used = 0
    test_leaves_distr = {}
    #ITerating the leaves constructed from the training data
    for leaf_num, leafinfo in leaves.items():
        (info, agree, disagree) = leafinfo
        t = agree * 1.0 / (agree + disagree)
        gold_distribution = chance_distribution
        agreement = False
        if isAgreement(feature_distribution, agree, disagree, threshold, hard):# t >= threshold: #threshold for definite agreeement
            gold_distribution = agreement_distribution
            only_agreement_count += 1
            agreement = True


        required_relation, required_head, required_child, required_head_feature, required_child_feature = parseLeafInformation(info)
        testagree, testdisagree = 0,0
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

                if token.head and token.head != "0":
                    head_pos = sentence[token.head].upos
                    head_feats = sentence[token.head].feats
                    shared, agreed = find_agreement(feats, head_feats)
                    yes_child, yes_head, yes_rel, yes_child_feature, yes_head_feature = assert_path(feature, head_pos,
                                                                                                    pos, relation,
                                                                                                    feats, head_feats,
                                                                                                    required_child,
                                                                                                    required_head,
                                                                                                    required_relation,
                                                                                                    required_head_feature,
                                                                                                    required_child_feature)

                    if feature in shared:
                        if yes_rel and yes_child and yes_head and yes_child_feature and yes_head_feature:
                            if feats[feature] == head_feats[feature]:
                                testagree += 1
                            else:
                                testdisagree += 1

        test_leaves_distr[leaf_num] = (testdisagree, testagree)
        testleaves[leaf_num] = (info, testagree, testdisagree)
        if testagree == 0 and testdisagree == 0: #No tuples with those rules found in the test data:
            continue
        leaves_used += 1

        test_distribution = [testdisagree * 1.0 /(testagree + testdisagree), testagree * 1.0/ (testagree + testdisagree) ]
        if agreement:
            only_agreement += wasserstein_distance(gold_distribution, test_distribution)
            found_agreement += 1
        d += wasserstein_distance(gold_distribution, test_distribution)
        wd += wasserstein_distance(gold_distribution, test_distribution) * ((testagree + testdisagree) / test_samples)

    if leaves_used == 0: #None of the conditions were present in the test data
         return 0.0, 0.0, 0.0
    WDS, UDS = np.round(1 - (wd * 1.0/ leaves_used), 3), np.round(1-  (d * 1.0 / leaves_used), 3)
    if only_agreement_count == 0: # There are no agreeing leaves, NA
        ADS = 'NA'
    elif found_agreement == 0: #None of the examples were in the agreeing
        ADS = 0.0
    else:
        ADS = np.round(1 - (only_agreement * 1.0 / only_agreement_count), 3)

    return WDS, UDS, ADS, test_leaves_distr

def retrivePossibleTuples(feature, leafdata, traindata):
    total = 0.0
    freq_tuple = defaultdict(lambda: 0)
    tuples = set()
    required_relation, required_head, required_child, required_head_feature, required_child_feature = parseLeafInformation(leafdata)
    for j, sentence in enumerate(traindata):
        for token in sentence:
            token_id = token.id
            relation = token.deprel
            pos = token.upos
            feats = token.feats
            if token.head and token.head != "0":
                head_pos = sentence[token.head].upos
                head_feats = sentence[token.head].feats
                shared, agreed = find_agreement(feats, head_feats)
                if feature in shared:
                    pos_feature = ",".join(list(feats[feature]))
                    head_feature = ",".join(list(head_feats[feature]))
                    total += 1
                    freq_tuple[(relation, head_pos, pos)] += 1

                    yes_child, yes_head, yes_rel, yes_child_feature, yes_head_feature = assert_path(feature,
                                                                                                          head_pos, pos,
                                                                                                          relation,
                                                                                                          feats,
                                                                                                          head_feats,
                                                                                                          required_child,
                                                                                                          required_head,
                                                                                                          required_relation,
                                                                                                          required_head_feature,
                                                                                                          required_child_feature)
                    if yes_rel and yes_child and yes_head:
                        tuples.add((relation, head_pos, pos))

    return tuples, total, freq_tuple

def getTestData(data, feature):
    # print("Retrieving Test data")

    freq_tuple = defaultdict(lambda: 0)
    tuple_agree_information = defaultdict(lambda : 0)

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

            if token.head and token.head != "0":
                head_pos = sentence[token.head].upos
                head_feats = sentence[token.head].feats
                shared, agreed = find_agreement(feats, head_feats)
                if feature in shared:
                    tuple = (relation, head_pos, pos)
                    freq_tuple[(relation, head_pos, pos)] += 1
                    if feats[feature] == head_feats[feature]:
                        tuple_agree_information[tuple] += 1

    return tuple_agree_information, freq_tuple

def automated_metric(leaves, test_path, feature, feature_distribution,  test_samples, threshold, hard, traindata):
    f = test_path.strip()
    data = pyconll.load_from_file(f"{f}")
    test_agree_tuple, test_freq_tuple = getTestData(data, feature)
    automated_evaluation_score = {}
    testleaves = {}
    total = 0.0
    test_leaves_distr = {}
    #ITerating the leaves constructed from the training data


    for leaf_num, leafinfo in leaves.items():
        class_ = 'chance-agreement'
        test_leaf_agreement, test_leaf_disagree = 0, 0
        (info, agree, disagree) = leafinfo
        if isAgreement(feature_distribution, agree, disagree, threshold, hard):# t >= threshold: #threshold for definite agreeement
            class_ = 'agreement'
        tuples, train_total, freq_tuple = retrivePossibleTuples(feature, info, traindata)
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

        test_leaves_distr[leaf_num] = (test_leaf_disagree, test_leaf_agreement)
        testleaves[leaf_num] = (info, test_leaf_agreement, test_leaf_disagree)

    total = 0
    correct = 0
    for tuple, count in automated_evaluation_score.items():
        correct += count
        total += 1
    metric = correct * 1.0/total
    return metric, test_leaves_distr

def wasserstein_distance(p1, p2):
    p1, p2 = np.array(p1), np.array(p2)
    d = np.mean(np.abs(p1-p2))
    return d

def convertProb(distr):
    new_distr = []
    distr = np.array(distr)
    total = np.sum(distr)
    for x in distr:
        new_distr.append(x * 1.0/total)
    return new_distr