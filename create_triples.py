import pickle, json, os, pyconll, sys
import utils, dataloader
import argparse
import numpy as np
np.random.seed(1)
import random
random.seed(1)
from collections import defaultdict
from django import forms

def getUnique(examples, data):
	covered = set()
	unique_examples = []
	for (sent_id, token_num) in examples:
		token_lemma = data[sent_id][token_num].lemma
		head =  data[sent_id][token_num].head
		head_lemma =  data[sent_id][head].lemma
		if token_lemma and head_lemma:
			token_lemma = token_lemma.lower()
			head_lemma = head_lemma.lower()
			if (token_lemma, head_lemma) not in covered:
				unique_examples.append((sent_id, token_num))
				covered.add((token_lemma, head_lemma))
	return unique_examples
def FrequretrivePossibleTuples(feature):
    total = 0.0
    freq_tuple = defaultdict(lambda:0)
    tuples = set()

    for j, sentence in enumerate(data):
        for token in sentence:
            token_id = token.id
            relation = token.deprel
            pos = token.upos
            feats = token.feats

            if token.head and token.head != "0":
                head_pos = sentence[token.head].upos
                head_feats = sentence[token.head].feats
                shared, agreed = utils.find_agreement(feats,head_feats)
                if feature in shared:
                    total += 1

                    freq_tuple[(relation, head_pos, pos)] += 1


    return tuples, total, freq_tuple

def retrivePossibleTuples(feature, tuple):
	agree_examples, non_agree_examples =[],[]

	for j, sentence in enumerate(data):
		for token in sentence:
			token_id = token.id
			relation = token.deprel
			pos = token.upos
			feats = token.feats
			if token.head and token.head != "0":
				head_pos = sentence[token.head].upos
				head_feats = sentence[token.head].feats
				shared, agreed = utils.find_agreement(feats,head_feats)
				if feature in shared:
					new_tuples = (relation, head_pos, pos)
					if new_tuples == tuple:
						if feature in agreed:
							agree_examples.append((j, token_id))
						else:
							non_agree_examples.append((j, token_id))


	return agree_examples, non_agree_examples

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--file", type=str, default="./decision_tree_files.txt")
	parser.add_argument("--features", type=str, default="Gender+Person+Number+Mood+Tense+Case", nargs='+')
	parser.add_argument("--percent", type=float, default=1.0)
	parser.add_argument("--relation_map", type=str, default="./relation_map")
	parser.add_argument("--seed", type=int, default=1)

	parser.add_argument("--simulate", action="store_true", default=False)
	#parser.add_argument("--rule_dir", type=str, default='/Users/aditichaudhary/Documents/CMU/agreement/sud_simulate_soft_outputs/')



	args = parser.parse_args()
	folder_name = f"./annotation_site/templates/"
	with open(f"{folder_name}/header.html") as inp:
		HEADER = inp.readlines()
	HEADER = ''.join(HEADER)

	with open(f"{folder_name}/footer.html") as inp:
		FOOTER = inp.readlines()
	FOOTER = ''.join(FOOTER)

	with open(args.file, "r") as inp:
		files = []
		for file in inp.readlines():
			if file.startswith("#"):
				continue
			files.append(file)

	with open(f"{folder_name}/index.html", 'w') as op:
		op.write(HEADER + '\n')
		op.write(f"<h1>Rules for Morphological Agreement</h1>")
		op.write("<p> We parsed the <a href=\"https://universaldependencies.org/\">Universal Dependencies</a> data in order to extract rules in many languages.</p>\n")
		op.write("<p>Here are the languages (and treebanks) we currently support:</p><br><ul>")

	fnum = 0
	args.features = args.features.split("+")
	while fnum < len(files):

		treebank = files[fnum].strip()
		if treebank.startswith("#"):
			continue
		fnum += 1

		train_path, dev_path, test_path = None, None, None
		for [path, dir, inputfiles] in os.walk(treebank):
			for file in inputfiles:
				if "-train.conllu" in file:
					train_path = treebank + "/" + file
					if args.simulate:
					# if "-simuall" in file:
					# 	train_path = treebank + "/" + file
						percent = treebank.split("-")[-2] + "-" + treebank.split("-")[-1]
						lang = train_path.strip().split('/')[-1].split("_")[0]
						lang += "-" + percent
					else:
						percent = 'all'
						lang = train_path.strip().split('/')[-1].split("_")[0]

		lang_full = lang
		f = train_path.strip()
		print("Processing treebank ", treebank, train_path)
		with open(f"{folder_name}/index.html", 'a') as op:
			op.write(f"<li> <a href=\"/{lang}/\">{lang_full}</a></li>\n")

		try:
			os.mkdir(f"{folder_name}/{lang}")
		except OSError:
			#print ("Creation of the directory failed")
			i =0

		with open(f"{folder_name}/{lang}/index.html", 'w') as outp:
			outp.write(HEADER+"\n" + f"<h1> {lang_full} </h1> <br>\n")
			outp.write(f"<strong>{lang_full}</strong> exhibits the following agreement:<br><ul>")

			data = pyconll.load_from_file(f"{f}")
			for feature in args.features:
				# try:
				# 	os.mkdir(f"{folder_name}/{lang}/{feature}")
				# except OSError:
				# 	# print ("Creation of the directory failed")
				# 	i = 0
				outp.write(f"<li>{feature} rules:. <a href=\"{feature.lower()}/\">Examples</a></li>\n")


				filename=f'{folder_name}/{lang}/{feature}.html'
				with open(filename,'w') as o:
					o.write(HEADER + "\n")
					o.write(f'<h1>Rules for {feature} agreement: </h1>')
					o.write(f'<form action="http://triton.lti.cs.cmu.edu:5000/{lang}/{feature.lower()}" method="POST">')

					with open(f'{folder_name}/{lang}/{lang}_{feature}_to_annotate.txt','w') as fin:
						fin.write(f'{feature}: Rules in the order of relation,head-pos,child-pos: Label each rule with AGREE, SOMETIMES-AGREE, NA\n')
						tuples, total, freq_tuple = FrequretrivePossibleTuples(feature)
						sorted_freq_tuples = sorted(freq_tuple.items(), key=lambda kv: kv[1], reverse=True)[:20]
						rule_num =0
						for (tuple,_)	 in sorted_freq_tuples:
							agree_examples, non_agree_examples = retrivePossibleTuples(feature,tuple)
							(required_relation, required_head, required_child) = tuple
							fin.write(f'Relation = {required_relation}; Head-POS = {required_head}; Child-POS = {required_child}\n')

							o.write(f'<p> relation={required_relation}, head={required_head}, dependent={required_child} &nbsp; &nbsp;<br>'
									f'<input type="radio" name="{feature}{rule_num}" value="1">Almost Always Agree</input> &nbsp;'
									f'<input type="radio" name="{feature}{rule_num}" value="2">Sometimes Agree</input> &nbsp;'
									f'<input type="radio" name="{feature}{rule_num}" value="0">Need Not Agree</input> &nbsp;<br>'
									f'<a href=\'javascript:toggle("{feature}{rule_num}")\'>[Agree Examples]</a>'
									f'<div class=\'bibtex\'  id=\'{feature}{rule_num}\'>')#'

							agree_examples = getUnique(agree_examples, data)
							random.shuffle(agree_examples)
							for ex in agree_examples[:5]:
								utils.example_web_print(ex, o, data)
							o.write("</div>")

							o.write(f'<a href=\'javascript:toggle("{feature}{rule_num}-nonagree")\'>[Non-Agree Examples]</a>')
							o.write(f'<div class=\'bibtex\'  id=\'{feature}{rule_num}-nonagree\'>')
							non_agree_examples = getUnique(non_agree_examples, data)
							random.shuffle(non_agree_examples)
							for ex in non_agree_examples[:5	]:
								utils.example_web_print(ex, o, data)
							o.write("</div>")

							o.write(f'<br><div><textarea name="{feature}{rule_num}-comment" id="{feature}{rule_num}-comment" style="font-family:sans-serif;font-size:1.2em;"></textarea></div>')
							rule_num+=1


					o.write(f'</div><button type="submit" type="button" onclick="displayRadioValue()"> Submit </button> </form> ')
					o.write(FOOTER+"\n")


	with open(f"{folder_name}/{lang}/index.html", 'a') as outp:
			outp.write("</ul><br><br><br>\n" + FOOTER+"\n")

	with open(f"{folder_name}/index.html", 'a') as op:
		op.write("</ul><br><br><br>\n" + FOOTER+"\n")
