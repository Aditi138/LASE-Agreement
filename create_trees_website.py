import pickle, json, os, pyconll, sys
import utils, dataloader
import argparse
import numpy as np
np.random.seed(1)
import sklearn
from sklearn.externals.six import StringIO
from sklearn.model_selection import GridSearchCV
from sklearn.tree.export import export_text
import pydotplus
from copy import deepcopy
from collections import defaultdict

def printTreeWithExamplesPDF(model, treerules, treelines, leaves, feature, leafcount, feature_names, test_samples, train_samples, dev_samples):
	dot_data = StringIO()
	sklearn.tree.export_graphviz(model.best_estimator_, out_file=dot_data,
								 feature_names=feature_names, node_ids=True
								 , class_names=["disagree", "agree"], proportion=False, rounded=True, filled=True,
								 leaves_parallel=False, impurity=False)

	nodes = []
	else_nodes = []
	#REtrieve the egde information in nodes, and else-nodes
	for tree_rule, treeline in zip(treerules.split("\n"), treelines.split("\n")):
		header = ""
		if feature in tree_rule:
			header = feature
			if "head" in tree_rule:
				if "<=" in tree_rule:
					nodes.append(header + " not in [head]")
				else:
					else_nodes.append(header + " in [head]")
			elif "child" in tree_rule:
				if "<=" in tree_rule:
					nodes.append(header + " not in [child]")
				else:
					else_nodes.append(header + " in [child]")
		else:
			if "relation" in tree_rule:
				header  = "relation"
			elif "head" in tree_rule:
				header="head-pos"
			elif "child" in tree_rule:
				header="child-pos"

			info = "[" + treeline.split("[")[-1]

			if "<=" in tree_rule:
				nodes.append(header + " in " + info.lstrip().rstrip().lower())
			if "else" in treeline:
				else_nodes.append(header + " in " +info.lstrip().rstrip().lower())

	try:
		os.mkdir(f"{folder_name}/{lang}/{feature}")
	except OSError:
		#print(f"Directory websiter/{lang}/{feature} already exists")
		i = 0

	#Traverse the tree to add the information in required format
	pos_set = printPOSInfomation(feature)

	filename = f"{folder_name}/{lang}/{feature}/{feature}.html"
	if args.hard:
		threshold = 0.9
	else:
		threshold = 0.01
	with open(filename, 'w') as outp:
		HEADER = ORIG_HEADER.replace("main.css", "../../main.css")
		outp.write(HEADER + '\n')
		outp.write(f'<ul class="nav"><li class="nav"><a class="active" href=\"../../index.html\">Home</a>'
				 f'</li><li class="nav"><a href=\"../../introduction.html\">Usage</a></li>'
				 f'<li class="nav"><a href="../../about.html\">About Us</a></li></ul>')
		outp.write(f"<br><li><a href=\"../index.html\">Back to {language_fullname} page</a></li>\n")
		outp.write(f"<h1> Token Distribution across {feature} </h1>")
		outp.write(f"<p>The following histogram captures the token distribution per different part-of-speech (POS) tags.</p>")
		outp.write(f"<p>Legend on the top-right shows the different values the {feature} attribute takes.<br>'NA' denotes those tokens which do not possess the {feature} attribute.</p>")
		outp.write(f"<img src=\"pos.png\" alt=\"{feature}\">")
		outp.write(f"<h2>Token examples for each POS:</h2>")
		for pos_tag in pos_set:
			outp.write(
				f"<li.h><a href=\"{pos_tag}.html\">&nbsp;{pos_tag}</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</li.h>")

		outp.write(f"<h2>{feature} agreement rules:</h2>\n")
		outp.write(f"<p> The following decision tree visualizes the rules used for classifying presence/absence of morphological agreement between two tokens that are connected by a dependency relation denoted by <i>relation</i>. "
				   f"<i>head-pos</i> and <i>child-pos</i> refer to the POS tag of the head and child token respectively.</p> ")
		outp.write(f"<p> Each node of the tree represents a portion of the data. <i>samples</i> denotes the number of training data points in that node. <i>value</i> is the class distribution within that node. Each edge denotes the feature used for splitting. <br>"
				   f"Leaf nodes contain the description of all of the features that appear in that leaf. <i>*</i> denotes that the feature can take any value.</p>")
		#outp.write(f"<p> Given that all feature values for {feature} are often not equally probable, we use a threshold <i>t</i> to decide the probability of agreement vs non-chance agreement. Click on <i>p</i> to toggle between showing/hiding the tree with p-value=p</p>")
		#outp.write(f"<p> We evalute the tree on test data along three metrics. <i>Unweighted Distributional Similarity</i> (UDS) measures how the class distribution over the training data matches with the test data for each leaf. <i>Weighted Distributional Similarity</i> (WDS) weighs the UDS score with percent of test data in that leaf. <i> Agreement-Only Distributional Similarity</i> (ADS) measures the UDS score for only the \"agreement\" class </p>  ")

		outp.write(f"<h2> <a id=\"show_image9\">Tree for p={threshold}</a> </h2>")
		outp.write(f"<div id=\"show_image9div\" >")
		outp.write(f"<p> Click on <button id=\"show_summary9\">Summary</button> to show summary of agreement rules. </p>")

		editedgraph = deepcopy(dot_data.getvalue()).split("\n")
		tree_dictionary, top_nodes = {}, []
		leafnodes = []
		leafedges = {}
		leafvalues = {}
		graph, collated_graph, relabeled_leaves, collate_leaves, graphLines = getTree(dot_data, editedgraph, else_nodes, feature, leaves, nodes,
														  tree_dictionary, top_nodes, leafnodes, leafedges, leafvalues,
														  threshold=threshold)
		'''
		WDS_9, DS_9, A_9, test_leaves_distr = utils.distributional_metric(relabeled_leaves, test_path, feature,
													   data_loader.feature_distribution[feature],
													   test_samples, threshold=threshold, hard=args.hard)
		'''
		automated_acc, test_leaves_distr = utils.automated_metric(relabeled_leaves, test_path, feature,
													   data_loader.feature_distribution[feature],
													   test_samples, threshold=threshold, hard=args.hard, traindata=data)
		#collated_graph = addValuesEval(collated_graph, graphLines, test_leaves_distr)
		with open(f'{args.seed}_{lang_full}_{feature}_{percent}_9.pkl', 'wb') as f:
			pickle.dump(collate_leaves, f)
		summary = getLeafInfo(collate_leaves, feature, leafvalues)
		outp.write(f"<div id=\"summary9\"> {summary} </div>")
		image = f"{folder_name}/" + lang_full + "/" + feature + "/" + feature + "_collate9.png"
		collated_graph.write_png(image)
		#collated_graph.write_pdf(f"{folder_name}/" + lang_full + "/" + feature + "/" + feature + "9.pdf")
		#collated_graph.write_pdf(f'{lang_full}-{feature}.pdf')
		outp.write(f"<img id=\"collate9img\"  src=\"{feature}_collate9.png\" alt=\"{feature}\"> ")
		outp.write(f"<h3> Examples for each leaf node: \n </h3>")
		for leaf_num, _ in enumerate(collate_leaves):
			outp.write(
				f"<li.h><a href=\"{feature}-{leaf_num}-.html\">&nbsp;Leaf-{leaf_num}</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</li.h>")
		if graph:
			outp.write(f"<p> Click on <button id=\"show_collate9\">Expand</button> to expand the tree. </p>")
			image = f"{folder_name}/" + lang_full + "/" + feature + "/" + feature + "9.png"
			graph.write_png(image)

			outp.write(f"<img id=\"my_images9\" src=\"{feature}9.png\" style=\"display:none;\">")

		#outp.write(
		#	f"<h3> <b>Test Metrics </b> </h3> <p> Unweighted Distributional Similarity: {DS_9}")
		#outp.write(f"Agreement-Only Distributional Similarity: {A_9} </p></div>")

		#print("test" + percent + ", " + str(train_samples) + ", " +   lang + ", " + feature + ", " + str(WDS_9) + ", " + str(DS_9) + ", " + str(A_9))

		print("test" + percent + ", " + str(train_samples) + ", " + lang + ", " + feature + ", " + str(	automated_acc) )

		if dev_samples > 0:
			'''
			dev_wds, dev_ds, dev_ads, _ = utils.distributional_metric(relabeled_leaves, dev_path, feature,
																			  data_loader.feature_distribution[feature],
																			  dev_samples, threshold=threshold,
																			  hard=args.hard)
			'''
			automated_acc, test_leaves_distr = utils.automated_metric(relabeled_leaves, dev_path, feature,
																	  data_loader.feature_distribution[feature],
																	  dev_samples, threshold=threshold, hard=args.hard,
																	  traindata=data)
			print("dev" + percent + ", " +  str(train_samples) + ", " + lang + ", " + feature + ", " + str(automated_acc)  )

		if not args.inTh:
			return
		outp.write(f"<h2> <a id=\"show_image5\">Tree for p=0.5</a> </h2>")
		outp.write(f"<div id=\"show_image5div\" style=\"display:none;\">")
		outp.write(f"<p> Click on <button id=\"show_summary5\">Summary</button> to show summary of agreement rules. </p>")

		editedgraph = deepcopy(dot_data.getvalue()).split("\n")
		tree_dictionary, top_nodes = {}, []
		leafnodes = []
		leafedges = {}
		leafvalues = {}
		graph, collated_graph, relabeled_leaves, collate_leaves, graphLines = getTree(dot_data, editedgraph, else_nodes, feature, leaves, nodes, tree_dictionary,
										   top_nodes, leafnodes, leafedges, leafvalues, threshold=0.5)
		WDS_5, DS_5, A_5, test_leaves_distr = utils.distributional_metric(relabeled_leaves, test_path, feature,
													   data_loader.feature_distribution[feature],
														   test_samples, threshold=0.5, hard=args.hard)
		#collated_graph = addValuesEval(collated_graph, graphLines, test_leaves_distr)

		with open(f'{args.seed}_{lang_full}_{feature}_{percent}_5.pkl', 'wb') as f:
			pickle.dump(collate_leaves, f)
		summary = getLeafInfo(collate_leaves, feature, leafvalues)
		outp.write(f"<div id=\"summary5\"> {summary} </div>")
		image = f"{folder_name}/" + lang_full + "/" + feature + "/" + feature + "_collate5.png"
		collated_graph.write_png(image)
		outp.write(f"<img id=\"collate5img\"  src=\"{feature}_collate5.png\" alt=\"{feature}\"> ")
		outp.write(f"<h3> Examples for each leaf node: \n </h3>")
		for leaf_num, _ in enumerate(collate_leaves):
			outp.write(
				f"<li.h><a href=\"{feature}-{leaf_num}-5.html\">&nbsp;Leaf-{leaf_num}</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</li.h>")
		if graph:
			image = f"{folder_name}/" + lang_full + "/" + feature + "/" + feature + "5.png"
			outp.write(f"<p> Click on <button id=\"show_collate5\">Expand</button> to expand the tree. </p>")
			graph.write_png(image)
			outp.write(f"<img id=\"my_images5\" src=\"{feature}5.png\" style=\"display:none;\" >")
			graph.write_pdf(f'./{lang_full}-{feature}-{0.5}.pdf')

		#outp.write(
		#	f"<h3> <b>Test Metrics </b> </h3> <p>Unweighted Distributional Similarity: {DS_5}")
		#outp.write(f"Agreement-Only Distributional Similarity: {A_5} </p> </div>")

		getLeafInfo(collate_leaves, feature, leafvalues, extension="5")
		print(percent + ", " + lang + ", " + feature + ", " + str(WDS_5) + ", " + str(DS_5) + ", " + str(A_5) + "\n")

def addValuesEval(collated_graph, graphLines, test_leaves_distr):
	graphforeval = []
	for line in graphLines:
		if 'class' in line:
			leaf_num = int(line.split("Leaf-")[-1].split("\\n")[0])
			info = line.split("\\l")
			distr = test_leaves_distr[leaf_num]
			distr = 'test-value = [' + str(distr[0]) + "," + str(distr[1]) + "]\\l"
			graphforeval.append("\\l".join(info[0:-1]) + "\\l" + distr + info[-1])
		else:
			graphforeval.append(line)
	collated_graph = pydotplus.graph_from_dot_data("\n".join(graphforeval))
	return collated_graph

def getLeafInfo(collate_leaves, feature, leafvalues, extension=""):
	# Add the examples for each leaf
	allAgreement = {}
	relationcount = defaultdict(lambda:0)
	total = 0
	for leaf_num, _ in enumerate(collate_leaves):
		leaf_node = collate_leaves[leaf_num]

		ag_examples, dis_examples, relation_dict, head_pos_dict, child_pos_dict, example, sorted_examplecount = utils.getAggreeingExamples(
		leaf_node, feature, data, leafvalues[leaf_num], data_loader.train_random_samples)

		with open(f"{folder_name}/{lang}/{feature}/{feature}-{leaf_num}-{extension}.html", 'w') as outp2:
			HEADER = ORIG_HEADER.replace("main.css", "../../main.css")
			outp2.write(HEADER + '\n')
			outp2.write(f'<ul class="nav"><li class="nav"><a class="active" href=\"../../index.html\">Home</a>'
					 f'</li><li class="nav"><a href=\"../../introduction.html\">Usage</a></li>'
					 f'<li class="nav"><a href=\"../../about.html\">About Us</a></li></ul>')
			outp2.write(f"<br><li><a href=\"{feature}.html\">Back to {feature} {language_fullname} page</a></li>\n")
			if len(relation_dict) > 0:
				utils.plot_histogram(relation_dict, color='peru', type='Relation',
								 	file=f"./{folder_name}/{lang}/{feature}/{leaf_num}-{extension}")
			if len(head_pos_dict) >0:
				utils.plot_histogram(head_pos_dict, color='seagreen', type='Head-POS',
									 file=f"./{folder_name}/{lang}/{feature}/{leaf_num}-{extension}")
			if len(child_pos_dict) > 0:
				utils.plot_histogram(child_pos_dict, color='olivedrab', type='Child-POS',
								 		file=f"./{folder_name}/{lang}/{feature}/{leaf_num}-{extension}")
			outp2.write(f"<h2>Distribution of features within this leaf </h2>")

			outp2.write(
				f"<p style = \"float: left; font-size: 15pt; text-align: center; width: 33%; \"><img src=\"{leaf_num}-Relation.png\" alt=\"Relation\" style=\"width:100%\"></p>")
			outp2.write(
				f"<p style = \"float: left; font-size: 15pt; text-align: center; width: 33%; \"><img src=\"{leaf_num}-Head-POS.png\" alt=\"head-pos\" style=\"width:100%\"></p>")
			outp2.write(
				f"<p style = \"float: left; font-size: 15pt; text-align: center; width: 33%;\"><img src=\"{leaf_num}-Child-POS.png\" alt=\"child-pos\" style=\"width:100%\"></p><br>")
			if not ag_examples:
				outp2.write("\tNo agree examples found.<br>")
			else:
				outp2.write("<h2>Agreement Rules sorted by frequency.</h2> <ul>")

				required_relation, required_head, required_child, _, _ = utils.parseLeafInformation(leaf_node[1])
				for (key, val) in sorted_examplecount:
					rule_template = ""
					(relation, head_pos, child_pos) = key
					ex = example[key]

					if leaf_node[0] == "agreement":
						if relation not in allAgreement:
							allAgreement[relation] = {}
						allAgreement[relation][(head_pos, child_pos)] = val

						relationcount[relation] += val
						total += val
					if required_relation is not None:
						if relation not in relation_map:
							if relation.split("@")[0] in relation_map:
								full_relation_name = relation_map[relation.split("@")[0]][0]
								url = relation_map[ relation.split("@")[0]][1]
							else:
								full_relation_name = relation
								url=f'https://universaldependencies.org/'
						else:
							full_relation_name = relation_map[relation][0]
							url = relation_map[relation][1]

						rule_template = f" When the dependent token is the "
						rule_template += f"<i>{full_relation_name}</i>(<a href=\"{url}\">{relation})</a> of the head token, "
					if required_head is not None:
						if len(rule_template) == 0:
							rule_template = f'<p> When the head token is <i>{head_pos}</i>  '
						else:
							rule_template += f" and the head token is <i>{head_pos}</i> "
					if required_child is not None:
						if len(rule_template) == 0:
							rule_template = f'<p> When the dependent token is <i>{head_pos}</i>  '
						else:
							rule_template += f" and the dependent token is <i>{child_pos}</i>."
					outp2.write(f"<li>{rule_template}</li>")
					utils.example_web_print(ex, outp2, data)
					outp2.write(f"<br>")
				outp2.write("</ul>")

			if not dis_examples:
				outp2.write("\tNo disagree examples found.<br>")
			else:
				outp2.write("\t<br><h2>Disagree Examples:</h2>")
				for ex in dis_examples:
					utils.example_web_print(ex, outp2, data)

			outp2.write(FOOTER)


	#GetSummary of the agreement rules
	#Sort the agreement by relation type
	if len(allAgreement) == 0:
		summary = [f'<p>There is no agreement for {feature}.</p>']
		return summary
	sorted_relation = sorted(relationcount.items(), key=lambda kv:kv[1],reverse=True)
	summary = []
	summary.append(f'<ol>')
	rulenum=1
	headchilddict = defaultdict(set)
	for (relation, val) in sorted_relation:
		sorted_headchild = sorted(allAgreement[relation].items(), key=lambda kv:kv[1], reverse=True)
		#Group-by head
		GroupbyHead, GroupbyHeadInfo, GroupByChild, GroupByChildInfo = defaultdict(lambda :0), defaultdict(set), defaultdict(lambda:0), defaultdict(set)
		child, head = False, False
		if relation is None:
			full_relation_name = 'anything'
		else:
			if relation not in relation_map:
				if relation.split("@")[0] in relation_map:
					full_relation_name = relation_map[relation.split("@")[0]][0]
				else:
					full_relation_name = relation
			else:
				full_relation_name = relation_map[relation][0]
		for (headchild, value) in sorted_headchild:
			if value *1.0/val < 0.5:
				continue
			(head, child) = headchild
			if child is not None and head is not None:
				GroupByChild[child] += value
				GroupByChildInfo[child].add(head)
				GroupbyHead[head] += value
				GroupbyHeadInfo[head].add(child)
				child, head = True, True
			else:
				if head is None:
					head = False
				else:
					GroupbyHead[head] = 0
					head=True
				if child is None:
					child = False
				else:
					GroupByChild[child] = 0
					child=True

		if not head and child:
			all_childpos = ",".join(list(GroupByChild.keys()))
			key=f'<i>{all_childpos}</i> tokens agree with their head'
			value=f'<i>{full_relation_name}({relation})</i>'

			headchilddict[key].add(value)
		elif not child and head:
			all_headpos = ",".join(list(GroupbyHead.keys()))
			key=f'<i>{all_headpos}</i> tokens agree with their dependent tokens'
			value=f'<i>{full_relation_name}({relation})</i>'
			headchilddict[key].add(value)

		elif not child and not head:
			key = f'All tokens agree with their head tokens'
			value = f'<i>{full_relation_name} ({relation})</i>'
			headchilddict[key].add(value)

		elif child and head:
			#sort group-by-head and group-by-child and compare the highest values, whichever is higher, choose that as the condition of groupping
			sort_grpchild = sorted(GroupByChild.items(), key =lambda kv:kv[1], reverse=True)
			sort_grphead = sorted(GroupbyHead.items(), key = lambda  kv:kv[1], reverse=True)

			if sort_grpchild[0][1] > sort_grphead[0][1]: #Grp by child
				for (child, _) in sort_grpchild:
					headpos = ", ".join(list(GroupByChildInfo[child]))
					headchilddict[f'<i>{child}</i> tokens agree when head token belongs to [<i>{headpos}</i>]'].add(f'<i>{full_relation_name}({relation})</i>')
			else:
				for (head, _) in sort_grphead:
					childpos = ", ".join(list(GroupbyHeadInfo[head]))
					headchilddict[f'<i>{head}</i> tokens agree when the dependent token belongs to [<i>{childpos}</i>]'].add(f'<i>{full_relation_name}({relation})</i>')

	for rule, relations in headchilddict.items():
		summary.append(f'<li> {rule} for the dependency relations: {", ".join(list(relations))} </li><br>')
		rulenum+=1
	summary.append(f'</ol>')
	summary = "".join(summary)
	return summary

def getTree(dot_data, editedgraph, else_nodes, feature, leaves, nodes, tree_dictionary, topnodes, leafnodes, leafedges, leafvalues, threshold):
	i = 0
	leaf_num = 0
	leftstart = 0
	rightstart = 0
	relabeled_leaves = {}
	for linenum, line in enumerate(dot_data.getvalue().split("\n")):
		if "<=" in line:  # If
			info = line.split("<=")
			info_index = info[-1].find("\\")
			nodenum = line.split("[")[0]
			textinfo = info[-1][info_index + 2:].split("fillcolor=")[0]
			edge = info[0] + " in " + nodes[i]
			values = info[-1].split("\\nclass")[0].split("value = ")[1].replace("[", "").replace("]", "").replace("\'", "").split(
				",")
			disagree, agree = int(values[0]), int(values[1])
			color = utils.colorRetrival(agree, disagree, data_loader.feature_distribution[feature], threshold, args.hard)
			editedgraph[linenum] = line.split("[")[0] + "[label=\"node - " + nodenum + "\\n" + textinfo.replace("\\n","\\l").replace("class = agree", "").replace("class = disagree", "") + 'fillcolor=\"{0}\"] ;'.format(color)
			tree_dictionary[int(nodenum)] = {"children": [], "edge": nodes[i], "info": editedgraph[linenum]}
			i += 1
			topnodes.append(int(nodenum))

		elif "->" in line:  # Edge
			lefttext = "[labeldistance={0},labelangle=50, headlabel=\"{1}\",labelfontsize=10];"
			righttext = "[labeldistance={0},labelangle=-50, headlabel=\"   {1}\", labelfontsize=10];"
			info = line.replace('\'', '').replace(";", "").split("->")
			leftnode, rightnode = int(info[0]), int(info[-1].split("[")[0])
			if rightnode - leftnode == 1:
				edge = nodes[leftstart]
				input = edge.split("[")[-1].replace("]", "").split(",")
				edge = edge.split("[")[0] + utils.printMultipleLines(input, t=7)
				leftstart += 1
				newtext = line.split(str(rightnode))[0] + " " + str(rightnode) + " " + lefttext.format(3.5, edge)
			else:
				edge = else_nodes[rightstart]
				input = edge.split("[")[-1].replace("]", "").split(",")
				edge = edge.split("[")[0] + utils.printMultipleLines(input, t=7)
				rightstart += 1
				newtext = line.split(str(rightnode))[0] + " " + str(rightnode) + " " + righttext.format(3.5, edge)
			editedgraph[linenum] = newtext
			tree_dictionary[leftnode]["children"].append(rightnode)
			tree_dictionary[rightnode]["top"] = leftnode
			leafedges[rightnode] = edge

		elif ">" in line:  # Else
			info = line.split(">")
			info_index = info[-1].find("\\")
			nodenum = line.split("[")[0]
			textinfo = info[-1][info_index + 2:].split("fillcolor=")[0]
			edge = info[0] + " in " + nodes[i]
			values = info[-1].split("\\nclass")[0].split("value = ")[1].replace("[", "").replace("]", "").replace("\'","").split(
				",")
			disagree, agree = int(values[0]), int(values[1])
			color = utils.colorRetrival(agree, disagree, data_loader.feature_distribution[feature], threshold, args.hard)
			editedgraph[linenum] = line.split("[")[0] + "[label=\"node - " + nodenum + "\\n" + textinfo.replace("\\n",
																												"\\l").replace(
				"class = agree", "").replace("class = disagree", "") + 'fillcolor=\"{0}\"] ;'.format(color)
			tree_dictionary[int(nodenum)] = {"children": [], "edge": nodes[i], "info": editedgraph[linenum]}
			i += 1
			topnodes.append(int(nodenum))

		else:  # Leaf
			if "class" in line:

				info = line.split("label=\"")
				info[-1] = "\\n".join(info[-1].split("\\n")[1:])
				leafvalues[leaf_num] = info[-1].split("\\n")[1].split("value = ")[1].replace("[", "").replace("]",
																											  "").replace(
					"\'", "").split(",")

				disagree, agree = int(leafvalues[leaf_num][0]), int(leafvalues[leaf_num][1])
				t = agree * 1.0 / (disagree + agree)
				agreement = "chance-agreement\\n"
				if utils.isAgreement(data_loader.feature_distribution[feature], agree, disagree, threshold, args.hard):# t >= threshold:
					agreement = "agreement\\n"

				color = utils.colorRetrival(agree, disagree, data_loader.feature_distribution[feature], threshold, args.hard)
				text_position = info[-1].split("agree")
				classinfo = text_position[0].replace("dis", "") + agreement + "\",fillcolor=\"{0}\"] ;".format(color)

				nodenum = line.split("[")[0]
				data_info = ""
				(leaf_node_class, leaf_node_data) = leaves[leaf_num]
				relabeled_leaves[leaf_num] = (leaf_node_data, agree, disagree)
				if leaf_node_data["head_feature"] != None:
					data_info = leaf_node_data["head_feature"] + "\\n\\n"

				if leaf_node_data["child_feature"] != None:
					data_info += leaf_node_data["child_feature"] + "\\n\\n"

				if leaf_node_data["relation"] == None:
					data_info += "relation = *" + "\\l\\l"
				else:
					class_relations = data_loader.class_relations[leaf_node_class]
					input = set(
						leaf_node_data["relation"].replace("\'", "").replace("[", "").replace("]", "").split(","))
					extra = input - class_relations
					actual = input - extra
					leaf_node_data["relation"] = ",".join(list(actual))
					data_info += "relation = " + utils.printMultipleLines(actual) + "\\l"

				if leaf_node_data["head"] == None:
					data_info += "head-pos = *" + "\\l\\l"
				else:
					class_pos = data_loader.class_headpos[leaf_node_class]
					input = set(
						leaf_node_data["head"].replace("\'", "").replace("[", "").replace("]", "").split(","))
					extra = input - class_pos
					actual = input - extra
					leaf_node_data["head"] = ",".join(list(actual))
					data_info += "head-pos = " + utils.printMultipleLines(actual) + "\\l"

				if leaf_node_data["child"] == None:
					data_info += "child-pos = *" + "\\l\\l"
				else:
					class_pos = data_loader.class_childpos[leaf_node_class]
					input = set(
						leaf_node_data["child"].replace("\'", "").replace("[", "").replace("]", "").split(","))
					extra = input - class_pos
					actual = input - extra
					leaf_node_data["child"] = ",".join(list(actual))
					data_info += "child-pos = " + utils.printMultipleLines(actual) + "\\l"
				textinfo = info[0] + "label=" + "\"Leaf- " + str(
					leaf_num) + "\\n" + data_info.lower() + classinfo.replace("\\n", "\\l")
				editedgraph[linenum] = textinfo

				tree_dictionary[int(nodenum)] = {"children": [], "edge": data_info.lower(),
												 "info": editedgraph[linenum]}
				leaf_num += 1
				leafnodes.append(int(nodenum))

	if args.prune:
		editedgraph, tree_dictionary, leafnodes, topleafnodes, removednodes, topnodes = utils.pruneTree(editedgraph, tree_dictionary, topnodes, leafnodes, leafedges, feature)
		#Original decision tree
		graph = pydotplus.graph_from_dot_data(editedgraph)

		#Collated tree with leaves with same labels are merged
		collatedGraph,leaves, relabeled_leaves = utils.collateTree(data_loader.feature_distribution[feature], leafedges, editedgraph, topleafnodes, tree_dictionary, leaves, threshold, topnodes, removednodes, args.hard)
		collated_graph  = pydotplus.graph_from_dot_data("\n".join(collatedGraph))
		new_graph = collatedGraph

	else:
		graph = pydotplus.graph_from_dot_data("\n".join(editedgraph))
		new_graph=editedgraph
	return graph, collated_graph, relabeled_leaves, leaves, new_graph

def printPOSInfomation(feature):
	pos_set = data_loader.getHistogram(folder_name, lang, train_path, feature)
	for pos, pos_dict in data_loader.feature_forms_num.items():
		feature_values = pos_dict.keys()
		filename = f"{folder_name}/{lang}/{feature}/{pos}.html"
		with open(filename, 'w') as outp:
			HEADER = ORIG_HEADER.replace("main.css", "../../main.css")
			outp.write(HEADER + '\n')
			outp.write(f'<ul class="nav"><li class="nav"><a class="active" href=\"../../index.html\">Home</a>'
					 f'</li><li class="nav"><a href=\"../../introduction.html\">Usage</a></li>'
					 f'<li class="nav"><a href=\"../../about.html\">About Us</a></li></ul>')
			outp.write(f"<br> <a href=\"{feature}.html\">Back to {feature} information</a><br>")
			outp.write(f"<h1> Examples of word types for  each {feature} value : </h1>")
			outp.write(f"<p> The word types shown below are ordered by token frequency in the treebank.")

			outp.write(f'<table><col><colgroup span=\"{len(feature_values)}\"></colgroup><tr><th rowspan=\"2\" style=\"text-align:center\">Lemma</th><th rowspan=\"2\" style=\"text-align:center\"> Morphosyntactic <br> Attributes</th><th colspan=\"{len(feature_values)}	scope=\"colgroup\" \" style=\"text-align:center\">{feature}</th></tr><tr>')

			for feat in feature_values:
				outp.write(f'<th scope=\"col\"> {feat} </th>')
			outp.write('</tr>')
			#Sort the tokens within a pos using lemma
			sorted_lemma_dict = sorted(data_loader.lemma_freq[pos].items(), key=lambda kv: kv[1], reverse=True)[:30]
			for (lemma, _) in sorted_lemma_dict:
				for inflection in  data_loader.lemma_inflection[pos][lemma].keys():
					outp.write(f'<tr><td> {lemma} </td>')
					outp.write(f'<td> {inflection} </td>')
					inflection_feature_value = data_loader.lemma_inflection[pos][lemma][inflection]
					for feat in feature_values:
						if feat in inflection_feature_value:
							outp.write(f'<td> {inflection_feature_value[feat]} </td>')
						else:
							outp.write(f'<td> - </td> ')
					outp.write('</tr>')
			outp.write('</table>')
			outp.write(FOOTER)
	return pos_set

def train(feature):
	x_train, x_test, y_train, y_test = train_features[feature] , test_features[feature], \
													 train_output_labels[feature], test_output_labels[feature]
	if dev_path:
		x_dev, y_dev = dev_features[feature], dev_output_labels[feature]
		x = np.concatenate([x_train, x_dev])
		y = np.concatenate([y_train, y_dev])
		test_fold = np.concatenate([
			# The training data.
			np.full(x_train.shape[0], -1, dtype=np.int8),
			# The development data.
			np.zeros(x_dev.shape[0], dtype=np.int8)
		])
		cv = sklearn.model_selection.PredefinedSplit(test_fold)
	else:
		x,y = x_train, y_train
		cv = None

	# Create lists of parameter for Decision Tree Classifier
	criterion = ['gini', 'entropy']
	parameters = {'criterion':criterion, 'max_depth':np.arange(6, 15), 'min_impurity_decrease':[1e-3]}
	decision_tree = sklearn.tree.DecisionTreeClassifier()
	model = GridSearchCV( decision_tree , parameters, cv=cv)
	model.fit(x, y)

	trainleave_id = model.best_estimator_.apply(x)

	uniqueleaves = set(trainleave_id)
	uniqueleaves = sorted(uniqueleaves)
	leafcount = {}
	for i, leaf in enumerate(uniqueleaves):
		leafcount[i] = round(np.count_nonzero(trainleave_id == leaf) * 100 / len(trainleave_id), 2)

	feature_names = []
	for i in range(len(data_loader.pos_dictionary)):
		feature_names.append("head@" + data_loader.pos_id2tag[i])
	for i in range(len(data_loader.pos_dictionary)):
		feature_names.append("child@" + data_loader.pos_id2tag[i])
	for i in range(len(data_loader.relation_dictionary)):
		feature_names.append("relation@" + data_loader.relation_id2tag[i])

	feature_names.append(feature + "@child")
	feature_names.append(feature + "@head")
	tree_rules = export_text(model.best_estimator_, feature_names= feature_names, max_depth=model.best_params_["max_depth"])
	treelines = utils.printTreeForBinaryFeatures(tree_rules, data_loader.pos_id2tag, data_loader.relation_id2tag,  data_loader.used_relations, data_loader.used_head_pos, data_loader.used_child_pos, feature)
	leaves = utils.constructTree(treelines, feature)
	assert len(leaves) == len(leafcount)
	dev_samples = len(x_dev) if dev_path else 0
	printTreeWithExamplesPDF(model, tree_rules, treelines, leaves, feature, leafcount, feature_names, len(y_test), len(x_train), dev_samples)
	with open(f"{folder_name}/{lang}/index.html", 'a') as outp:
		outp.write(
			f"<li>{feature} agreement:. <a href=\"{feature}/{feature}.html\">Examples</a></li>\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--file", type=str, default="./decision_tree_files.txt")
	parser.add_argument("--features", type=str, default="Gender+Person+Number+Tense+Mood+Case", nargs='+')
	parser.add_argument("--prune", action="store_true", default=True)
	parser.add_argument("--binary", action="store_true", default=True)
	parser.add_argument("--debug_folder", type=str, default="./")
	parser.add_argument("--percent", type=float, default=1.0)
	parser.add_argument("--relation_map", type=str, default="./relation_map")
	parser.add_argument("--inTh", action="store_true", default=False)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--simulate", action="store_true", default=False)
	parser.add_argument("--hard", action="store_true", default=False)
	parser.add_argument("--folder_name", type=str, default='./website/')


	args = parser.parse_args()
	folder_name = f'{args.folder_name}'
	with open(f"{folder_name}/header.html") as inp:
		ORIG_HEADER = inp.readlines()
	ORIG_HEADER = ''.join(ORIG_HEADER)

	with open(f"{folder_name}/footer.html") as inp:
		FOOTER = inp.readlines()
	FOOTER = ''.join(FOOTER)

	with open(args.file, "r") as inp:
		files = []
		for file in inp.readlines():
			if file.startswith("#"):
				continue
			files.append(file)

	d = {}
	relation_map = {}
	with open(args.relation_map, "r") as inp:
		for line in inp.readlines():
			relation_map[line.split(";")[0]] = (line.split(";")[1].lstrip().rstrip(), line.split(";")[-1].lstrip().rstrip())


	with open(f"{folder_name}/index.html", 'w') as op:
		op.write(ORIG_HEADER + '\n')
		op.write(f'<ul class="nav"><li class="nav"><a class="active" href=\"index.html\">Home</a>'
				 f'</li><li class="nav"><a href=\"introduction.html\">Usage</a></li>'
				 f'<li class="nav"><a href=\"about.html\">About Us</a></li></ul>')
		op.write(f'<h2>LASE: Language Structure Explorer</h2>')
		op.write(f'<h3>	Most of the world\'s languages have an adherence to grammars — sets of morpho-syntactic rules specifying how to create sentences in the language. '
				 f'Hence, an important step in the understanding and documentation of languages is the creation of a grammar sketch, a concise and human-readabled escription of the unique characteristics of that particular language. </h3>')

		op.write(f'<h3> LASE is a tool for exploring language structure and provides an automated framework for '
				 f'extracting a first-pass grammatical specification from raw text in a concise,  human-and machine-readable  format.'
				 f'</h3>')
		op.write("<h3> We apply our framework to all languages of the <a href=\"https://universaldependencies.org/\"> Universal Dependencies project </a>. </h3><h3> Here are the languages (and treebanks) we currently support.</h3><br><ul>")
		op.write(f'<h3> Linguistic analysis based on automatically parsed syntactic analysis </h3>')
		op.write(f'<table><tr><th>ISO</th><th>Language</th><th>Treebank</th><th>Linguistic Analysis </th></tr>')


	fnum = 0
	args.features = args.features.split("+")
	while fnum < len(files):

		treebank = files[fnum].strip()
		fnum += 1
		#print("Processing treebank ", treebank)
		train_path, dev_path, test_path = None, None, None
		for [path, dir, inputfiles] in os.walk(treebank):
			for file in inputfiles:
				if "-train.conllu" in file:
					train_path = treebank + "/" + file
					if args.simulate: #For simulated low-resource training
						percent = treebank.split("-")[-2] + "-" + treebank.split("-")[-1]
						lang = train_path.strip().split('/')[-1].split("_")[0]
						lang += "-" + percent
					else:
						percent = 'all'
						lang = train_path.strip().split('/')[-1].split("-")[0]

				if "dev.conllu" in file:
					dev_path = treebank + "/" + file

				if "test.conllu" in file:
					test_path = treebank + "/" + file


		if train_path is None:
			continue
		language_fullname = "_".join(os.path.basename(treebank).split("_")[1:])
		lang_full = lang
		f = train_path.strip()


		i = 0
		with open(f"{folder_name}/index.html", 'a') as op:
			lang_id = lang.split("_")[0]
			language_name = language_fullname.split("-")[0]
			treebank_name = language_fullname.split("-")[1]
			op.write(f'<tr><td>{lang_id}</td> '
					 f'<td>{language_name}</td> '
					 f'<td> {treebank_name} </td>'
					 f' <td> <li> <a href=\"{lang}/index.html\">Agreement</a></li> </td>\n')

		try:
			os.mkdir(f"{folder_name}/{lang}")
		except OSError:
			i =0
	
		with open(f"{folder_name}/{lang}/index.html", 'w') as outp:
			HEADER = ORIG_HEADER.replace("main.css", "../main.css")
			outp.write(HEADER + "\n")
			outp.write(f'<ul class="nav"><li class="nav"><a class="active" href=\"../index.html\">Home</a>'
					 f'</li><li class="nav"><a href=\"../introduction.html\">Usage</a></li>'
					 f'<li class="nav"><a href=\"../about.html\">About Us</a></li></ul>')
			outp.write(f"<br><a href=\"../index.html\">Back to language list</a><br>")
			outp.write( f"<h1> {language_fullname} </h1> <br>\n")
			outp.write(f'<h3> We  present  a  framework that automatically creates a first-pass specification of morphological agreement rules for various morphological features (Gender, Number, Person, Tense, Mood and Case.) from a raw text corpus for the language in question.</h3>')
			outp.write("<h3> We parsed the <a href=\"https://universaldependencies.org/udw18/PDFs/33_Paper.pdf\">Surface-Syntactic Universal Dependencies</a> (SUD) data in order to extract these rules.</h3>\n")
			outp.write(f"<br><strong>{language_fullname}</strong> exhibits the following agreement:<br><ul>")

		#Decision Tree code
		data = pyconll.load_from_file(f"{f}")
		data_loader = dataloader.DataLoader(args, relation_map)

		# Creating the vocabulary
		inputFiles = [train_path, dev_path, test_path]
		data_loader.readData(inputFiles)

		# creating the features for training
		train_features, train_output_labels = data_loader.getBinaryFeatures(train_path,type="train", p=args.percent, shuffle=True)
		if dev_path:
			dev_features, dev_output_labels = data_loader.getBinaryFeatures(dev_path, type="dev", p=1.0, shuffle=False)
		test_features, test_output_labels = data_loader.getBinaryFeatures(test_path, type="test", p=1.0, shuffle=False)

		for feature in args.features:
			if feature in test_features and feature in train_features:
				try:
					train(feature)
				except:
					print("error processing ", feature, lang)


		with open(f"{folder_name}/{lang}/index.html", 'a') as outp:
			outp.write("</ul><br><br><br>\n" + FOOTER+"\n")

	with open(f"{folder_name}/index.html", 'a') as outp:
		outp.write(f'</table>')
		outp.write("</ul><br><br><br>\n" + FOOTER+"\n")
