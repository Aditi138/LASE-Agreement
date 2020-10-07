# LASE: Automated Extraction of Agreement Rules

## Requirements
A python version >=3.5 is required. Additional requirements are present in the **requirements.txt**

1. In the **decision_tree_files.txt**, enter the path to the treebanks for which you want to extract the rules. This code can work with UD/SUD dependency (v2.5) treebanks.
To download all the SUD treebanks: https://surfacesyntacticud.github.io/data/

2. Run the following command to create the rules:

```
   python create_trees_website.py --folder_name website | tee output.log
```

Running this will create the html files with the decision trees, examples and other relevant information. Simply open the `website/index.html` in any browser to navigate the trees.
This will also output the leaves as pickle files in the running folder. Example pickle files are provided in the outputs folder.
In the `output.log` file the results for ARM metrics are stored in the order: **setting, treebank, number of sentences, ARM**
This command by default uses the statistical thresholding.

3. To learn the rules with hard threshold, simply add the command `--hard` in the run command.


## Low-resource experiments.
1. The five training runs for **x=50,100,500** are present in the **./sud-data** directory.
2. Run the following command to create the rules:

```
    python create_trees_website.py --folder_name website --simulate | tee transfer.log
```

This will output the ARM metric in `transfer.log`. To compute the average of all five runs, run the following:

```
    python  computeAverage.py --input transfer.log
```


## Baseline
1. To run the baseline, we re-use the pickle files generated from before (which were stored in outputs/):

```
    python baseline.py --input outputs/ | tee baseline.out
```

This will output the results in `baseline.out`.

## Creating the annotation site
1. Run:
   ```
    python create_triples.py
   ```

This will create the files for annotation under **annotation_site/templates**. Use the treebanks from **decision_tree_files.txt.**

2.To host it on the a flask server, run:

 ```
    python serve.py
 ```
This will host it on **http://localhost:5000/el/**. The triples selected for annotation are also outputted in the run folder. These are stored in **annotation_site/templates** as shown here.
After annotation, the pickle files will be stored with the annotations.

We release the annotation results under **annotation_site/**.


## Citing
If you make use of this software for research purposes, we will appreciate citing the following: 
``` 
@inproceedings{chaudhary20emnlp,
                   title = {Automatic Extraction of Rules Governing Morphological Agreement},
                   author = {Aditi Chaudhary and Antonios Anastasopoulos and Adithya Pratapa and David R. Mortensen and Zaid Sheikh and Yulia Tsvetkov and Graham Neubig},
                   booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP)},
                   address = {Online},
                   month = {November},
                   year = {2020}}
```

