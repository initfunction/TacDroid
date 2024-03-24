# TacDroid

This is the implication of the paper "TacDroid: Detection of Illicit Apps through Hybrid Analysis of UI-based Transition Graphs"

# Requirement

JDK 8

Python 3.7

Android SDK

# Step

TacDroid needs to perform static analysis and dynamic analysis respectively to obtain static UTG and dynamic UTG.

## static UTG construction

1. Open the PermDroid project using IDEA or eclipse
2. Modify the three parameters of androidPlatformLocation, apkDir, and apkName of `src/PermDroid/staticanalysis/Main.java`

3. Run Main.java to get static UTG

The output file is : 'apkDir'+'apkName'+'staitc'+'sUTG.txt'.

## dynamic UTG construction

We use DroidBot to perform dynamic exploration to obtain dynamic UTG. The specific process can be viewed [DroidBot](https://github.com/initfunction/TacDroid/tree/main/DroidBot)

After executing Dynamic explore, you can get a dynamic UTG of UTG.json

## Link prediction and classification

This folder contains the following files:

- align.py

This is the file defining the Dynamic Graph class and Static Graph class, which includes functions for extracting information from the information folder to construct, align and merge graphs.

- embedding.py

This file is responsible for converting features of nodes in the graph, including activity names, text, etc., into vectors. It is responsible for training, testing, and predicting with multiple feature models.

- prediction.py

This is the file responsible for the link predictor part in link prediction, including model definition, negative sampling method, and setting methods for dataloaders. It is worth noting that it imports the modified MyGCNConv (for implementing enhanced GAE). For convenience, we directly define this class in the torch package, and only need to make simple changes to the original normalization method.

-  globalfeature.py

This is the file responsible for converting the Metadata information of the apps, including icons, application names, package names, etc., into vectors. It is responsible for training, testing, and predicting with multiple feature models.

-  classification.py

This is the file responsible for the Detection and Classification process, defining the model used to classify the UTG with added global features.

-  main.py

This file is used for the entire experimental process, which includes obtaining information from folders with completed static and dynamic feature extraction to construct UTGs. It is used for conducting 5-fold cross-validation for the entire process, outputting accuracy for both the Detection and Classification tasks. This file involves ablation experiments, specifically including the following features: Static UTG, Dynamic UTG, Metadata, and whether to enable link prediction.

In order to facilitate operation, we have integrated the align and link prediction processes into main.py. Configuring the dynamic UTG folder and static UTG folder in main.py can automatically perform the align, link prediction and classification processes.