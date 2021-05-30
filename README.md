# Milestone II: Wikipedia Classification Spring 2021
## By Sean Cafferty and Laura Stagnaro

### This GitHub repo contains several notebooks, python modules, and data sets used to complete the Milestone II project. Below is a description of the folder, python modules and notebooks pertinent to this project.

#### assets folder
- Description: a folder which contains all the necessary data except wiki_train_preprocessed5.csv and glove.6B.50d.txt because they were too large to store in GitHub. wiki_train_preprocessed5.csv can be downloaded here: https://drive.google.com/drive/u/0/folders/1xJn3azqIBENWohlluOAmNXG6fbXcalM4 and glove.6B.50d.txt can be downloaded here: http://nlp.stanford.edu/data/glove.6B.zip
- Note: in order for the below notebooks to run all the data must be stored in a folder titled "assets"

#### custom_word_embeddings folder
- Description: contains all the customer word embeddings created for data exploration
- Note: in order to use the custom word embedding data it must be stored in a folder titled "custom_word_embeddings"

#### feature_extractor.py
- Description: contains functions for feature engineering 
- Notebooks: 1_data_exploration.ipynb, 2_preprocessing.ipynb, 4_deep_learning.ipynb, 5_topic_modeling.ipynb

#### train_model.py
- Description: contains functions for training the different classification models
- Notebooks: 3_train_classification_models.ipynb

#### viz_engine.py
- Description: contains functions for creating different exploratory visualizations
- Notebooks: 1_data_exploration.ipynb, 5_topic_modeling.ipynb

#### word_embedder.py
- Description: contains functions for creating custom word embeddings
- Notebooks: 1_data_exploration.ipynb
- Note: does not need to be used since the custom word embeddings were already created

#### unsupervised_engine.py
- Description: contains functions to perform topic modeling
- Notebooks: 5_topic_modeling.ipynb

#### 1_data_exploration.ipynb
- Description: contains all exploratory data analysis
- Data sets: wiki_train_preprocessed5.csv, WikiTrain_Large_WordEmbeddings_model.bin, negative_class_WordEmbeddings_model.bin, positive_class_WordEmbeddings_model.bin
- Modules: feature_extractor.py, viz_engine.py, word_embedder.py (was used to create word embeddings but does not need to be run again since all created word embeddings are stored in the custom_word_embeddings folder)

#### 2_preprocessing.ipynb
- Description: preprocesses the Wikipedia train data
- Data sets: WikiLarge_Train.csv, Concreteness_ratings_Brysbaert_et_al_BRM.txt, AoA_51715_words.csv, dale_chall.txt, glove.6B.50d.txt
- Modules: feature_extractor.py

#### 3_train_classification_models.ipynb
- Description: trains the various classification models with different feature representations
- Data sets: wiki_train_preprocessed5.csv
- Modules: train_models.py

#### 4_deep_learning.ipynb
- Description: train the deep learning models
- Data sets: wiki_train_preprocessed5.csv
- Modules: feature_extractor.py

#### 5_topic_modeling.ipynb
- Description: create the unsupervised topics
- Data sets: wiki_train_preprocessed5.csv
- Modules: feature_extractor.py, viz_engine.py, unsupervised_engine.py
