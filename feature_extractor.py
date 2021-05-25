import pandas as pd
from tqdm import tqdm
import nltk
import ast
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import re
from spacy import displacy
import en_core_web_sm
nlp = spacy.load('en_core_web_sm')
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


class feature_mods():
    
    def choices():
        return ['length',
                'average_word_length'
                'tokens',
                'token_lemmas',
                'get_easy_words',
                'get_hard_word_ratio',
                'pos_tokens',
                'add_difficulty_columns',
                'add_pos_count_columns',
                'named_entity_counter',
                'add_punctuation',
                'create_embeddings',
                'overrepresented_pos_words',
                'overrepresented_neg_words'
               ]
    
    def length(df):
        '''
        input - pandas df w/ an 'original_text' column
        ## This counts the number of words in a given entry
        output - pandas df w/ an additional 'length' column
        '''
        
        df['length'] = df['original_text'].str.split().str.len()
        
        return df
    def average_word_length(df):
        '''
        input - Pandas df w/ an 'original_text' column
        
        ## This function includes stopwords when calculating the average length of words
        
        ''' 
        
        df['avg_word_length'] = df['original_text'].str.len() / df['length']
        
        return df
    
    def tokens(df): 
        '''
        input - Pandas df w/ an 'original_text' column
        ## This function adds a column of tokens   
        output - Pandas df w/ additional 'tokens' column
        
        '''
        df['tokens'] = df['original_text'].apply(lambda x: word_tokenize(x))
        
        return df
    

    def token_lemmas(df):
        '''
        input - Pandas df w/ an 'original_text' column
        
        ## This function adds a column of tokens 
           - tokenize
           - lemmatize
           - lower case
           - remove punctuation
                
        output - Pandas df w/ additional 'tokens' column
        
        '''
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        word_Lemmatized = WordNetLemmatizer()
        
        word_column = []
        b = df.original_text
        for entry in tqdm(b):
            word_list = []
            sent = nltk.pos_tag(nltk.word_tokenize(entry))
            for word, tag in sent:
                if word.isalpha():
                    word_list.append(word_Lemmatized.lemmatize(word,tag_map[tag[0]]).lower())
            if len(word_list) > 0: 
                word_column.append(word_list)
            else:
                word_column.append([entry])
        df['token_lemmas'] = word_column
        
        return df
    

    def get_easy_words(age):
        '''Find percentage of words that would not be considered more basic in a sentence'''
        
        file3 = "assets/Concreteness_ratings_Brysbaert_et_al_BRM.txt"
        file4 = "assets/AoA_51715_words.csv"
        file5 = 'assets/dale_chall.txt'
        concreteness = pd.read_csv(file3,delimiter = "\t")
        aoa = pd.read_csv(file4,encoding = "ISO-8859-1")
        dale_words_list = [word.strip() for word in open(file5).readlines()]
        
        #Lemmatize the dale_chall words so they can be matched with the lemmatized words in the cleaned data
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        word_Lemmatized = WordNetLemmatizer()

        # Get the pos tag for all the words
        tagged = pos_tag(dale_words_list)

        # Lemmatize, Remove Stopwords, Lowercase (stopword removal and lowercasing needs to come after pos tagging)
        dale_lemms = []
        for word, tag in tagged:
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]]).lower()
                dale_lemms.append(word_Final)
                
        
        #Returns a list of "easy" words determined by using the additional data sources

        # Get words that were acquired before a certain age
        aged_words = list(aoa[aoa['AoA_Kup_lem']<age].Lemma_highest_PoS)

        # Get words that at least 85% of people knew
        percentaged_words = list(concreteness.Word)

        # Put the lists together
        easy_words = set(aged_words + percentaged_words + dale_lemms)
        return easy_words
        
   
    def get_hard_word_ratio(df, age):
    
        easy_words = feature_mods.get_easy_words(age)
        # Find the count and percentage of words in the sentence that would not be considered easy
        df['hard_word_count'] = df['token_lemmas'].apply(lambda x: [word for word in x if word not in easy_words]).str.len()
        df['hard_word_ratio'] = df['hard_word_count']/df['token_lemmas'].str.len()

        return df
    
    
    def pos_tokens(df): 
        '''
        input - Pandas df w/ an 'original_text' column
        ## This function adds a column of Part of Speech tokens  
        '''     
        df['pos_tokens'] = df['original_text'].apply(lambda x: [i[1] for i in nltk.pos_tag(word_tokenize(x))])
        return df
    
    
    def add_difficulty_columns(df,age,percent_known):
        """
        params - dataframe
        age - average age of aquisition
        percent_known - percent of people who know word
        """
        
        file3 = "assets/Concreteness_ratings_Brysbaert_et_al_BRM.txt"
        file4 = "assets/AoA_51715_words.csv"
        df3 = pd.read_csv(file3,delimiter = "\t")
        df4 = pd.read_csv(file4,encoding = "ISO-8859-1")
    
    
        aged_words = df4[(df4['AoA_Kup_lem']>age)]
        percentaged_words = df3[df3['Percent_known']<percent_known]
    
        advanced_words = set(aged_words.Word)|set(percentaged_words.Word) 
        all_words = set(df4.Word) | set(df3.Word)
        
        df['advanced_words'] = df['tokens'].apply(lambda x: [word for word in x if word in advanced_words])
        df['advanced_word_count'] = df['advanced_words'].str.len()
        df['advanced_words_ratio'] = df['advanced_word_count']/df['tokens'].str.len()

        return df
    
    
    
    def named_entity_counter(df):
        
        df['named_entities'] = df['original_text'].apply(lambda x: nlp(x).ents)
        df['named_entity_count'] = df['named_entities'].str.len()
        df['named_entity_ratio'] = df['named_entity_count']/ df['length']
        
        return df
    
    def add_punctuation(df):
        
        df['punctuation_score'] = df['original_text'].apply(lambda x: len(re.findall('[!?\\-,:;]',x)))
        df['punctuation_ratio'] = df['punctuation_score'] / df['original_text'].str.len()

        return df
    
   
    def add_pos_count_columns(df):

        parts_of_speech = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]
    
        for pos in tqdm(parts_of_speech):
            pos_count_list = []
            pos_ratio_list = []
            for entry in df.pos_tokens:
                counter = 0
                for tag in entry:
                    if tag == pos:
                        counter += 1
                    else:
                        None
                pos_count_list.append(counter)
                pos_ratio_list.append(counter/len(entry))
            df[pos] = pos_ratio_list
            df['{}_count'.format(pos)] = pos_count_list
            
            
        return df
    
    def create_embeddings(df):
        '''Get word embeddings for each sentence'''
        # Code used from https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
        embeddings = {}
        with open('assets/GLOVE/glove.6B.50d.txt', 'r') as file:
            for line in file:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], 'float32')
                embeddings[word] = vector
                
        # Create an embedding vector for each sentence with is an average of all words in the sentence
        # Code adapted from Practical Natural Language Processing A Comprehensive Guide to 
        # Building Real-World NLP Systems by Sowmya Vajjala, Bodhisattwa Majumder, 
        # Anuj Gupta and Harshit Surana

        list_of_sentences = list(df['tokens'])
        dimension = 50
        features = []
        for sent in list_of_sentences:
            feat_for_sent = np.zeros(dimension)
            count_of_words = 0
            for word in sent:
                count_of_words += 1
                if word in embeddings:
                    feat_for_sent += embeddings[word]
            features.append(feat_for_sent/count_of_words)
        return pd.DataFrame(features)
    
        
    def overrepresented_pos_words(df, listy):
        
        words = set(listy)
        
        new_column_overrep = []
        new_column_ratio = []

        for index in tqdm(range(len(df))):
            entry = df.preprocessed_tokens.iloc[index]
            overrep = []
            #entry = ast.literal_eval(entry)
            for word in entry:
                if word in words:
                    overrep.append(word)
                else:
                    None
            b = len(overrep)
            new_column_overrep.append(overrep)
            if len(entry)>0:
                new_column_ratio.append(b/len(entry))
            else:
                new_column_ratio.append(0)
                
        df['pos_overrep_ratio'] = new_column_ratio
        
        x = df.pos_overrep_ratio.values.reshape(-1, 1) 
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df['pos_overrep_ratio_norm'] = x_scaled
        
        return df
    

    
    def overrepresented_neg_words(df, listy):
        
        words = set(listy)
        
        new_column_overrep = []
        new_column_ratio = []

        for index in tqdm(range(len(df))):
            entry = df.preprocessed_tokens.iloc[index]
            overrep = []
            #entry = ast.literal_eval(entry)
            for word in entry:
                if word in words:
                    overrep.append(word)
                else:
                    None
            b = len(overrep)
            new_column_overrep.append(overrep)
            if len(entry)>0:
                new_column_ratio.append(b/len(entry))
                
            else:
                new_column_ratio.append(0)
                
        df['neg_overrep_ratio'] = new_column_ratio
        
        x = df.neg_overrep_ratio.values.reshape(-1, 1) 
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df['neg_overrep_ratio_norm'] = x_scaled
        
        return df
    
class tools():
    def get_numeric_columns_list(df):
    
        numeric_columns = []
    
    
        for column in list(df.columns):
            if dict(df.dtypes)[column] == 'int64':
                numeric_columns.append(column)
            elif dict(df.dtypes)[column] == 'float64':
                numeric_columns.append(column)
            else:
                None
        numeric_columns.remove('label')
        return numeric_columns
    