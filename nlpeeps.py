import pandas as pd
from tqdm import tqdm
import nltk
import ast
import re
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from spacy import displacy
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import en_core_web_sm
nlp = spacy.load('en_core_web_sm')
from sklearn.decomposition import PCA
import altair as alt

class feature_engine():
    

    
    ## SEE CHOICES 
    
    def choices():
        b = ["preprocessed_tokens",
                "tokens",
                "pos_tokens",
                'dep_tokens',
                "length",
                "average_word_length_with_stopwords",
                "average_word_length_no_stopwords",
                "std_word_length_with_stopwords",
                "std_word_length_no_stopwords",
                "pos_ratios",
                'dep_ratios',
                'advanced_words',
                'easy_words',
                'advanced_to_easy_ratio',
                'named_entity_counter',
                'punctuation',
                'odd_characters',
                'relative_pos',
                'overrepresented_pos_words',
                'overrepresented_neg_words']
        return b 
    
    
    ### PREPROCESSED TOKENS
    
    
    def preprocessed_tokens(df):
        '''
        input - pandas df w/ an 'original_text' column
        
        ## This function adds a column of preprocessed tokens:
                -lowercase
                -removes punctuation
                -removes stopwords
                
        output - pandas df w/ additional 'preprocessed_tokens' column
        '''
        sentences = list()
        lines = df['original_text'].values.tolist()
        for line in tqdm(lines):
            tokens = word_tokenize(line)
            tokens = [word.lower() for word in tokens]
            table = str.maketrans('','',string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            words = [word for word in stripped if word.isalpha()]
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if not w in stop_words]
            sentences.append(words)     
        df['preprocessed_tokens'] = sentences
        
        return df

    
    ## LENGTH FEATURES
        
    
    def length(df):
        '''
        input - pandas df w/ an 'original_text' column
        
        ## This counts the number of words in a given entry
        
        output - pandas df w/ an additional 'length' column
        '''
        length_column = []
        b = df.original_text
        for entry in tqdm(b):
            sent = len(nltk.word_tokenize(entry))
            length_column.append(sent)
        df['length'] = length_column
        
        ## NORMALIZED
        
        x = df.length.values.reshape(-1, 1) 
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df['length_norm'] = x_scaled
        
        return df
    
    
    ## TOKENS

    def tokens(df): 
        '''
        input - Pandas df w/ an 'original_text' column
        
        ## This function adds a column of tokens with NO preprocessing
                
        output - Pandas df w/ additional 'tokens' column
        '''
        word_column = []
        b = df.original_text
        for entry in tqdm(b):
            word_list = []
            sent = nltk.word_tokenize(entry)
            for word in sent:
                word_list.append(word)
            word_column.append(word_list)       
        df['tokens'] = word_column
        
        return df
    
    
    ## PART OF SPEECH TOKENS
    
    def pos_tokens(df): 
        '''
        input - Pandas df w/ an 'original_text' column
        
        ## This function adds a column of Part of Speech tokens
                
        '''      
        pos_column = []
        b = df.original_text
        for entry in tqdm(b):
            pos_list = []
            word_list = []
            sent = nltk.word_tokenize(entry)
            sent = nltk.pos_tag(sent)
            for word in sent:
                pos = word[1]
                pos_list.append(pos)
            pos_column.append(str(pos_list))  
        df['pos_tokens'] = pos_column
        
        return df
    

    
    ## AVERAGE WORD LENGTH (including stopwords)
    
    def average_word_length_with_stopwords(df):
        '''
        input - Pandas df w/ an 'original_text' column
        
        ## This function includes stopwords when calculating the average length of words
        
        ''' 
    
        avg_word_length_column = []
        
        b = df.original_text
        for entry in tqdm(b):
            word_length_list = []
            sent = nltk.word_tokenize(entry)
            if len(sent) <= 1:
                avg_word_length_column.append(0)
            else:
                for word in sent:
                    length = len(word)
                    word_length_list.append(length)
                if len(word_length_list) > 0:
                    average = np.mean(word_length_list)
                    avg_word_length_column.append(average)
                else:
                    average = 0
                    avg_word_length_column.append(average)       
        df['average_word_length_with_stopwords'] = avg_word_length_column
        
 
        return df
    
    
    def std_word_length_with_stopwords(df):
        '''
        input - Pandas df w/ an 'original_text' column
        
        ## This function includes stopwords when calculating the average length of words
        
        '''
    
        avg_word_length_column = []
        b = df.original_text
        for entry in tqdm(b):
            word_length_list = []
            sent = nltk.word_tokenize(entry)
            for word in sent:
                length = len(word)
                word_length_list.append(length)
            if len(word_length_list) > 0:
                average = np.std(word_length_list)
                avg_word_length_column.append(average)
            else:
                average = 0
                avg_word_length_column.append(average)       
        
        df['std_word_length_with_stopwords'] = avg_word_length_column
        

        return df
    
    ## AVERAGE WORD LENGTH (NO stopwords)
        
    def average_word_length_no_stopwords(df):
        
        '''
        input - Pandas df w/ an 'original_text' column
        
        ## This function DOES NOT include stopwords when calculating the average length of words
        
        '''
        avg_word_length_column = []
        
        sentences = list()
        lines = df['original_text'].values.tolist()

        for line in tqdm(lines):
            tokens = word_tokenize(line)
            tokens = [word.lower() for word in tokens]
            table = str.maketrans('','',string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            words = [word for word in stripped if word.isalpha()]
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if not w in stop_words]
            sentences.append(words)
        
        for entry in tqdm(sentences):
            word_length_list = []
            if len(entry) <= 1:
                avg_word_length_column.append(0)
            else:
                for word in entry:
                    length = len(word)
                    word_length_list.append(length)
                if len(word_length_list) > 0:
                    average = np.mean(word_length_list)
                    avg_word_length_column.append(average)
                else:
                    average = 0
                    avg_word_length_column.append(average)
        
        df['average_word_length_no_stopwords'] = avg_word_length_column

        
        return df
    
    def std_word_length_no_stopwords(df):
        
        '''
        input - Pandas df w/ an 'original_text' column
        
        ## This function DOES NOT include stopwords when calculating the average length of words
        
        '''
        avg_word_length_column = []
        
        sentences = list()
        lines = df['original_text'].values.tolist()

        for line in tqdm(lines):
            tokens = word_tokenize(line)
            tokens = [word.lower() for word in tokens]
            table = str.maketrans('','',string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            words = [word for word in stripped if word.isalpha()]
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if not w in stop_words]
            sentences.append(words)
        
        for entry in tqdm(sentences):
            word_length_list = []
            for word in entry:
                length = len(word)
                word_length_list.append(length)
            if len(word_length_list) > 0:
                average = np.std(word_length_list)
                avg_word_length_column.append(average)
            else:
                average = 0
                avg_word_length_column.append(average)
        
        df['std_word_length_no_stopwords'] = avg_word_length_column
        
        return df
    
    ### ADVANCED WORDS
    
    def advanced_words(df):
        
        
        '''
        This function adds a list of advanced words
        
        '''
        age = 8
        
        percent_known = .93 
        
        file3 = "assets/Concreteness_ratings_Brysbaert_et_al_BRM.txt"
        file4 = "assets/AoA_51715_words.csv"
        df3 = pd.read_csv(file3,delimiter = "\t")
        df4 = pd.read_csv(file4,encoding = "ISO-8859-1")
    
    
        difficult_age_words = df4[(df4['AoA_Kup_lem']>age)]
        difficult_percentage_words = df3[df3['Percent_known']<percent_known]
            
        advanced_words = set(difficult_age_words.Word)|set(difficult_percentage_words.Word) 
        
        all_words = set(df4.Word) | set(df3.Word)
   
        difficult_new_column = []
        difficult_new_column_count = []
        difficult_new_column_ratio = []
        

        for index in tqdm(range(len(df))):
            entry = df.preprocessed_tokens.iloc[index]
            advanced = []
            #entry = ast.literal_eval(entry)
            for word in entry:
                if word in advanced_words:
                    advanced.append(word)
                else:
                    None
            b = len(advanced)
            difficult_new_column.append(advanced)
            difficult_new_column_count.append(b)
            if len(entry)>0:
                difficult_new_column_ratio.append(b/len(entry))
            else:
                difficult_new_column_ratio.append(0)
                
        df['advanced_words'] = difficult_new_column
        df['advanced_words_count'] = difficult_new_column_count
        df['advanced_words_ratio'] = difficult_new_column_ratio

        x = df.advanced_words_count.values.reshape(-1, 1) 
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df['advanced_words_count_norm'] = x_scaled
        
        x = df.advanced_words_ratio.values.reshape(-1, 1) 
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df['advanced_words_ratio_norm'] = x_scaled
        
        difficult_word_avg_lengths = []
        
        for advanced in df['advanced_words']:
            if len(advanced) > 0:
                avg_len = []
                for word in advanced: 
                    b = len(word)
                    avg_len.append(b)
                avg_len = np.mean(avg_len)
            else:
                avg_len = 0
            difficult_word_avg_lengths.append(avg_len)
        df['avg_advanced_word_length'] = difficult_word_avg_lengths
        
 
        
        difficult_word_std_lengths = []
        
        for advanced in df['advanced_words']:
            if len(advanced) > 0:
                avg_len = []
                for word in advanced: 
                    b = len(word)
                    avg_len.append(b)
                avg_len = np.mean(avg_len)
            else:
                avg_len = 0
            difficult_word_std_lengths.append(avg_len)
        df['std_advanced_word_length'] = difficult_word_std_lengths
        
        x = df.std_advanced_word_length.values.reshape(-1, 1) 
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df['std_advanced_word_length_norm'] = x_scaled
                      
        
        return df
    
    
    
    
    
    
    
    ## EASY WORDS 
    
    def easy_words(df):
        """
        params - dataframe
        age - average age of aquisition
        percent_known - percent of people who know word
        """
        
        age = 8
        
        percent_known = .93 
        
        file3 = "assets/Concreteness_ratings_Brysbaert_et_al_BRM.txt"
        file4 = "assets/AoA_51715_words.csv"
        df3 = pd.read_csv(file3,delimiter = "\t")
        df4 = pd.read_csv(file4,encoding = "ISO-8859-1")
     
        easy_age_words = df4[(df4['AoA_Kup_lem']<age)]
        easy_percentage_words = df3[df3['Percent_known']>percent_known]
        easy_words = set(easy_age_words.Word)|set(easy_percentage_words.Word) 

        easy_new_column = []
        easy_new_column_count = []
        easy_new_column_ratio = []
                
        for index in tqdm(range(len(df))):
            entry = df.preprocessed_tokens.iloc[index]
            easy = []
            #entry = ast.literal_eval(entry)
            for word in entry:
                if word in easy_words:
                    easy.append(word)
                else:
                    None
            b = len(easy)
            easy_new_column.append(easy)
            easy_new_column_count.append(b)
            if len(entry)>0:
                easy_new_column_ratio.append(b/len(entry))
            else:
                easy_new_column_ratio.append(0)

        
        df['easy_words'] = easy_new_column
        df['easy_words_count'] = easy_new_column_count
        df['easy_words_ratio'] = easy_new_column_ratio

        x = df.easy_words_count.values.reshape(-1, 1) 
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df['easy_words_count_norm'] = x_scaled

        x = df.easy_words_ratio.values.reshape(-1, 1) 
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df['easy_words_ratio_norm'] = x_scaled
        
        
        easy_word_avg_lengths = []
        
        for easy in df['easy_words']:
            if len(easy) > 0:
                avg_len = []
                for word in easy: 
                    b = len(word)
                    avg_len.append(b)
                avg_len = np.mean(avg_len)
            else:
                avg_len = 0
            easy_word_avg_lengths.append(avg_len)
        df['avg_easy_word_length'] = easy_word_avg_lengths
        
        x = df.avg_easy_word_length.values.reshape(-1, 1) 
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df['avg_easy_word_length_norm'] = x_scaled
        
        easy_word_std_lengths = []
        
        for advanced in df['easy_words']:
            if len(advanced) > 0:
                avg_len = []
                for word in advanced: 
                    b = len(word)
                    avg_len.append(b)
                avg_len = np.std(avg_len)
            else:
                avg_len = 0
            easy_word_std_lengths.append(avg_len)
        df['std_easy_word_length'] = easy_word_std_lengths
        

        return df
    
    def advanced_to_easy_ratio(df):
        
        df['advanced_to_easy_ratio'] = df['easy_words_count']/df['advanced_words_count']
        df['advanced_to_easy_ratio'].replace([np.inf, -np.inf], 0, inplace=True)
        df['advanced_to_easy_ratio'].replace(np.nan, 0, inplace=True)

        return df
    
    
    
    
    
    
    ## OVERREPRESENTED WORDS IN POSITIVE CLASS
    
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
    
    
    
    
    ## OVERREPRESENTED WORDS IN NEGATIVE CLASS
    
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
    
    

    
    ## NAMED ENTITY COUNTER 
    
    def named_entity_counter(df):
        
        named_entity_list = []
        named_entity_count = []
        named_entity_ratio = []
        
        ## To speed this up, exclude short entries
        
        for entry in tqdm(df.original_text):
            tokens = word_tokenize(entry)
            if len(tokens) > 1:
                doc = nlp(entry)
                named_entities = doc.ents
                entity_count = len(named_entities)
                named_entity_list.append(named_entities)
                named_entity_count.append(entity_count)
                named_entity_ratio.append(entity_count/len(entry))
            else:
                named_entity_list.append([])
                named_entity_count.append(0)
                named_entity_ratio.append(0)
            
        df['named_entites'] = named_entity_list
        df['named_entity_count'] = named_entity_count
        df['named_entity_ratio'] = named_entity_ratio
         
        x = df.named_entity_ratio.values.reshape(-1, 1) 
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df['named_entity_ratio_norm'] = x_scaled
        
        return df
    

    
    ### PUNCTUATION SCORES
    
    def punctuation(df):
        
        punctuation_score = []
        punctuation_ratio = []
        
        for index in range(len(df)):
            entry = df.original_text.iloc[index]
            B = re.findall('[!?\\-,:;&]',entry)
            D = re.findall('(RRB)',entry)
            E = re.findall('(LRB)',entry)
            F = re.findall('(ndash)',entry)
            
            C = len(B) + len(D) + len(E) + len(F)
            punctuation_score.append(C)
            punctuation_ratio.append(C/len(entry))
            
        df['punctuation_count']=punctuation_score
        df['punctuation_ratio']=punctuation_ratio
        
        x = df.punctuation_count.values.reshape(-1, 1) 
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df['punctuation_count_norm'] = x_scaled 
        
        x = df.punctuation_ratio.values.reshape(-1, 1) 
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df['punctuation_ratio_norm'] = x_scaled        
        
        return df
    
    
    
    
    
    ### ODD CHARACTER SCORES
    
    def odd_characters(df):
        
        odd_characters_score = []
        odd_characters_ratio = []
        
        for index in range(len(df)):
            entry = df.original_text.iloc[index]
            entre = str(entry)
            B = re.findall('[ÅÂÀÁÄÜÃØÎÊÉÈÕÛÔÒÓåâàáäüãøîêéèõûôòó]',entry)
            C = len(B)
            odd_characters_score.append(C)
            odd_characters_ratio.append(C/len(entry))
            
        df['odd_characters_count']=odd_characters_score
        df['odd_characters_ratio']=odd_characters_ratio
        
        x = df.odd_characters_count.values.reshape(-1, 1) 
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df['odd_characters_count_norm'] = x_scaled 
        
        
        x = df.odd_characters_ratio.values.reshape(-1, 1) 
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df['odd_characters_ratio_norm'] = x_scaled        
        
        return df
    
   


    ### IN-TEXT RATIOS of PARTS OF SPEECH 
   

    def pos_ratios(df):
        pos_column = []
        for entry in tqdm(df.original_text):
            pos_list = []
            sent = nltk.word_tokenize(entry)
            sent = nltk.pos_tag(sent)
            for word in sent:
                pos = word[1]
                pos_list.append(pos)
            pos_column.append(pos_list)
            
        df['parts_of_speech'] = pos_column

        parts_of_speech = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]
    
        for pos in tqdm(parts_of_speech):
            pos_count_list = []
            pos_ratio_list = []
            for entry in df.parts_of_speech:
                if len(entry) > 1:
                    counter = 0
                    for tag in entry:
                        if tag == pos:
                            counter += 1
                        else:
                            None
                    pos_count_list.append(counter)
                    pos_ratio_list.append(counter/len(entry))
                else:
                    pos_count_list.append(0)
                    pos_ratio_list.append(0)
            
            df[pos] = pos_count_list
            df["{}_ratio".format(pos)] = pos_ratio_list

            x = df[pos].values.reshape(-1, 1) 
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df['{}_count_norm'.format(pos)] = x_scaled
            
            x = df['{}_ratio'.format(pos)].values.reshape(-1, 1) 
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df['{}_ratio_norm'.format(pos)] = x_scaled
        
        
            
        return df

    
    ### ALL PARTS OF SPEECH RELATIVE TO EACH OTHER 
    
    def relative_pos(df):
        

        parts_of_speech = ["JJ", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

        for pos_n in tqdm(parts_of_speech):
            for pos_d in parts_of_speech:
                df['{}_over_{}'.format(pos_n,pos_d)] = df[pos_n]/df[pos_d]
                df['{}_over_{}'.format(pos_n,pos_d)].replace([np.inf, -np.inf], 0, inplace=True)
                df['{}_over_{}'.format(pos_n,pos_d)].replace(np.nan, 0, inplace=True)
                x = df['{}_over_{}'.format(pos_n,pos_d)].values.reshape(-1, 1) 
                min_max_scaler = preprocessing.MinMaxScaler()
                x_scaled = min_max_scaler.fit_transform(x)
                df['{}_over_{}_normalized'.format(pos_n,pos_d)] = x_scaled

        
        return df



    
### TOOLS ####
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
    
    
