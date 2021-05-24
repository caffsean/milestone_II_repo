import altair as alt
import matplotlib
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from plotly.offline import plot
import networkx as nx
import pandas as pd
import numpy as np
#matplotlib.rcParams['figure.dpi'] = 800
import matplotlib.pyplot as plt
import squarify # pip install squarify
plt.style.use('ggplot')
import texthero as hero
from texthero import preprocessing
from texthero import stopwords
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class viz_engine():

    ## Histogrammer takes the dataframe and a column to create a comparative histogram of positive and negative classes. 
    
    def choices():
        return ['histogrammer',
               'float_histogrammer',
               'word_embeddings_matrix',
               'word_embeddings_network',
               'pos_ratio_barchart',
               'adar_staticTreemap',
               'adar_interactiveTreemap'
               'simple_wordcloud'
               ]
    
    def histogrammer(df,column_to_compare):
        
        if len(df) >= 5000:
            sample_size = 4999
            df = df.sample(sample_size)
        else:
            None
 
        chart = alt.Chart(df).mark_area(opacity=0.3).encode(
            x=alt.X(column_to_compare),
            y=alt.Y("count()", stack=None),
            color="label:N"
        )
        return chart

    def float_histogrammer(df,column_to_compare):
        
        if len(df) >= 5000:
            sample_size = 4999
            df = df.sample(sample_size)
        else:
            None
        
        df[column_to_compare] = np.round(df[column_to_compare],1)
    

  
        chart = alt.Chart(df).mark_area(opacity=0.3).encode(
            x=alt.X(column_to_compare),
            y=alt.Y("count()", stack=None),
            color="label:N"
        )
        return chart
    
    def word_embeddings_matrix(model, common_words):
        '''
        params{
            model: W2V model
            common_words: a list of words to be matrixed
        }
        '''
        ### Make a list of common words you'd like to compare
        
        words, vectors = [], []
        for item in common_words:
            try:
                vectors.append(model.wv.get_vector(item))
                words.append(item)
            except:
                None
            
        sims = cosine_similarity(vectors,vectors)
        indices = list(range(len(vectors)))
        
        small_vectors = [vectors[i] for i in indices]
        small_words = [words[i] for i in indices]
        
        small_sims = cosine_similarity(small_vectors, small_vectors)
        
        for x in range(len(small_vectors)):
            small_sims[x,x] = 1
            
        fig, ax = plt.subplots()
        im = ax.imshow(small_sims)

        ax.set_xticks(np.arange(len(small_vectors)))
        ax.set_yticks(np.arange(len(small_vectors)))

        ax.set_xticklabels(small_words)
        ax.set_yticklabels(small_words)
        ax.grid(False)

        plt.setp(ax.get_xticklabels(), rotation=90)

        return fig.tight_layout()
    
    
    
    
    
    def pos_ratio_barchart(df):
     
        df0 = df[df['label']==0]
        df1 = df[df['label']==1]
    
        dict_0 = {}
        dict_1 = {}
    
        dict_0['id'] = 'simple_wiki'
        dict_1['id'] = 'standard_wiki'
    
        pos = ['CC','CD','DT','IN','JJ','NN','NNP','NNS','PRP','RB','TO','VB','VBD','VBG','VBN','VBP','VBZ']
        counts = []
        pos_counts_0 = {}
        pos_counts_1 = {}

        for p in pos:
            total = sum(df0[p])
            pos_counts_0[p] = int(total)
    
        for p in pos:
            total = sum(df1[p])
            pos_counts_1[p] = int(total)
    
        df0_sum = sum(pos_counts_0.values())
        df1_sum = sum(pos_counts_1.values())   

    
        for p in pos:
            total = sum(df0[p])
            pos_counts_0[p] = int(total)/df0_sum
    
        for p in pos:
            total = sum(df1[p])
            pos_counts_1[p] = int(total)/df1_sum
    
    
    
        ndf0 = pd.DataFrame(pos_counts_0.values(),columns=['ratio'])
        ndf0['class'] = 0
        ndf0['pos'] = pos_counts_0.keys()
        
        ndf1 = pd.DataFrame(pos_counts_1.values(),columns=['ratio'])
        ndf1['class'] = 1
        ndf1['pos'] = pos_counts_1.keys()
   
        ndf = pd.concat([ndf0,ndf1])
    
        chart = alt.Chart(ndf).mark_bar(opacity=0.7).encode(
            x='class:N',
            y=alt.Y('ratio:Q', stack=True),
            color='class:N',
            column='pos:N')
    
    
    
    
        return chart




    def adar_staticTreemap(inputFrame):
    
        level_0_df = inputFrame[inputFrame['level']==0]
        level_1_df = inputFrame[inputFrame['level']==1]
        level_2_df = inputFrame[inputFrame['level']==2]
    
        level_0_chart = alt.Chart(level_0_df,width=800,height=800).mark_rect(color='black').encode(
            x=alt.X('x:Q',axis=alt.Axis(labels=False,ticks=False,title='')),
            x2='x2:Q',
            y=alt.Y('y:Q',axis=alt.Axis(labels=False,ticks=False,title='')),
            y2='y2:Q',)
        
    
        level_1_chart = alt.Chart(level_1_df,width=800,height=800).mark_rect(color='grey').encode(
            x='x:Q',
            x2='x2:Q',
            y='y:Q',
            y2='y2:Q',
            )
    
        level_2_chart = alt.Chart(level_2_df,width=800,height=800).mark_rect().encode(
            x='x:Q',
            x2='x2:Q',
            y='y:Q',
            y2='y2:Q',
            color=alt.Color('id:N',legend=None))
    
        text_1 = alt.Chart(level_1_df).mark_text(align='center', dx=55,size=15, dy=-45).encode(x='x:Q',y='y:Q',text='label:N')
        text_0 = alt.Chart(level_0_df).mark_text(
            align='center',
            color='white',
            dx=85,
            size=20, 
            dy=-15).encode(
            x='x:Q',y='y:Q',text='label:N')

    
        layered = alt.layer(level_0_chart,level_1_chart,level_2_chart,text_1,text_0,)
        return layered
    
    def adar_interactiveTreemap(inputFrame):
        selector = alt.selection_single(fields=['id'])

        level_0_df = inputFrame[inputFrame['level']==0]
        level_1_df = inputFrame[inputFrame['level']==1]
        level_2_df = inputFrame[inputFrame['level']==2]
    
    
        color1 = alt.condition(selector,
                       alt.Color('id:N',legend=None),
                       alt.value('lightgray'))
   
        level_0_chart = alt.Chart(level_0_df,width=800,height=800).mark_rect(color='black').encode(
            x=alt.X('x:Q',axis=alt.Axis(labels=False,ticks=False,title='')),
            x2='x2:Q',
            y=alt.Y('y:Q',axis=alt.Axis(labels=False,ticks=False,title='')),
            y2='y2:Q',)
        
    
        level_1_chart = alt.Chart(level_1_df,width=800,height=800).mark_rect(color='grey').encode(
            x='x:Q',
            x2='x2:Q',
            y='y:Q',
            y2='y2:Q',
            )
    
        level_2_chart = alt.Chart(level_2_df,width=800,height=800).mark_rect().encode(
            x='x:Q',
            x2='x2:Q',
            y='y:Q',
            y2='y2:Q',
            color=color1,
            tooltip=['parentid','id','value']).add_selection(selector)
  
    
        layered = alt.layer(level_0_chart,level_1_chart,level_2_chart)
        return layered
    
    
    
    def word_embeddings_network(model,max_words,min_similarity,mode):
        
        ### Make a network of word embeddings
        '''
        params{
            model: W2V model
            max_words = max number of words to be in the network (selected based on highest frequency)
            min_similarity = Between 0 and 1, -remember smaller corpuses have higher similarity scores
            mode = 'markers','text','markers+text'
        }
        '''
        
        common_words = list(model.wv.index2entity[:max_words])
        words, vectors = [], []
        for item in common_words:
            try:
                vectors.append(model.wv.get_vector(item))
                words.append(item)
            except:
                None
        sims = cosine_similarity(vectors, vectors)
    
        for i in range(len(vectors)):
            for j in range(len(vectors)):
                if i<=j:
                    sims[i, j] = False
        indices = np.argwhere(sims > min_similarity)     
    
        G = nx.Graph()

        for index in indices:
            G.add_edge(words[index[0]], words[index[1]], weight=sims[index[0],
                                                             index[1]])
    
        weight_values = nx.get_edge_attributes(G,'weight')
    
        positions = nx.spring_layout(G)
    
        nx.set_node_attributes(G,name='position',values=positions)
    
        searches = []
    
        edge_x = []
        edge_y = []
        weights = []
        ave_x, ave_y = [], []
        for edge in G.edges():
            x0, y0 = G.nodes[edge[0]]['position']
            x1, y1 = G.nodes[edge[1]]['position']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            ave_x.append(np.mean([x0, x1]))
            ave_y.append(np.mean([y0, y1]))
            weights.append(f'{edge[0]}, {edge[1]}: {weight_values[(edge[0], edge[1])]}')

        edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                opacity=0.4,
                line=dict(width=2, color='White'),
                hoverinfo='text',
                mode='lines')

        edge_trace.text = weights


        node_x = []
        node_y = []
        sizes = []
        for node in G.nodes():
            x, y = G.nodes[node]['position']
            node_x.append(x)
            node_y.append(y)
            if node in searches:
                sizes.append(50)
            else:
                sizes.append(15)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode=mode,
            hoverinfo='text',
            marker=dict(
                showscale=True,
                line=dict(color='White'),
                colorscale='Picnic',
                reversescale=False,
                color=[],
                opacity=0.9,
                size=sizes,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2
            )
        )

        invisible_similarity_trace = go.Scatter(
            x=ave_x, y=ave_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color=[],
                opacity=0,
            )
        )

        invisible_similarity_trace.text=weights
    
        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append(adjacencies[0])

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text
    
        fig = go.Figure(
            data=[edge_trace, node_trace, invisible_similarity_trace],
            layout=go.Layout(
                title='Network Graph of Word Embeddings',
                template='plotly_dark',
                titlefont_size=20,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[
                    dict(
                        text="",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) 
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
    
        return plot(fig, show_link = True, filename = 'result.html') ## Doesn't work in Chrome (Works in Firefox)
    
    def simple_wordcloud(df,background_color):
        custom_pipeline = [preprocessing.lowercase,
                           preprocessing.remove_punctuation]
        data = hero.clean(df['original_text'])
        default_stopwords = stopwords.DEFAULT
        custom_stopwords = default_stopwords.union(set(['lrb','rrb','also','first','one','s'])) ## Add as per requirement
        data = hero.remove_stopwords(data, custom_stopwords )

        return hero.visualization.wordcloud(data, font_path = None, width = 400, height = 200, max_words=500, 
                             mask=None, contour_width=0, 
                             contour_color='PAPAYAWHIP', background_color=background_color, 
                             relative_scaling='auto', colormap=None, return_figure=False)
    
    
    
class tools():
    
    def rectangleIter(data,width,height,xof=0,yof=0,frame=None,level=-1,parentid=""):
        
        # data: hierarchical structured data
        # width: width we can work in
        # height: height we can work in
        # xof: x offset
        # yof: y offset
        # frame: the dataframe to add the data to, if None, we create one
        # the level of the treemap (will default to 0 on first run)
        # parentid: a string representing the parent of this node
        # returns dataframe of all the rectangles
    
        if (frame is None):
            frame = pd.DataFrame()
        level = level + 1
        values = []
        children = []
        for parent in data:
            values.append(parent['value'])
            if ('children' in parent):
                children.append(parent['children'])
            else:
                children.append([])
            
        # normalize
        values = squarify.normalize_sizes(values, width, height)
   
        # generate the 
        padded_rects = squarify.padded_squarify(values, xof, yof, width, height)
    
        i = 0
        for rect in padded_rects:
            # adjust the padding and copy the useful pieces of data over
            parent = data[i]
            rect['width'] = rect['dx']
            rect['height'] = rect['dy']
            del rect['dx']
            del rect['dy']
            rect['x2'] = rect['x'] + rect['width'] - 2
            rect['y2'] = rect['y'] + rect['height'] - 2
            rect['x'] = rect['x'] + 2
            rect['y'] = rect['y'] + 2
            rect['width'] = rect['x2'] - rect['x']
            rect['height'] = rect['y2'] - rect['y']
            rect['id'] = parent['id']
            rect['value'] = parent['value']
            rect['level'] = level
            if 'label' in parent:
                rect['label'] = parent['label']
            else:
                rect['label'] = parent['id']
            rect['parentid'] = parentid
            frame = frame.append(rect,ignore_index=True)
        
            # iterate
            frame = rectangleIter(children[i],rect['width'],rect['height'],rect['x'],rect['y'],
                              frame=frame,level=level,parentid=parentid+" → "+rect['label'])
            i = i + 1
        return(frame)
    
    def tree_dict(df):
    
    
        df0 = df[df['label']==0]
        df1 = df[df['label']==1]
    
        dict_0 = {}
        dict_1 = {}
    
        dict_0['id'] = 'simple_wiki'
        dict_1['id'] = 'standard_wiki'
    
        pos = ['CC', 'CD', 'DT', 'EX', 'FW','IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT','POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD','VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
        counts = []
        pos_counts_0 = {}
        pos_counts_1 = {}
    
        
        other = ['CC', 'CD', 'DT', 'EX', 'FW','IN','LS', 'MD','PDT','POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'WDT', 'WP', 'WP$', 'WRB']
        nouns = ['NN', 'NNS', 'NNP', 'NNPS']
        adjectives = ['JJ', 'JJR', 'JJS']
        verbs = ['VB','VBD','VBG', 'VBN', 'VBP', 'VBZ']
    
        for p in pos:
            total = sum(df0[p])
            pos_counts_0[p] = int(total)+1
    
        for p in pos:
            total = sum(df1[p])
            pos_counts_1[p] = int(total)+1
        
        df0_sum = sum(pos_counts_0.values())
        df1_sum = sum(pos_counts_1.values())
    
        dict_0['value'] = df0_sum
        dict_1['value'] = df1_sum
    

        noun_count_0 = 1
        nouns_0 = []
        for n in nouns:
            nouns_sub0 = {}
            nouns_sub0['id'] = n
            nouns_sub0['value'] = pos_counts_0[n]
            noun_count_0 += pos_counts_0[n]
            nouns_0.append(nouns_sub0)
    
        noun_count_1 = 1
        nouns_1 = []
        for n in nouns:
            nouns_sub1 = {}
            nouns_sub1['id'] = n
            nouns_sub1['value'] = pos_counts_1[n]
            noun_count_1 += pos_counts_1[n]
            nouns_1.append(nouns_sub1)

        noun_dict0 = {}
        noun_dict0['id'] = 'NOUNS'
        noun_dict0['value'] = int(noun_count_0)
        noun_dict0['label'] = 'NOUNS'
        noun_dict0['children'] = nouns_0

        noun_dict1 = {}
        noun_dict1['id'] = 'NOUNS'
        noun_dict1['value'] = int(noun_count_1)
        noun_dict1['label'] = 'NOUNS'
        noun_dict1['children'] = nouns_1
    
        verb_count_0 = 1
        verb_0 = []
        for v in verbs:
            verb_sub0 = {}
            verb_sub0['id'] = v
            verb_sub0['value'] = pos_counts_0[v]
            verb_count_0 += pos_counts_0[v]
            verb_0.append(verb_sub0)
    
        verb_count_1 = 1
        verb_1 = []
        
        for v in verbs:
            verb_sub1 = {}
            verb_sub1['id'] = v
            verb_sub1['value'] = pos_counts_1[v]
            verb_count_1 += pos_counts_1[v]
            verb_1.append(verb_sub1)

        verb_dict0 = {}
        verb_dict0['id'] = 'VERBS'
        verb_dict0['value'] = int(verb_count_0)
        verb_dict0['label'] = 'VERBS'
        verb_dict0['children'] = verb_0

        verb_dict1 = {}
        verb_dict1['id'] = 'VERBS'
        verb_dict1['value'] = int(verb_count_1)
        verb_dict1['label'] = 'VERBS'
        verb_dict1['children'] = verb_1
    
    
        jj_count_0 = 1
        jj_0 = []
        for jj in adjectives:
            jj_sub0 = {}
            jj_sub0['id'] = jj
            jj_sub0['value'] = pos_counts_0[jj]
            jj_count_0 += pos_counts_0[jj]
            jj_0.append(jj_sub0)
    
        jj_count_1 = 1
        jj_1 = []
        for jj in adjectives:
            jj_sub1 = {}
            jj_sub1['id'] = jj
            jj_sub1['value'] = pos_counts_1[jj]
            jj_count_1 += pos_counts_1[jj]
            jj_1.append(jj_sub1)

        jj_dict0 = {}
        jj_dict0['id'] = 'ADJECTIVES'
        jj_dict0['value'] = int(jj_count_0)
        jj_dict0['label'] = 'ADJECTIVES'
        jj_dict0['children'] = jj_0

        jj_dict1 = {}
        jj_dict1['id'] = 'ADJECTIVES'
        jj_dict1['value'] = int(jj_count_1)
        jj_dict1['label'] = 'ADJECTIVES'
        jj_dict1['children'] = jj_1
    
        o_count_0 = 1
        o_0 = []
        for o in other:
            o_sub0 = {}
            o_sub0['id'] = o
            o_sub0['value'] = pos_counts_0[o]
            o_count_0 += pos_counts_0[o]
            o_0.append(o_sub0)
    
        o_count_1 = 1
        o_1 = []
        for o in other:
            o_sub1 = {}
            o_sub1['id'] = o
            o_sub1['value'] = pos_counts_1[o]
            o_count_1 += pos_counts_1[o]
            o_1.append(o_sub1)

        o_dict0 = {}
        o_dict0['id'] = 'OTHER'
        o_dict0['value'] = int(o_count_0)
        o_dict0['label'] = 'OTHER'
        o_dict0['children'] = o_0

        o_dict1 = {}
        o_dict1['id'] = 'OTHER'
        o_dict1['value'] = int(o_count_1)
        o_dict1['label'] = 'OTHER'
        o_dict1['children'] = o_1
    
    
    
        dict_0['children'] = [noun_dict0,verb_dict0,jj_dict0,o_dict0]
        dict_1['children'] = [noun_dict1,verb_dict1,jj_dict1,o_dict1]
        big_dict = [dict_0,dict_1]
    
        return big_dict
