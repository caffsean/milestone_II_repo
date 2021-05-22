import altair as alt
import matplotlib
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from plotly.offline import plot
import networkx as nx
import numpy as np
#matplotlib.rcParams['figure.dpi'] = 800
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class viz_engine():

    ## Histogrammer takes the dataframe and a column to create a comparative histogram of positive and negative classes. 
    
    def choices():
        return ['histogrammer',
               'word_embeddings_matrix',
               'word_embeddings_network']
    
    def histogrammer(df, column_to_compare, maxbins):
        '''
        params{
            df: a pandas dataframe
            column_to_compate = a column in the dataframe to be compared across classes
            maxbins = the max number of bins for the histogram
        }
        '''
        
        investigation_list = []

        df = df[['label',column_to_compare]]
        df0 = df[df['label']==0]
    
        df1 = df[df['label']==1]
    
        chart0 = alt.Chart(df0
              ).mark_area(opacity=0.5,
                          interpolate='step',
                          color='orange'
                          ).encode(
            alt.X(column_to_compare,bin=alt.Bin(maxbins=maxbins)),
            alt.Y('count()', stack=None))
        chart1 = alt.Chart(df1
              ).mark_area(opacity=0.3,
                          interpolate='step',
                          color='blue'
                         ).encode(
            alt.X(column_to_compare,bin=alt.Bin(maxbins=maxbins)),
            alt.Y('count()', stack=None))
        
        
        return chart0+chart1
    
    
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
            small_sims[x,x] = 0
            
        fig, ax = plt.subplots()
        im = ax.imshow(small_sims)

        ax.set_xticks(np.arange(len(small_vectors)))
        ax.set_yticks(np.arange(len(small_vectors)))

        ax.set_xticklabels(small_words)
        ax.set_yticklabels(small_words)
        ax.grid(False)

        plt.setp(ax.get_xticklabels(), rotation=90)

        return fig.tight_layout()
    
    
    def word_embeddings_network(model,max_words,min_similarity):
        
        ### Make a network of word embeddings
                '''
        params{
            model: W2V model
            max_words = max number of words to be in the network (selected based on highest frequency)
            min_similarity = Between 0 and 1, -remember smaller corpuses have higher similarity scores
    
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
                opacity=0.7,
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
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                line=dict(color='White'),
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
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