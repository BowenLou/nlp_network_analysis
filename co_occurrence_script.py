# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import nltk
import networkx as nx
import matplotlib.pyplot as plt
import os
import time
import json
import jsonrpclib
import string
import argparse
import getopt
import sys
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import io

"""
    This script will process a bunch of files in the input directory, and generate co-occurrence graph of the corresponding file and finally build a combined graph of all these files. Also an analysis file is generated.
    
    usage: co_occurrence_script.py [-h] [-t TOP_NUMBER] [-w WINDOW_SIZE]
    [-f OCCURRENCE_FREQUENCY] [-s] [-v]
    dir_path
    
    positional arguments:
    dir_path              input directory path
    
    optional arguments:
    -h, --help            show this help message and exit
    -t TOP_NUMBER, --top_number TOP_NUMBER
                          input integer defining <TOP_NUMBER> top-scoring degree centrality, clustering and clique analysis. default is 10
    -w WINDOW_SIZE, --window_size WINDOW_SIZE
                          input integer defining co-occurrence window size, i.e, how far in both directions to look for co-occurrences to each target word. default is 1
    -f OCCURRENCE_FREQUENCY, --occurrence_frequency OCCURRENCE_FREQUENCY
                          input integer defining the minimum, collection-wide frequency a word must occur with to be included in the final combined graph. default is 1
    -s, --no_boundary     not to consider sentence boundary when finding co-occurrence word pair
    -v, --no_lemma        not to lemmatize words in the file
    
    input:
    1. The path to a directory of text files to be analyzed individually and as a collection.
    2. An integer, defining the number of top-scoring nodes (or edges) for which to report various analysis (for example, the top 10 most central nodes in the graph). Default is 10.
    3. An integer, defining the co-occurrence window; how far in both directions to look for co-occurrences. Default is 1.
    4. An integer, defining the minimum, collection-wide frequency a word must occur with to be included in the final combined graph. This should not effect individual files' graphs. Default is 1.
    5. A Boolean, defining whether or not sentence boundaries should be respected (ie. to include co-occurrences across sentences or not). Default is to consider the sentence boundary.
    6. A Boolean, defining whether or not to lemmatize all words in the file before building the graphs. Default is to do the lemmatization.
    
    output:
    1. A .pajek graph file for each file in the input directory.
    2. A .pajek graph file for the combined graph (summing node and edge frequencies in the way we set, and presenting based on the integer defining the minimum, collection-wide frequency a word must occur with)
    3. A CSV reporting the top <TOP_NUMBER> highest-scoring elements (nodes or edges) for the essential analyses
"""

__author__ = "Bowen Lou, Knowledge Lab | Computation Institute"
__version__ = "v0.1"

class StandfordNLP:
    """
        The StandfordNLP class is referenced from client.py in corenlp-python
        """
    def __init__(self, port_number = 8080):
        self.server = jsonrpclib.Server("http://localhost:%d" % port_number)
    
    def parse(self, text):
        return json.loads(self.server.parse(text))


def preprocessSent(file_content, bound_flag, lemma_flag):
    """
        input:
            content in the file to be processed
        return:
            list of meaningful words in the file
    """
    from nltk.corpus import stopwords
    import string
    
    sent_seg = nltk.sent_tokenize(file_content)
    # omit some special character which can not decoded in ascii and omit punctuation
    sent_seg = [sent.encode('ascii', 'ignore').translate(None, string.punctuation) for sent in sent_seg]
    sent_token = [nltk.word_tokenize(sent.strip()) for sent in sent_seg]
    stop_word = stopwords.words('english')
    
    nlp = StandfordNLP()
    list_word_preprocess = []
    
    # print sent_seg
    if lemma_flag:
        for sent in sent_seg:
            # print 'sent:', sent
            # print 'no:', sent.strip()
            # print len(sent.strip())
            if len(sent.strip()) != 0:
                for word in nlp.parse(sent)['sentences'][0]['words']:
                    word_lemma = str(word[1]['Lemma'])
                    # remove stop words and make the letter in lower case
                    if ((not word_lemma.lower() in stop_word) and len(word_lemma) > 2):
                        list_word_preprocess.append(word_lemma.lower())
                if bound_flag:
                    list_word_preprocess.append('*') # a sign to mark the boundary.
    else:
        for sent in sent_token:
            if len(sent) != 0:
                for word in sent:
                    if ((not word.lower() in stop_word) and len(word) > 2):
                        list_word_preprocess.append(word.lower())
                if bound_flag:
                    list_word_preprocess.append('*') # a sign to mark the boundary.

    # print list_word_preprocess
    return list_word_preprocess

def buildCOMatrix(list_word_no_stopword, window_size, bound_flag):
    """
        Build the co-occurrence word-pair matrix of each file in the directory
    """
    if bound_flag:
        list_bound_index = [bound_index for bound_index, word in enumerate(list_word_no_stopword) if word == '*']
        # print list_bound_index
        max_sent_len = max(list_bound_index[i] - list_bound_index[i - 1] - 1 for i in range(1, len(list_bound_index)))
        # print max_sent_len

        if window_size > max_sent_len - 1:
            window_size = max_sent_len - 1
            print 'adjust the window size to ', window_size
                
        list_word_unique = set([word for word in list_word_no_stopword if word != '*'])
        # print list_word_unique
        
        df_init = DataFrame(np.zeros((len(list_word_unique), len(list_word_unique))),
                            index = list_word_unique,
                            columns = list_word_unique)
            
        df_process = df_init
        
        for ix, word in enumerate(list_word_no_stopword):
            if word != '*':
                for i in range(1, window_size + 1):
                    # backward direction
                    if ix - i >= 0:
                        if list_word_no_stopword[ix - i] == '*':
                            break
                        else:
                            df_process.ix[word, list_word_no_stopword[ix - i]] += 1
            
                for i in range(1, window_size + 1):
                    # forward direction
                    if (ix + i <= len(list_word_no_stopword) - 1):
                        if list_word_no_stopword[ix + i] == '*':
                            break
                        else:
                            df_process.ix[word, list_word_no_stopword[ix + i]] += 1
    else:
        if window_size > len(list_word_no_stopword) - 1:
            window_size = len(list_word_no_stopword) - 1
            print 'adjust the window size to ', window_size

        list_word_unique = set(list_word_no_stopword)
        # print list_word_unique

        df_init = DataFrame(np.zeros((len(list_word_unique), len(list_word_unique))),
                            index = list_word_unique,
                            columns = list_word_unique)
        df_process = df_init

        for ix, word in enumerate(list_word_no_stopword):
            for i in range(1, window_size + 1):
                # backward direction
                if ix - i >= 0:
                    df_process.ix[word, list_word_no_stopword[ix - i]] += 1
                # forward direction
                if ix + i <= len(list_word_no_stopword) - 1:
                    df_process.ix[word, list_word_no_stopword[ix + i]] += 1
        
    # print df_process
    return df_process


def countWordFreq(list_word):
    """
        input:
            list of words in the file
        return:
            a dictionary in which the key is the word and the value is the frequency of the word in the input
    """
    word_freq_count = dict((word, list_word.count(word)) for word in list_word)
    return word_freq_count


def drawCOGraph(list_word_no_stopword, window_size, bound_flag, lemma_flag, dir_path, file_name):
    """
        Draw and save the co-occurrence graph of each file in the directory
    """
    co_occurrence_graph = nx.Graph()
    co_occurrence_matrix = buildCOMatrix(list_word_no_stopword, window_size, bound_flag)

    if bound_flag:
        list_word = [word for word in list_word_no_stopword if word != '*']
        list_word_unique = set(list_word)
        word_freq_dict = countWordFreq(list_word)

    else:
        list_word_unique = set(list_word_no_stopword)
        word_freq_dict = countWordFreq(list_word_no_stopword)

    # print word_freq_dict

    for word_1 in list_word_unique:
        for word_2 in list_word_unique:
            if co_occurrence_matrix.ix[word_1, word_2] >= 1:
                if word_1 not in co_occurrence_graph.nodes():
                    co_occurrence_graph.add_node(word_1, freq = word_freq_dict[word_1])
                if word_2 not in co_occurrence_graph.nodes():
                    co_occurrence_graph.add_node(word_2, freq = word_freq_dict[word_2])
                edge_freq = co_occurrence_matrix.ix[word_1, word_2]
                co_occurrence_graph.add_edge(word_1, word_2, freq = int(edge_freq))

    # write to a .pajek graph file for each file in the input directory.

    if lemma_flag and bound_flag:
        if not os.path.exists(dir_path + 'co_lemma_bound/'):
            os.makedirs(dir_path + 'co_lemma_bound/')
        nx.write_pajek(co_occurrence_graph, dir_path + 'co_lemma_bound/' + file_name + '_co_occurrence_lemma_bound.net')
    elif lemma_flag and not bound_flag:
        if not os.path.exists(dir_path + 'co_lemma_no_bound/'):
            os.makedirs(dir_path + 'co_lemma_no_bound/')
        nx.write_pajek(co_occurrence_graph, dir_path + 'co_lemma_no_bound/' + file_name + '_co_occurrence_lemma_no_bound.net')
    elif not lemma_flag and bound_flag:
        if not os.path.exists(dir_path + 'co_bound_no_lemma/'):
            os.makedirs(dir_path + 'co_bound_no_lemma/')
        nx.write_pajek(co_occurrence_graph, dir_path + 'co_bound_no_lemma/' + file_name + '_co_occurrence_bound_no_lemma.net')
    else:
        if not os.path.exists(dir_path + 'co_no_lemma_no_bound/'):
            os.makedirs(dir_path + 'co_no_lemma_no_bound/')
        nx.write_pajek(co_occurrence_graph, dir_path + 'co_no_lemma_no_bound/' + file_name + '_co_occurrence_no_lemma_no_bound.net')
    return co_occurrence_graph

def graphAnalysis(graph, top_number, save_file_path):
    """
        Do the essential analysis to the final combined graph
    """
    with io.open(save_file_path, 'w') as save_file:

        # centrality
        # degree centrality
        deg_central = nx.degree_centrality(graph)
        deg_central_sort = sorted(deg_central.items(), key = lambda x: x[1], reverse = True)
        top_deg_central_sort = deg_central_sort[:top_number]
        save_file.write('top %d degree centrality items,' % top_number)
        save_file.write(','.join('%s %s' % x for x in top_deg_central_sort))

        # clustering

        # number of triangles: triangles() is not defined for directed graphs
        triangle_num = nx.triangles(graph)
        triangle_num_sort = sorted(triangle_num.items(), key = lambda x: x[1], reverse = True)
        top_triangle_num_sort = triangle_num_sort[:top_number]
        save_file.write('\ntop %d number of triangles including a node as one vertex,' % top_number)
        save_file.write(','.join('%s %s' % x for x in top_triangle_num_sort))

        # clustering coefficient of node in the graph
        cluster_coefficient = nx.clustering(graph)
        cluster_coefficient_sort = sorted(cluster_coefficient.items(), key = lambda x: x[1], reverse = True)
        top_cluster_coefficient_sort = cluster_coefficient_sort[:top_number]
        save_file.write('\ntop %d clustering coefficient items,' % top_number)
        save_file.write(','.join('%s %s' % x for x in top_cluster_coefficient_sort))

        # transitivity of the graph
        triangle_transitivity = nx.transitivity(graph)
        save_file.write('\ntransitivity of the graph,%f' % triangle_transitivity)

        # average clustering coefficient of the graph
        avg_cluster = nx.average_clustering(graph)
        save_file.write('\naverage clustering coefficient of the graph,%f' % avg_cluster)

        # clique
        # size of the largest clique in the graph
        size_largest_clique = nx.graph_clique_number(graph)
        save_file.write('\nsize of the largest clique in the graph,%d' % size_largest_clique)
        
        # all the cliques in the graph
        
        all_clique = nx.find_cliques(graph) # a generator
        list_all_clique = list(all_clique)
        list_all_clique_sort = sorted(list_all_clique, key = lambda x: len(x), reverse = True)
        list_all_clique_sort = [' '.join(clique) for clique in list_all_clique_sort]
        # print list_all_clique_sort
        save_file.write('\ncliques,')
        save_file.write(','.join(x for x in list_all_clique_sort))

def  processDirectory(dir_path, top_number, window_size, occurrence_frequency, bound_flag, lemma_flag):
    """
        Process the directory the user inputs
    """
    if (not bound_flag):
        print 'NO SENTENCE BOUNDARY'
    
    if (not lemma_flag):
        print 'NO LEMMATIZING'

    cross_graph = nx.Graph()

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            print 'analyzing ' + file_name, '...'
            try:
                file_open = io.open(file_path, 'r')
            except IOError:
                print 'OPEN FILE ' + file_path + ' ERROR'
                sys.exit(1)
            list_word_no_stopword = preprocessSent(file_open.read(), bound_flag, lemma_flag)

            file_open.close()
            
            single_graph = drawCOGraph(list_word_no_stopword, window_size, bound_flag, lemma_flag, dir_path, file_name)
            
            # Doing the combination
            cross_graph.add_nodes_from([v for v, d in single_graph.nodes(data = True) if v not in cross_graph.nodes()], freq = 0)
            cross_graph.add_edges_from([(u, v) for u, v, d in single_graph.edges(data = True) if (u, v) not in cross_graph.edges()], freq = 0)

            for v, d in cross_graph.nodes(data = True):
                if v in single_graph.nodes():
                    d['freq'] += single_graph.node[v]['freq']

            for u, v, d in cross_graph.edges(data = True):
                if (u, v) in single_graph.edges():
                    d['freq'] += single_graph[u][v]['freq']
                elif (v, u) in single_graph.edges():
                    d['freq'] += single_graph[v][u]['freq']

    # occurrence_frequency to filter final graph edges
    if occurrence_frequency > 1:
        cross_graph.remove_edges_from([(u, v) for u, v, d in cross_graph.edges(data = True) if d['freq'] < occurrence_frequency])
        cross_graph.remove_nodes_from([v for v, degree in cross_graph.degree().items() if degree == 0])

    if lemma_flag and bound_flag:
        nx.write_pajek(cross_graph, dir_path + 'co_lemma_bound/' + 'co_graph_cross_txt_lemma_bound.net')
        graphAnalysis(cross_graph, top_number, dir_path + 'co_lemma_bound/' + 'co_graph_lemma_bound_analysis.csv')
    elif lemma_flag and not bound_flag:
        nx.write_pajek(cross_graph, dir_path + 'co_lemma_no_bound/' +'co_graph_cross_txt_lemma_no_bound.net')
        graphAnalysis(cross_graph, top_number, dir_path + 'co_lemma_no_bound/' + 'co_graph_lemma_no_bound_analysis.csv')
    elif not lemma_flag and bound_flag:
        nx.write_pajek(cross_graph, dir_path + 'co_bound_no_lemma/' + 'co_graph_cross_txt_bound_no_lemma.net')
        graphAnalysis(cross_graph, top_number, dir_path + 'co_bound_no_lemma/' + 'co_graph_bound_no_lemma_analysis.csv')
    else:
        nx.write_pajek(cross_graph, dir_path + 'co_no_lemma_no_bound/' + 'co_graph_cross_txt_no_lemma_no_bound.net')
        graphAnalysis(cross_graph, top_number, dir_path + 'co_no_lemma_no_bound/' + 'co_graph_no_lemma_no_bound_analysis.csv')


def main():
    """
        The main function of this script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_path', help = 'input directory path')
    # optional
    parser.add_argument('-t', '--top_number', default = 10, help = 'input integer defining <TOP_NUMBER> top-scoring degree centrality, clustering and clique analysis. default is 10', type = int)
    parser.add_argument('-w','--window_size', default = 1, help = 'input integer defining co-occurrence window size, i.e, how far in both directions to look for co-occurrences to each target word. default is 1', type = int)
    parser.add_argument('-f', '--occurrence_frequency', default = 1, help = 'input integer defining the minimum, collection-wide frequency a word must occur with to be included in the final combined graph. default is 1', type = int)
    parser.add_argument('-s', '--no_boundary', help = 'not to consider sentence boundary when finding co-occurrence word pair', action = 'store_true')
    parser.add_argument('-v','--no_lemma', help = 'not to lemmatize words in the file', action = 'store_true')
    
    args = parser.parse_args()
    dir_path = args.dir_path
    top_number = args.top_number
    window_size = args.window_size
    occurrence_frequency = args.occurrence_frequency
    
    print 'DIR_PATH', dir_path
    print 'TOP_NUMBER', top_number
    print 'WINDOW_SIZE', window_size
    print 'OCCURRENCE_FREQUENCY', occurrence_frequency
    
    if not os.path.exists(dir_path):
        print 'NO SUCH DIRECTORY! PLEASE INPUT AGAIN'
        sys.exit(1)
    else:
        if len([file_name for file_name in os.listdir(dir_path) if file_name.endswith('.txt')]) == 0:
            print 'NO TEXT FILE IN SUCH DIRECTORY! PLEASE INPUT ANOTHER DIRECTORY WITH TEXT FILES'
            sys.exit(1)

    if args.no_boundary:
        bound_flag = False
    else:
        bound_flag = True

    if args.no_lemma:
        lemma_flag = False
    else:
        lemma_flag = True
    program_start = time.clock()
    processDirectory(dir_path, top_number, window_size, occurrence_frequency, bound_flag, lemma_flag)
    program_end = time.clock()
    print 'DONE'
    # print 'Running time of the program is: %.2fs' % (program_end - program_start)


if __name__ == '__main__':
    main()





