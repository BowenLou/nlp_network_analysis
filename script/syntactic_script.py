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
import io

"""
    This script will process a bunch of files in the input directory, and generate syntactic graph of the corresponding file and finally build a combined graph of all these files. Also an analysis file is generated.
    
    usage: syntactic_script.py [-h] [-t TOP_NUMBER] [-v] dir_path
    
    positional arguments:
    dir_path              input directory path
    
    optional arguments:
    -h, --help            show this help message and exit
    -t TOP_NUMBER, --top_number TOP_NUMBER
                                input integer defining <TOP_NUMBER> top-scoring degree
                                centrality analysis. default is 10
    -v, --no_lemma        not to lemmatize words in the file
    
    input:
    1. The path to a directory of text files to be analyzed individually and as a collection.
    2. An integer, defining the the number of top-scoring nodes (or edges) for which to report various analysis (for example, the top 10 most central nodes in the graph). Default is 10.
    3. A Boolean, defining whether or not to lemmatize all words in the file before building the graphs. Default is to do the lemmatization.
    
    output:
    1. A .pajek graph file for each file in the input directory.
    2. A .pajek graph file for the combined graph (summing node and edge frequencies in the way we set)
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

def preprocessSent(file_content):
    """
        input:
            content in the file to be processed
        return:
            segmented sentences of the input file
        
    """
    sent_seg = nltk.sent_tokenize(file_content)
    # omit some special character which can not decoded in ascii and omit punctuation
    sent_seg = [sent.encode('ascii', 'ignore').translate(None, string.punctuation) for sent in sent_seg]
    return sent_seg

def produceDepend(list_sent, lemma_flag):
    """
        input:
            list of sentences in the file
        return:
            dependencies of all the sentences
    """
    nlp = StandfordNLP()
    sentences_dependencies = []
    
    if lemma_flag:
        for sent in list_sent:
            if len(sent.strip()) != 0:
                sentence_dependencies = []
                word_lemma_index_dict = {}
                word_index = 1
                for word in nlp.parse(sent)['sentences'][0]['words']:
                    word_lemma_index_dict[word_index] = word[1]['Lemma']
                    word_index += 1
                word_lemma_index_dict[0] = 'ROOT'

                for indexeddependency in nlp.parse(sent)['sentences'][0]['indexeddependencies']:
                    sentence_dependencies.append((indexeddependency[0], word_lemma_index_dict[int(indexeddependency[1].split('-')[1])], word_lemma_index_dict[int(indexeddependency[2].split('-')[1])]))
                sentences_dependencies.append(sentence_dependencies)
    else:
        for sent in list_sent:
            if len(sent.strip()) != 0:
                sentence_dependencies = []
                word_index_dict = {}
                word_index = 1
                for word in nlp.parse(sent)['sentences'][0]['words']:
                    word_index_dict[word_index] = word[0]
                    word_index += 1
                word_index_dict[0] = 'ROOT'
    
                for indexeddependency in nlp.parse(sent)['sentences'][0]['indexeddependencies']:
                    sentence_dependencies.append((indexeddependency[0], word_index_dict[int(indexeddependency[1].split('-')[1])], word_index_dict[int(indexeddependency[2].split('-')[1])]))
                sentences_dependencies.append(sentence_dependencies)

    return sentences_dependencies


def countWordFreq(list_sent):
    """
        input:
            list of sentences in the file
        return:
            a dictionary in which the key is the word and the value is the frequency of the word in the input
    """
    list_word = []
    for sent in list_sent:
        for word in nltk.word_tokenize(sent):
            list_word.append(word.lower())

    word_freq_count = dict((word, list_word.count(word)) for word in list_word)
    return  word_freq_count

def drawDependGraph(list_list_sent_depend, list_sent, dir_path, file_name, lemma_flag):
    """
        Draw and save the syntactic graph of each file in the directory
    """
    
    from nltk.corpus import stopwords
    stop_word = stopwords.words('english')
    
    nlp = StandfordNLP()
    
    if lemma_flag:
        list_lemma_sent = []
        for sent in list_sent:
            if len(sent.strip()) != 0:
                lemma_sent = []
                for word in nlp.parse(sent)['sentences'][0]['words']:
                    lemma_sent.append(str(word[1]['Lemma']))
                list_lemma_sent.append(' '.join(lemma_sent))
        # print list_lemma_sent
        word_freq_dict = countWordFreq(list_lemma_sent)
    else:
        list_no_lemma_sent = []
        for sent in list_sent:
            if len(sent.strip()) != 0:
                no_lemma_sent = []
                for word in nlp.parse(sent)['sentences'][0]['words']:
                    no_lemma_sent.append(word[0])
                list_no_lemma_sent.append(' '.join(no_lemma_sent))
        word_freq_dict = countWordFreq(list_no_lemma_sent)

    depend_graph = nx.MultiDiGraph()
    
    # do the dependent analysis before removing stop words
    for sentence_dependencies in list_list_sent_depend:
        for sentence_dependency in sentence_dependencies:
            if sentence_dependency[0] != 'root':
                head = sentence_dependency[1].lower()
                depend = sentence_dependency[2].lower()
                if not (head in stop_word or depend in stop_word):
                    depend_graph.add_node(head, freq = word_freq_dict[head])
                    depend_graph.add_node(depend, freq = word_freq_dict[depend])
                    
                    # process the edge
                    if not (head, depend) in depend_graph.edges():
                        depend_graph.add_edge(head, depend, dependency = sentence_dependency[0], label = sentence_dependency[0], freq = 1)
                    else:
                        # deal with the condition that different relations with the same head and dependent nodes
                        list_dif_depend = [depend_graph[head][depend][i]['dependency'] for i in range(len(depend_graph[head][depend].items()))]
                        dict_dif_depend = dict(zip(list_dif_depend, range(len(depend_graph[head][depend].items()))))
                        
                        # print dict_dif_depend
                        
                        if sentence_dependency[0] not in list_dif_depend:
                            depend_graph.add_edge(head, depend, dependency = sentence_dependency[0], label = sentence_dependency[0], freq = 1)
                        else:
                            depend_index = dict_dif_depend[sentence_dependency[0]]
                            depend_graph[head][depend][depend_index]['freq'] += 1

    # write to a .pajek graph file for each file in the input directory.
    if lemma_flag:
        if not os.path.exists(dir_path + 'syntactic_lemma/'):
            os.makedirs(dir_path + 'syntactic_lemma/')
        nx.write_pajek(depend_graph, dir_path + 'syntactic_lemma/' + file_name + '_syntactic_lemma.net')
    else:
        if not os.path.exists(dir_path + 'syntactic_no_lemma/'):
            os.makedirs(dir_path + 'syntactic_no_lemma/')
        nx.write_pajek(depend_graph, dir_path + 'syntactic_no_lemma/' + file_name + '_syntactic_no_lemma.net')
    return depend_graph

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
        #  http://networkx.github.io/documentation/latest/_modules/networkx/algorithms/cluster.html
        # networkx.exception.NetworkXError: triangles() is not defined for directed graphs.
        # networkx.exception.NetworkXError: ('Clustering algorithms are not defined ', 'for directed graphs.')


def processDirectory(dir_path, top_number, lemma_flag):
    """
        Process the directory the user inputs
    """
    if (not lemma_flag):
        print 'NO LEMMATIZING'
    
    cross_graph = nx.MultiDiGraph()

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            print 'analyzing ' + file_name, '...'
            try:
                file_open = io.open(file_path, 'r')
            except IOError:
                print 'OPEN FILE ' + file_path + ' ERROR'
                sys.exit(1)
            sent_seg = preprocessSent(file_open.read())
            sent_depend = produceDepend(sent_seg, lemma_flag)
            # print sent_seg
            # print sent_depend
            file_open.close()
            
            single_graph = drawDependGraph(sent_depend, sent_seg, dir_path, file_name, lemma_flag)
            
            # Doing the combination
            cross_graph.add_nodes_from([v for v, d in  single_graph.nodes(data = True) if v not in cross_graph.nodes()], freq = 0)
            
            for u, v, d in single_graph.edges(data = True):
                if (u, v) not in cross_graph.edges():
                    cross_graph.add_edge(u, v, dependency = d['dependency'], label = d['label'], freq = 0)
                else:
                    list_dif_depend = [cross_graph[u][v][i]['dependency'] for i in range(len(cross_graph[u][v].items()))]
                    if d['dependency'] not in list_dif_depend:
                        cross_graph.add_edge(u, v, dependency = d['dependency'], label = d['label'], freq = 0)
                    

            for v, d in cross_graph.nodes(data = True):
                if v in single_graph.nodes():
                    d['freq'] += single_graph.node[v]['freq']
            
            for u, v, d in cross_graph.edges(data = True):
                if (u, v) in single_graph.edges():
                    list_dif_depend = [single_graph[u][v][i]['dependency'] for i in range(len(single_graph[u][v].items()))]
                    dict_dif_depend = dict(zip(list_dif_depend, range(len(single_graph[u][v].items()))))
                    if d['dependency'] in list_dif_depend:
                        depend_index = dict_dif_depend[d['dependency']]
                        d['freq'] += single_graph[u][v][depend_index]['freq']

    if lemma_flag:
        nx.write_pajek(cross_graph, dir_path + 'syntactic_lemma/' + 'syntactic_graph_cross_txt_lemma.net')
        graphAnalysis(cross_graph, top_number, dir_path + 'syntactic_lemma/' + 'syntactic_graph_lemma_analysis.csv')
    else:
        nx.write_pajek(cross_graph, dir_path + 'syntactic_no_lemma/' + 'syntactic_graph_cross_txt_no_lemma.net')
        graphAnalysis(cross_graph, top_number, dir_path + 'syntactic_no_lemma/' + 'syntactic_graph_analysis_no_lemma.csv')


def main():
    """
        The main function of this script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_path', help = 'input directory path')
    
    parser.add_argument('-t', '--top_number',default = 10, help = 'input integer defining <TOP_NUMBER> top-scoring degree centrality analysis. default is 10', type = int)
    parser.add_argument('-v','--no_lemma', help = 'not to lemmatize words in the file', action = 'store_true')
    args = parser.parse_args()
    dir_path = args.dir_path
    top_number = args.top_number
    
    print 'DIR_PATH: ', dir_path
    print 'TOP_NUMBER: ', top_number

    if not os.path.exists(dir_path):
        print 'NO SUCH DIRECTORY! PLEASE INPUT AGAIN'
        sys.exit(1)
    else:
        if len([file_name for file_name in os.listdir(dir_path) if file_name.endswith('.txt')]) == 0:
            print 'NO TEXT FILE IN SUCH DIRECTORY! PLEASE INPUT ANOTHER DIRECTORY WITH TEXT FILES'
            sys.exit(1)
    
    if args.no_lemma:
        lemma_flag = False
    else:
        lemma_flag = True
    program_start = time.clock()
    processDirectory(dir_path, top_number, lemma_flag)
    program_end = time.clock()
    print 'DONE'
    # print 'Running time of the program is: %.2fs' % (program_end - program_start)


if __name__ == '__main__':
    main()

