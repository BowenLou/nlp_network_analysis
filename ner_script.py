# -*- coding: utf-8 -*-
from __future__ import unicode_literals, division
import nltk
import networkx as nx
import matplotlib.pyplot as plt
import ner
import os
import time
import string
import argparse
import getopt
import sys
import io
import itertools

# from nltk.stem.wordnet import WordNetLemmatizer

"""
    This script will process a bunch of files in the input directory, and generate named entity graph of the corresponding file and finally build a combined graph of all these files. Also analysis files are generated.
    
    usage: ner_script.py [-h] [-t TOP_NUMBER] [-d] [-n] [-e] dir_path
    
    positional arguments:
    dir_path              input directory path
    
    optional arguments:
    -h, --help            show this help message and exit
    -t TOP_NUMBER, --top_number TOP_NUMBER  
                          input integer defining <TOP_NUMBER> top-scoring degree centrality, clustering and clique analysis. Also process closeness centrality and betweenness centrality analysis if edge weight is in distance. default is 10
    -d, --edge_distance   consider edge distance in the edge attribute
    -n, --no_node_frequency_normalization
                          not to consider normalizing node frequencies when finding NE pair
    -e, --no_edge_attribute_normalization   
                          not to consider normalizing edge attribute when finding NE pair


    input:
    1. The path to a directory of text files to be analyzed individually and as a collection.
    2. An integer, defining the number of top-scoring nodes (or edges) for which to report various analysis (for example, the top 10 most central nodes in the graph). Default is 10.
    3. A Boolean, specifying whether or not to assign the edge distance to edge attribute in the final combined graph. Default is not to assign.
    4. A Boolean, specifying whether or not to normalize the node-frequencies in the final combined graph. Default is to normalize.
    5. A Boolean, specifying whether or not to normalize the edge attribute in the final combined graph. Default is to normalize.
    
    output:
    1. A .pajek graph file for each file in the input directory.
    2. A .pajek graph file for the combined graph (summing node and edge frequencies in the way we set)
    3. A CSV reporting the top <TOP_NUMBER> highest-scoring elements (nodes or edges) for the essential analyses

"""

__author__ = "Bowen Lou, Knowledge Lab | Computation Institute"
__version__ = "v1.1"


def preprocessSent(file_content):
    """
        input:
            content in the file to be processed
        return:
            segmented sentences of the input file
    """
    sent_seg = nltk.sent_tokenize(file_content)
    # omit some special character which can not be encoded in ascii and omit punctuation
    sent_seg = [sent.encode('ascii', 'ignore') for sent in sent_seg]
    return sent_seg

def produceNamedEntity(list_sent):
    """
        input:
            list of sentences in the file
        return:
            list of all the named entities in the file
    """
    
    # lmtzr = WordNetLemmatizer()
    pyner_tagger = ner.SocketNER(host = 'localhost', port = 8080)

    list_ne = []
    for sent in list_sent:
        if len(sent.strip()) != 0:
            sent_ne_dict = pyner_tagger.get_entities(sent)
            sent_ne_dict_len = len(sent_ne_dict.items())

            if sent_ne_dict_len != 0:
                for entity, list_word in sent_ne_dict.items():
                    # print entity, list_word
                    for word in list_word:
                        if len(word) > 2:
                            list_ne.append((str(word.lower()), str(entity)))
    return list_ne

def countWordFreq(list_ne):
    """
        input:
            list of named entities in the file
        return:
            a dictionary in which the key is the named entity and the value is the frequency of the named entity in the input
    """
    ne_freq_count_dict = dict((ne, list_ne.count(ne)) for ne in list_ne)
    return ne_freq_count_dict

def drawCompleteNEGraph(list_ne, edge_distance_flag, dir_path, file_name):
    """
        Draw and save the complete named entity graph of each file in the directory
    """
    # modify complete_graph method from http://networkx.github.io/documentation/latest/_modules/networkx/generators/classic.html#complete_graph

    ne_graph = nx.Graph()
    ne_freq_count_dict = countWordFreq(list_ne)
    for ne_freq in ne_freq_count_dict.items():
        ne_graph.add_node(ne_freq[0], freq = ne_freq[1])
    
    if len(list_ne) > 1:
        edges = itertools.combinations(list_ne, 2)
        if edge_distance_flag:
            for edge in edges:
                if edge[0] != edge[1]:
                    edge_len = 1.0 / min(ne_freq_count_dict[edge[0]], ne_freq_count_dict[edge[1]])
                    ne_graph.add_edge(edge[0], edge[1], freq = edge_len)
        else:
            for edge in edges:
                if edge[0] != edge[1]:
                    ne_graph.add_edge(edge[0], edge[1], freq = 1)

    if not os.path.exists(dir_path + 'ner/'):
        os.makedirs(dir_path + 'ner/')

    if edge_distance_flag:
        if not os.path.exists(dir_path + 'ner/edge_dist/'):
            os.makedirs(dir_path + 'ner/edge_dist/')
        nx.write_pajek(ne_graph, dir_path + 'ner/edge_dist/' + file_name + '_ner.net')
    else:
        if not os.path.exists(dir_path + 'ner/edge_weight/'):
            os.makedirs(dir_path + 'ner/edge_weight/')
        nx.write_pajek(ne_graph, dir_path + 'ner/edge_weight/' + file_name + '_ner.net')

    return ne_graph

def graphAnalysis(graph, top_number, edge_distance_flag, save_file_path):
    """
        Do the essential analysis to the final combined graph
    """
    with io.open(save_file_path, 'w') as save_file:
        
        # centrality
        # degree centrality
        deg_central = nx.degree_centrality(graph)
        deg_central_sort = sorted(deg_central.items(), key = lambda x: x[1], reverse = True)
        top_deg_central_sort = []
        for ne_deg in deg_central_sort[:top_number]:
            top_deg_central_sort.append((' '.join(ne_deg[0]), ne_deg[1]))
        save_file.write('top %d degree centrality items,' % top_number)
        save_file.write(','.join('%s %s' % x for x in top_deg_central_sort))
        
        if edge_distance_flag:
            # closeness centrality
            close_central = nx.closeness_centrality(graph, distance = 'freq')
            close_central_sort = sorted(close_central.items(), key = lambda x: x[1], reverse = True)
            top_close_central_sort = []
            for ne_close in close_central_sort[:top_number]:
                top_close_central_sort.append((' '.join(ne_close[0]), ne_close[1]))
            save_file.write('\ntop %d closeness centrality items,' % top_number)
            save_file.write(','.join('%s %s' % x for x in top_close_central_sort))
    
            # betweenness centrality
            between_central = nx.betweenness_centrality(graph, weight = 'freq')
            between_central_sort = sorted(between_central.items(), key = lambda x: x[1], reverse = True)
            top_between_central_sort = []
            for ne_between in between_central_sort[:top_number]:
                top_between_central_sort.append((' '.join(ne_between[0]), ne_between[1]))
            save_file.write('\ntop %d betweenness centrality items,' % top_number)
            save_file.write(','.join('%s %s' % x for x in top_between_central_sort))
        
        # clustering
        
        # number of triangles: triangles() is not defined for directed graphs
        triangle_num = nx.triangles(graph)
        triangle_num_sort = sorted(triangle_num.items(), key = lambda x: x[1], reverse = True)
        top_triangle_num_sort = []
        for ne_triangle in triangle_num_sort[:top_number]:
            top_triangle_num_sort.append((' '.join(ne_triangle[0]), ne_triangle[1]))
        save_file.write('\ntop %d number of triangles including a node as one vertex,' % top_number)
        save_file.write(','.join('%s %s' % x for x in top_triangle_num_sort))
        
        # clustering coefficient of node in the graph
        cluster_coefficient = nx.clustering(graph)
        cluster_coefficient_sort = sorted(cluster_coefficient.items(), key = lambda x: x[1], reverse = True)
        top_cluster_coefficient_sort = []
        for ne_cluster_coefficient in cluster_coefficient_sort[:top_number]:
            top_cluster_coefficient_sort.append((' '.join(ne_cluster_coefficient[0]), ne_cluster_coefficient[1]))
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
        save_file.write('\ncliques,')
        write_all_clique_sort = []
        for clique in list_all_clique_sort:
            clique_string = ''
            for ne in clique:
                clique_string += str(' '.join(ne)) + '|'
            write_all_clique_sort.append(clique_string)
        save_file.write(','.join(x for x in write_all_clique_sort))

def processDirectory(dir_path, top_number, edge_distance_flag, node_norm_flag, edge_norm_flag):
    """
        Process the directory the user inputs
    """
    if edge_distance_flag:
        print "EDGE ATTRIBUTE IS EDGE DISTANCE"
    else:
        print "EDGE ATTRIBUTE IS EDGE WEIGHT"
    
    if (not node_norm_flag):
        print "NO NODE FREQUENCY NORMALIZATION"
    else:
        print "NODE FREQUENCY NORMALIZATION: Divide each node frequency by the sum of all frequencies"

    if (not edge_norm_flag):
        print "NO EDGE ATTRIBUTE NORMALIZATION"
    else:
        print "EDGE ATTRIBUTE NORMALIZATION: Divide each edge attribute by the sum of all attribute value"

    cross_graph = nx.Graph()

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            print 'analyzing ' + file_name, '...'
            try:
                file_open = io.open(file_path, 'r')
            except IOError:
                print 'OPEN FILE ' + file_path + 'ERROR'
                sys.exit(1)
            sent_seg = preprocessSent(file_open.read())
            file_open.close()
            
            list_ne = produceNamedEntity(sent_seg)
            single_graph = drawCompleteNEGraph(list_ne, edge_distance_flag, dir_path, file_name)

            cross_graph.add_nodes_from([v for v, d in single_graph.nodes(data = True) if v not in cross_graph.nodes()], freq = 0)
            cross_graph.add_edges_from([(u, v) for u, v, d in single_graph.edges(data = True) if (u, v) not in cross_graph.edges() and (v, u) not in cross_graph.edges()], freq = 0)

            for v, d in cross_graph.nodes(data = True):
                if v in single_graph.nodes():
                    d['freq'] += single_graph.node[v]['freq']
        
            for u, v, d in cross_graph.edges(data = True):
                if edge_distance_flag:
                    if (u,v) in single_graph.edges():
                        if d['freq'] == 0:
                            cross_dist = 0
                        else:
                            cross_dist = 1.0 / d['freq']
                        single_graph_dist = 1.0 / single_graph[u][v]['freq']
                        cross_dist += single_graph_dist
                        d['freq'] = 1.0 / cross_dist
                    elif (v, u) in single_graph.edges():
                        if d['freq'] == 0:
                            cross_dist = 0
                        else:
                            cross_dist = 1.0 / d['freq']
                        single_graph_dist = 1.0 / single_graph[v][u]['freq']
                        cross_dist += single_graph_dist
                        d['freq'] = 1.0 / cross_dist
                else:
                    if (u, v) in single_graph.edges():
                        d['freq'] += single_graph[u][v]['freq']
                    elif (v, u) in single_graph.edges():
                        d['freq'] += single_graph[v][u]['freq']

    sum_node_freq = 0
    for v, d in cross_graph.nodes(data = True):
        sum_node_freq += d['freq']
    
    sum_edge_attribute = 0
    for u, v, d in cross_graph.edges(data = True):
        sum_edge_attribute += d['freq']

    if node_norm_flag and edge_norm_flag:
        for v, d in cross_graph.nodes(data = True):
            d['freq'] /= sum_node_freq
        for u, v, d in cross_graph.edges(data = True):
            d['freq'] /= sum_edge_attribute
            
        if edge_distance_flag:
            nx.write_pajek(cross_graph, dir_path + 'ner/edge_dist/' + 'ne_graph_cross_txt_all_norm_edge_dist.net')
            graphAnalysis(cross_graph, top_number, edge_distance_flag, dir_path + 'ner/edge_dist/' + 'ne_graph_all_norm_edge_dist_analysis.csv')
        else:
            nx.write_pajek(cross_graph, dir_path + 'ner/edge_weight/' + 'ne_graph_cross_txt_all_norm_edge_weight.net')
            graphAnalysis(cross_graph, top_number, edge_distance_flag, dir_path + 'ner/edge_weight/' + 'ne_graph_all_norm_edge_weight_analysis.csv')
    elif node_norm_flag and not edge_norm_flag:
        for v, d in cross_graph.nodes(data = True):
            d['freq'] /= sum_node_freq
            
        if edge_distance_flag:
            nx.write_pajek(cross_graph, dir_path + 'ner/edge_dist/' + 'ne_graph_cross_txt_node_norm_edge_dist.net')
            graphAnalysis(cross_graph, top_number, edge_distance_flag, dir_path + 'ner/edge_dist/' + 'ne_graph_node_norm_edge_dist_analysis.csv')
        else:
            nx.write_pajek(cross_graph, dir_path + 'ner/edge_weight/' + 'ne_graph_cross_txt_node_norm_edge_weight.net')
            graphAnalysis(cross_graph, top_number, edge_distance_flag, dir_path + 'ner/edge_weight/' + 'ne_graph_node_norm_edge_weight_analysis.csv')
    elif not node_norm_flag and edge_norm_flag:
        for u, v, d in cross_graph.edges(data = True):
            d['freq'] /= sum_edge_attribute
        
        if edge_distance_flag:
            nx.write_pajek(cross_graph, dir_path + 'ner/edge_dist/' + 'ne_graph_cross_txt_edge_norm_edge_dist.net')
            graphAnalysis(cross_graph, top_number, edge_distance_flag, dir_path + 'ner/edge_dist/' + 'ne_graph_edge_norm_edge_dist_analysis.csv')
        else:
            nx.write_pajek(cross_graph, dir_path + 'ner/edge_weight/' + 'ne_graph_cross_txt_edge_norm_edge_weight.net')
            graphAnalysis(cross_graph, top_number, edge_distance_flag, dir_path + 'ner/edge_weight/' + 'ne_graph_edge_norm_edge_weight_analysis.csv')
    else:
        if edge_distance_flag:
            nx.write_pajek(cross_graph, dir_path + 'ner/edge_dist/' + 'ne_graph_cross_txt_no_norm_edge_dist.net')
            graphAnalysis(cross_graph, top_number, edge_distance_flag, dir_path + 'ner/edge_dist/' + 'ne_graph_no_norm_edge_dist_analysis.csv')
        else:
            nx.write_pajek(cross_graph, dir_path + 'ner/edge_weight/' + 'ne_graph_cross_txt_no_norm_edge_weight.net')
            graphAnalysis(cross_graph, top_number, edge_distance_flag, dir_path + 'ner/edge_weight/' + 'ne_graph_no_norm_edge_weight_analysis.csv')

def main():
    """
        The main function of this script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_path', help = 'input directory path')
    # optional
    parser.add_argument('-t', '--top_number', default = 10, help = 'input integer defining <TOP_NUMBER> top-scoring degree centrality, clustering and clique analysis. Also process closeness centrality and betweenness centrality analysis if edge weight is in distance. default is 10', type = int)
    parser.add_argument('-d','--edge_distance', help = 'consider edge distance in the edge attribute', action = 'store_true')
    parser.add_argument('-n', '--no_node_frequency_normalization', help = 'not to consider normalizing node frequencies when finding NE pair', action = 'store_true')
    parser.add_argument('-e','--no_edge_attribute_normalization', help = 'not to consider normalizing edge attribute when finding NE pair', action = 'store_true')
    
    args = parser.parse_args()
    dir_path = args.dir_path
    top_number = args.top_number
    
    print 'DIR_PATH', dir_path
    print 'TOP_NUMBER', top_number

    if not os.path.exists(dir_path):
        print 'NO SUCH DIRECTORY! PLEASE INPUT AGAIN'
        sys.exit(1)
    else:
        if len([file_name for file_name in os.listdir(dir_path) if file_name.endswith('.txt')]) == 0:
            print 'NO TEXT FILE IN SUCH DIRECTORY! PLEASE INPUT ANOTHER DIRECTORY WITH TEXT FILES'
            sys.exit(1)
    
    if args.edge_distance:
        edge_distance_flag = True
    else:
        edge_distance_flag = False
    
    if args.no_node_frequency_normalization:
        node_norm_flag = False
    else:
        node_norm_flag = True

    if args.no_edge_attribute_normalization:
        edge_norm_flag = False
    else:
        edge_norm_flag = True

    program_start = time.clock()
    processDirectory(dir_path, top_number, edge_distance_flag, node_norm_flag, edge_norm_flag)
    program_end = time.clock()
    print 'DONE'
    # print 'Running time of the program is: %.2fs' % (program_end - program_start)


if __name__ == '__main__':
    main()

