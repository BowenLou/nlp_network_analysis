# -*- coding: utf-8 -*-
import nltk
import matplotlib.pyplot as plt
import ner
import os
import time
import string
import argparse
import getopt
import sys
import io

from igraph import *
# from nltk.stem.wordnet import WordNetLemmatizer

"""
    This script will process a bunch of files in the input directory, and do the community detection to see the variation of the community structure when file is added one by one.
    
    usage: ner_community_detection_script.py [-h] [-d] [-n] [-e] dir_path

    positional arguments:
      dir_path              input directory path

    optional arguments:
      -h, --help            show this help message and exit
      -d, --edge_distance   consider edge distance in the edge attribute
      -n, --no_node_frequency_normalization
                            not to consider normalizing node frequencies when finding NE pair
      -e, --no_edge_weight_normalization
                            not to consider normalizing edge weights when finding NE pair

    input:
    1. The path to a directory of text files to be analyzed individually and as a collection.
    2. A Boolean, specifying whether or not to assign the edge distance to edge attribute in the final combined graph. Default is not to assign.
    3. A Boolean, specifying whether or not to normalize the node-frequencies in the final combined graph. Default is to normalize.
    4. A Boolean, specifying whether or not to normalize the edge-weights in the final combined graph. Default is to normalize.
    
    output:
    1. Texts in generated folder to report NE community variation by added files.
        (reference algorithm: community_infomap(self, edge_weights=None, vertex_weights=None, trials=10), community_optimal_modularity(self, *args, **kwds) )

"""

__author__ = "Bowen Lou, Knowledge Lab | Computation Institute"
__version__ = "v0.1"


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
                            list_ne.append(str(word.lower()) + ':' + str(entity))
    
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

def drawCompleteNEGraph(list_ne, edge_distance_flag):
    """
        Generate the complete named entity graph of each file in the directory
    """
    
    ne_freq_count_dict = countWordFreq(list_ne)
    
    list_ne_set = ne_freq_count_dict.keys()
    ne_graph = Graph.Full(len(list_ne_set))
    ne_graph.vs['name'] = list_ne_set
    ne_graph.es['weight'] = 0
    # ne_graph.vs['label'] = list_ne

    for v in ne_graph.vs:
        v['freq'] = ne_freq_count_dict[v['name']]
    
    for e in ne_graph.es:
        if edge_distance_flag:
            e['weight'] = 1.0 / min(ne_graph.vs[e.source]['freq'], ne_graph.vs[e.target]['freq'])
        else:    
            e['weight'] = 1
    
#     # test
#     for e in ne_graph.es:
#         source = ne_graph.vs[e.source]['name']
#         target = ne_graph.vs[e.target]['name']
#         print source, target, ne_graph[source, target]
    return ne_graph

def graphCommunityAnalysis(analysis_graph, edge_distance_flag, dir_path, list_analysis_file, file_index):
    if not os.path.exists(dir_path + 'ner_community_variation/'):
        os.makedirs(dir_path + 'ner_community_variation/')
    
    if edge_distance_flag:
        if not os.path.exists(dir_path + 'ner_community_variation/edge_dist/'):
            os.makedirs(dir_path + 'ner_community_variation/edge_dist/')
        cl_infomap = analysis_graph.community_infomap(edge_weights = analysis_graph.es['weight'], vertex_weights = analysis_graph.vs['freq'])
#         print analysis_graph.modularity(cl_infomap)
#         print cl_infomap
        list_cl_community = []
    
        for cl_community in list(cl_infomap):
            list_community = []
            for member in cl_community:
                list_community.append(analysis_graph.vs[member]['name'])
            list_cl_community.append(list_community)
            
        with io.open(dir_path + 'ner_community_variation/edge_dist/' + 'text' + str(file_index) + '.txt', 'w') as save_file:
            save_file.write('analysis text:\n' + ','.join(list_analysis_file) + '\n')
            save_file.write('modularity:\n')
            save_file.write('%f' % analysis_graph.modularity(cl_infomap))
            save_file.write('\ncommunity:\n')
            for community_index, cl_community in enumerate(list_cl_community):
                save_file.write('c' + str(community_index + 1) + ' ' + ','.join(cl_community) + '\n')
    
    else:
        if not os.path.exists(dir_path + 'ner_community_variation/edge_weight/'):
            os.makedirs(dir_path + 'ner_community_variation/edge_weight/')
        cl_optimal_modularity = analysis_graph.community_optimal_modularity()
#         print analysis_graph.modularity(cl_optimal_modularity)
#         print cl_optimal_modularity
        list_cl_community = []
    
        for cl_community in list(cl_optimal_modularity):
            list_community = []
            for member in cl_community:
                list_community.append(analysis_graph.vs[member]['name'])
            list_cl_community.append(list_community)
    
        with io.open(dir_path + 'ner_community_variation/edge_weight/' + 'text' + str(file_index) + '.txt', 'w') as save_file:
            save_file.write('analysis text:\n' + ','.join(list_analysis_file) + '\n')
            save_file.write('modularity:\n')
            save_file.write('%f' % analysis_graph.modularity(cl_optimal_modularity))
            save_file.write('\ncommunity:\n')
            for community_index, cl_community in enumerate(list_cl_community):
                save_file.write('c' + str(community_index + 1) + ' ' + ','.join(cl_community) + '\n')
        
def processDirectory(dir_path, edge_distance_flag, node_norm_flag, edge_norm_flag):
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
    
    cross_graph = Graph()
    cross_graph.vs['name'] = []
    cross_graph.es['weight'] = 0
    list_cross_graph_edge = []
    
    list_analysis_file = []
    
    file_index = 1 # just count the number of files in certain directory
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            print 'analyzing ' + file_name, '...'
            list_analysis_file.append(file_name)
            try:
                file_open = io.open(file_path, 'r')
            except IOError:
                print 'OPEN FILE ' + file_path + 'ERROR'
                sys.exit(1)
            sent_seg = preprocessSent(file_open.read())
            file_open.close()
            
            list_ne = produceNamedEntity(sent_seg)
            single_graph = drawCompleteNEGraph(list_ne, edge_distance_flag)   
            
            list_single_graph_vertex_name = single_graph.vs['name']
            for v in single_graph.vs:
                if v['name'] not in cross_graph.vs['name']:
                    cross_graph.add_vertex(v['name'])
                    cross_graph.vs.find(name = v['name'])['freq'] = 0
            
            for index, v in enumerate(list_single_graph_vertex_name):
                for u in list_single_graph_vertex_name[index + 1:]:
                    if (v, u) not in list_cross_graph_edge and (u, v) not in list_cross_graph_edge:
                        list_cross_graph_edge.append((v, u))
                        cross_graph.add_edge(v, u)
                        cross_graph[v, u] = 0
            
            for v_name in cross_graph.vs['name']:
                if v_name in list_single_graph_vertex_name:
                    cross_graph.vs.find(name = v_name)['freq'] += single_graph.vs.find(name=v_name)['freq']
            
            list_single_graph_edge = []
            for e in single_graph.es:
                list_single_graph_edge.append((single_graph.vs[e.source]['name'], single_graph.vs[e.target]['name']))
            
            for (v, u) in list_cross_graph_edge:
                if edge_distance_flag:
                    if (v, u) in list_single_graph_edge:
                        if cross_graph[v, u] == 0:
                            cross_dist = 0
                        else:
                            cross_dist = 1.0 / cross_graph[v, u]
                        single_graph_dist = 1.0 / single_graph[v, u]
                        cross_dist += single_graph_dist
                        cross_graph[v, u] = 1.0 / cross_dist
                    elif (u, v) in list_single_graph_edge:
                        if cross_graph[u, v] == 0:
                            cross_dist = 0
                        else:
                            cross_dist = 1.0 / cross_graph[u, v]
                        single_graph_dist = 1.0 / single_graph[u, v]
                        cross_dist += single_graph_dist
                        cross_graph[u, v] = 1.0 / cross_dist         
                else:
                    if (v, u) in list_single_graph_edge:
                        cross_graph[v, u] += single_graph[v, u]
                    elif (u, v) in list_single_graph_edge:
                        cross_graph[u, v] += single_graph[u, v]
            
            # can do vertex and edge normalization in this certain cross_graph
            graphCommunityAnalysis(cross_graph, edge_distance_flag, dir_path, list_analysis_file, file_index)
            file_index += 1
    
    # final combined graph info
    sum_node_freq = 0
    for v in cross_graph.vs:
        sum_node_freq += v['freq']
    
    sum_edge_weight = 0
    for e in cross_graph.es:
        source = cross_graph.vs[e.source]['name']
        target = cross_graph.vs[e.target]['name']
        sum_edge_weight += cross_graph[source, target]

    if node_norm_flag and edge_norm_flag:
        for v in cross_graph.vs:
            v['freq'] /= float(sum_node_freq)
        for e in cross_graph.es:
            source = cross_graph.vs[e.source]['name']
            target = cross_graph.vs[e.target]['name']
            cross_graph[source, target] /= float(sum_edge_weight)
    elif node_norm_flag and not edge_norm_flag:
        for v in cross_graph.vs:
            v['freq'] /= float(sum_node_freq)
    elif not node_norm_flag and edge_norm_flag:
        for e in cross_graph.es:
            source = cross_graph.vs[e.source]['name']
            target = cross_graph.vs[e.target]['name']
            cross_graph[source, target] /= float(sum_edge_weight)
    # else:

#     # see final combined graph info
#     print cross_graph
    
#     print '--------------vertex----------------'
#     for v in cross_graph.vs:
#         print v
    
#     print '--------------edge----------------'
#     for e in cross_graph.es:
#         source = cross_graph.vs[e.source]['name']
#         target = cross_graph.vs[e.target]['name']
#         print source, target, cross_graph[source, target]
        

def main():
    """
        The main function of this script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_path', help = 'input directory path')
    # optional
    parser.add_argument('-d','--edge_distance', help = 'consider edge distance in the edge attribute', action = 'store_true')
    parser.add_argument('-n', '--no_node_frequency_normalization', help = 'not to consider normalizing node frequencies when finding NE pair', action = 'store_true')
    parser.add_argument('-e','--no_edge_weight_normalization', help = 'not to consider normalizing edge weights when finding NE pair', action = 'store_true')
    
    args = parser.parse_args()
    dir_path = args.dir_path
    
    print 'DIR_PATH', dir_path
    
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
    
    if args.no_edge_weight_normalization:
        edge_norm_flag = False
    else:
        edge_norm_flag = True
    
    program_start = time.clock()
    processDirectory(dir_path, edge_distance_flag, node_norm_flag, edge_norm_flag)
    program_end = time.clock()
    print 'DONE'
    # print 'Running time of the program is: %.2fs' % (program_end - program_start)


if __name__ == '__main__':
    main()

