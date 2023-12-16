import json
import pydotplus

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def draw_pydotplus(node, edge, node_weight=None, edge_weight=None, node_info=None, edge_info=None, path='pydotplus.pdf'):
    color_list = ['#eeeeff', '#ccccff', '#9999ff', '#7777ff', '#5555ff']
    pydot_node = []
    graph = pydotplus.Dot(graph_type='digraph')
    for i in range(len(node)):
        ni = str(node_info[i]) if node_info else ''
        cl = color_list[node_weight[i] if node_weight and node_weight[i] < len(color_list) else 0]
        na = pydotplus.Node(node[i], xlabel=ni, style='filled', shape='rect', fillcolor=cl)
        pydot_node.append(na)
        graph.add_node(na)
    
    for i in range(len(edge)):
        n = edge[i]
        s = n[0]
        t = n[1]
        graph.add_edge(pydotplus.Edge(pydot_node[s], pydot_node[t], label=str(edge_info[i]) if edge_info else ''))

    graph.write_pdf(path)


def draw_pydotplus_withsubgraph(node, edge, node_to_sub, node_weight=None, edge_weight=None, node_info=None, edge_info=None, path='pydotplus.pdf'):
    color_subgraph_list = ['red', 'blue', 'green', 'yellow', 'orange', 'blueviolet', 'aquamarine', 'purple', 'grey']
    color_list = ['#eeeeff', '#ccccff', '#9999ff', '#7777ff', '#5555ff']
    subgraph = list(set([node_to_sub[i] for i in node_to_sub]))
    subgraph_set = []
    pydot_node = []
    graph = pydotplus.Dot(graph_type='digraph')
    num = 0
    for sub_name in subgraph:
        sub = pydotplus.Subgraph('cluster_' + str(num), label=sub_name, color=color_subgraph_list[num % len(color_subgraph_list)])
        for i in range(len(node)):
            if node[i] in node_to_sub and node_to_sub[node[i]] == sub_name: 
                ni = str(node_info[i]) if node_info else ''
                cl = color_list[node_weight[i] if node_weight and node_weight[i] < len(color_list) else 0]
                na = pydotplus.Node(node[i], xlabel=ni, style='filled', shape='rect', fillcolor=cl)
                sub.add_node(na)
        graph.add_subgraph(sub)
        num += 1
        subgraph_set.append(graph)  
    for i in range(len(node)):
        ni = str(node_info[i]) if node_info else ''
        cl = color_list[node_weight[i] if node_weight and node_weight[i] < len(color_list) else 0]
        na = pydotplus.Node(node[i], xlabel=ni, style='filled', shape='rect', fillcolor=cl)
        pydot_node.append(na)  
        if node[i] not in node_to_sub or node_to_sub[node[i]] == None:  
            graph.add_node(na)
            
    for i in range(len(edge)):
        n = edge[i]
        s = n[0]
        t = n[1]
        graph.add_edge(pydotplus.Edge(pydot_node[s], pydot_node[t], label=str(edge_info[i]) if edge_info else ''))
    graph.write_pdf(path)

def draw_networkx(node, edge, node_weight=None, edge_weight=None, node_info=None, edge_info=None):
    color_list = ['#eeeeff', '#ccccff', '#9999ff', '#7777ff', '#5555ff']
    net_node = []
    node_colors = []
    graph = nx.DiGraph()
    for i in range(len(node)):
        net_node.append(node[i])
        graph.add_node(node[i])
        cl = color_list[node_weight[i] if node_weight and node_weight[i] < len(color_list) else 0]
        node_colors.append(cl)

    for i in range(len(edge)):
        n = edge[i]
        s = n[0]
        t = n[1]
        graph.add_edge(net_node[s], net_node[t], label=edge_info[i] if edge_info else '')
        
    plt.figure()
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color=node_colors)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'label'))
    plt.show()

def draw_networkx_withsubgraph(node, edge, node_to_sub, node_weight=None, edge_weight=None, node_info=None, edge_info=None):
    color_subgraph_list = ['red', 'blue', 'green', 'yellow', 'orange', 'blueviolet', 'aquamarine', 'purple', 'grey']
    color_list = ['#eeeeff', '#ccccff', '#9999ff', '#7777ff', '#5555ff']
    net_node = []
    node_colors = []
    subgraph = list(set([node_to_sub[i] for i in node_to_sub]))
    subgraph_set = []
    graph = nx.DiGraph()
    num = 0
    for sub_name in subgraph:
        sub = nx.DiGraph()
        for i in range(len(node)):
            if node[i] in node_to_sub and node_to_sub[node[i]] == sub_name:  # 如果该节点属于该子图
                net_node.append(node[i])
                sub.add_node(node[i])
                cl = color_list[node_weight[i] if node_weight and node_weight[i] < len(color_list) else 0]
                node_colors.append(cl)
        num += 1
        graph.add_node(sub)
        subgraph_set.append(sub) 
    for i in range(len(node)):
        net_node.append(node[i])
        graph.add_node(node[i])
        cl =  color_list[node_weight[i] if node_weight and node_weight[i] < len(color_list) else 0]
        node_colors.append(cl)

    for i in range(len(edge)):
        n = edge[i]
        s = n[0]
        t = n[1]
        graph.add_edge(net_node[s], net_node[t], label=edge_info[i] if edge_info else '')
    
    color = color_subgraph_list[num % len(color_subgraph_list)]
    plt.figure()
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color=node_colors)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'label'))
    plt.show()
