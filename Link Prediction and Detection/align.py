import os
import ast
import json
import draw

def get_json(path):
    json_list = []
    for dir, ds, files in os.walk(path):
        for file in files:
            if file.endswith('.json'):
                json_list.append(os.path.join(dir, file))
    return json_list

def get_values_by_key(dictionary, key):
    values = []
    for k, v in dictionary.items():
        if k == key:
            if len(v):
                values.append(v)
        elif isinstance(v, dict):
            values.extend(get_values_by_key(v, key))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    values.extend(get_values_by_key(item, key))
    return values

def get_static_text(dictionary):
    values = []
    key = 'textAttributes'
    for k, v in dictionary.items():
        if k == key:    # textAttributes
            if isinstance(v, list): # textAttributes = ['', {value: 'a'}]
                for item in v:
                    if isinstance(item, str):
                        if len(item):
                            values.append(item)
                    elif isinstance(item, dict):
                        if len(item['value']):
                            values.append(item['value'])
            else:
                pass
        elif isinstance(v, dict):
            values.extend(get_static_text(v))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    values.extend(get_static_text(item))
    return values

def jaccard_similarity(str1, str2):
    set1 = set(str1)
    set2 = set(str2)
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    if union == 0:
        return 0
    similarity = intersection / union
    return similarity

class static_graph():
    def __init__(self, json_path):
        self.activity = {}
        self.node = {}
        self.edge = []
        self.edge_json = []    
        
        try:
            with open(json_path, encoding='gbk') as f:  # gbk or utf-8 or GB2312
                j = json.load(f)
        except Exception as e:
            print(e)
            return
        for js_node in j.setdefault('activities', []):
            # self.node['mainactivity'] = dict()
            self.activity[js_node['name']] = js_node
        for js_transition in j.setdefault('transitions', []):
            src = js_transition['scr']
            dst = js_transition['dest']
            trigger = js_transition['trigger']   
            if src in self.activity:
                self.activity[src].setdefault('src', {})[dst] = {'trigger': trigger}
            else:
                self.node.setdefault(src, {}).setdefault('src', {})[dst] = {'trigger': trigger}
            if dst in self.activity:
                self.activity[dst].setdefault('dst', {})[src] = {'trigger': trigger}
            else:
                self.node.setdefault(dst, {}).setdefault('dst', {})[src] = {'trigger': trigger} 
            self.edge.append([src, dst])
            self.edge_json.append(js_transition)
        
    def draw_graph(self, pydotpath=None):
        node = [n for n in self.activity] + [n for n in self.node]
        node_weight = [0 for n in self.activity] + [1 for n in self.node]
        edge = []
        edge_info = []
        for e in self.edge_json:
            src = e['scr']
            dst = e['dest']
            trigger = e['trigger'] 
            info = ','.join(trigger)
            id = []
            for i in [src, dst]:
                id.append(node.index(i))
            if id in edge:
                print('repeat edge', id, info)
            edge.append(id)
            edge_info.append(info)
        draw.draw_networkx(node, edge, node_weight=node_weight, edge_info=edge_info)
        if pydotpath:
            draw.draw_pydotplus(node, edge, node_weight=node_weight, edge_info=edge_info, path=pydotpath)

    def get_text(self):
        res_dict = {}
        for act in self.activity:
            text = get_static_text(self.activity[act])
            res_dict[act] = '。'.join(text)
        return res_dict
    
    def get_fragment_dialog_menu_num(self, to_a=None):
        res_dict = {}
        for act in self.activity:
            frag = get_values_by_key(self.activity[act], 'fragmentClass')
            dial = get_values_by_key(self.activity[act], 'dialogs')
            menu = get_values_by_key(self.activity[act], 'menu')
            try:
                if to_a:
                    act = to_a[act]
                res_dict[act] = [len(frag), len(dial[0]) if len(dial) else 0, \
                                            len(menu[0]) if len(menu) else 0]
            except:
                res_dict[act] = [0, 0, 0]
        return res_dict
    

def change_js_to_json(js_path, json_path):
    with open(js_path, encoding='gbk') as f:
        js_content = f.readlines()
    json_content = js_content[1:]
    with open(json_path, 'w', encoding='gbk') as f:
        f.writelines(json_content)

class dynamic_graph():
    def __init__(self, json_path):
        self.state = {}
        self.state_edge = []
        self.state_edge_json = []
        # self.activity['mainactivity']['state'] = ['123...add', '353...ad8']
        self.activity = {}
        self.edge = []
        try:
            with open(json_path, encoding='gbk') as f:
                j = json.load(f)
        except:
            return
        for js_node in j.setdefault('nodes', []):
            # self.state['353...ad8'] = dict()
            id = js_node['id']
            act = js_node['activity']
            self.state[id] = js_node
            self.activity.setdefault(act, {}).setdefault('state', []).append(id)
        for js_transition in j.setdefault('edges', []):
            src = js_transition['from']
            dst = js_transition['to']
            trigger = js_transition['events']   
            self.state[src].setdefault('src', {})[dst] = {'events': trigger}
            self.state[dst].setdefault('dst', {})[src] = {'events': trigger}
            self.state_edge.append([src, dst])
            self.state_edge_json.append(js_transition)
            src_act = self.state[src]['activity']
            dst_act = self.state[dst]['activity']
            if src_act != dst_act:
                edge = [src_act, dst_act]
                if edge not in self.edge:
                    self.edge.append(edge)
                    self.activity[src_act].setdefault('src', {})[dst_act] = {'trigger': trigger, 'weight': 1}
                    self.activity[dst_act].setdefault('dst', {})[src_act] = {'trigger': trigger, 'weight': 1}
                else:
                    self.activity[src_act]['src'][dst_act]['weight'] += 1
                    self.activity[dst_act]['dst'][src_act]['weight'] += 1
                    self.activity[src_act]['src'][dst_act]['trigger'] += trigger
                    self.activity[dst_act]['dst'][src_act]['trigger'] += trigger
        
    def draw_graph_state(self, pydotpath=None):
        node = [n for n in self.state]
        node_info = [self.state[n]['activity'] for n in self.state]
        edge = []
        edge_info = []
        for e in self.state_edge_json:
            src = e['from']
            dst = e['to']
            trigger = e['events'] 
            info = ' '.join([t['event_type'] for t in trigger])
            id = []
            for i in [src, dst]:
                id.append(node.index(i))
            if id in edge:
                print('repeat edge', id, info)
            edge.append(id)
            edge_info.append(info)
        draw.draw_networkx(node, edge, node_info=node_info, edge_info=edge_info)
        if pydotpath:
            draw.draw_pydotplus(node, edge, node_info=node_info, edge_info=edge_info, path=pydotpath)

    def draw_graph_activity(self, pydotpath=None):
        node = [n for n in self.activity]
        node_info = ['states: ' + str(len(self.activity[n]['state'])) for n in self.activity]
        edge = []
        edge_info = []
        for n in node:
            src = n
            dst_info = self.activity[n].setdefault('src', {})   
            for d in dst_info:
                dst = d
                info = ' '.join([t['event_type'] for t in dst_info[d]['trigger']]) + ' ' + str(dst_info[d]['weight'])
                id = []
                for i in [src, dst]:
                    id.append(node.index(i))
                if id in edge:
                    print('repeat edge', id, info)
                edge.append(id)
                edge_info.append(info)
        draw.draw_networkx(node, edge, node_info=node_info, edge_info=edge_info)
        if pydotpath:
            draw.draw_pydotplus(node, edge, node_info=node_info, edge_info=edge_info, path=pydotpath)

    def draw_graph_activity_and_state(self, pydotpath=None):
        node = [n for n in self.state]
        node_to_sub = {}
        for n in node:
            node_to_sub[n] = self.state[n]['activity']
        edge = []
        edge_info = []
        for e in self.state_edge_json:
            src = e['from']
            dst = e['to']
            trigger = e['events'] 
            info = ' '.join([t['event_type'] for t in trigger])
            id = []
            for i in [src, dst]:
                id.append(node.index(i))
            if id in edge:
                print('repeat edge', id, info)
            edge.append(id)
            edge_info.append(info)
        draw.draw_networkx(node, edge, edge_info=edge_info)
        if pydotpath:
            draw.draw_pydotplus_withsubgraph(node, edge, node_to_sub, edge_info=edge_info, path=pydotpath)

class new_dynamic_graph():
    def __init__(self, json_path, json_folder=None):
        self.state = {}
        self.state_edge = []
        self.state_edge_json = []
        # self.activity['mainactivity']['state'] = ['123...add', '353...ad8']
        self.activity = {}
        self.edge = []
        if json_folder is None:
            self.json_folder = os.path.dirname(json_path)
        else:
            self.json_folder = json_folder
        try:
            with open(json_path, encoding='utf-8') as f:
                j = json.load(f)
        except Exception as e:
            print(e)
            return
        for js_act in j.setdefault('activities', []):
            for js_node in js_act['fragments']:
                # self.state['MainActivity_-1222299932'] = dict()
                id = js_node['signature']
                act = js_node['activity']
                self.state[id] = js_node
                self.activity.setdefault(act, {}).setdefault('state', []).append(id)
                js_node_more = os.path.join(self.json_folder, 'info', id + '.json')
                if os.path.exists(js_node_more):
                    try:
                        with open(js_node_more, encoding='utf-8') as f:
                            j_more = json.load(f)
                            j_more_text = get_values_by_key(j_more, 'viewText')
                            self.state[id].setdefault('more', {})['viewText'] = j_more_text
                            self.state[id].setdefault('more', {})['viewText'] = j_more_text
                    except Exception as e:
                        self.state[id].setdefault('more', {})['viewText'] = []
                else:
                    self.state[id].setdefault('more', {})['viewText'] = []
                    print('补充json文件不存在', id, js_node_more)
        for js_node in self.state:
            webfragments = j.setdefault('webFragments', [])
            if js_node in webfragments:
                self.state[js_node]['isweb'] = True
            else:
                self.state[js_node]['isweb'] = False
        for js_act in j.setdefault('activities', []):
            for js_node in js_act['fragments']:
                for js_transition in js_node['allPaths']:
                    src = id
                    dst = js_transition['target']
                    trigger = [js_transition['action']]  
                    self.state[src].setdefault('src', {})[dst] = {'events': trigger}
                    self.state[dst].setdefault('dst', {})[src] = {'events': trigger}
                    self.state_edge.append([src, dst])
                    js_transition['source'] = id   
                    self.state_edge_json.append(js_transition)
                    src_act = self.state[src]['activity']
                    dst_act = self.state[dst]['activity']
                    if src_act != dst_act:
                        edge = [src_act, dst_act]
                        if edge not in self.edge:
                            self.edge.append(edge)
                            self.activity[src_act].setdefault('src', {})[dst_act] = {'trigger': trigger, 'weight': 1}
                            self.activity[dst_act].setdefault('dst', {})[src_act] = {'trigger': trigger, 'weight': 1}
                        else:
                            self.activity[src_act]['src'][dst_act]['weight'] += 1
                            self.activity[dst_act]['dst'][src_act]['weight'] += 1
                            self.activity[src_act]['src'][dst_act]['trigger'] += trigger
                            self.activity[dst_act]['dst'][src_act]['trigger'] += trigger
    
    def draw_graph_state(self, pydotpath=None):
        node = [n for n in self.state]
        node_info = [self.state[n]['activity'] for n in self.state]
        edge = []
        edge_info = []
        for e in self.state_edge_json:
            src = e['source']
            dst = e['target']
            trigger = e['action'] 
            info = str(trigger)
            id = []
            for i in [src, dst]:
                id.append(node.index(i))
            if id in edge:
                print('repeat edge', id, info)
            edge.append(id)
            edge_info.append(info)
        draw.draw_networkx(node, edge, node_info=node_info, edge_info=edge_info)
        if pydotpath:
            draw.draw_pydotplus(node, edge, node_info=node_info, edge_info=edge_info, path=pydotpath)

    def draw_graph_activity(self, pydotpath=None):
        node = [n for n in self.activity]
        node_info = ['states: ' + str(len(self.activity[n]['state'])) for n in self.activity]
        edge = []
        edge_info = []
        for n in node:
            src = n
            dst_info = self.activity[n].setdefault('src', {})  
            for d in dst_info:
                dst = d
                info = str(dst_info[d]['trigger']) + ' ' + str(dst_info[d]['weight'])
                id = []
                for i in [src, dst]:
                    id.append(node.index(i))
                if id in edge:
                    print('repeat edge', id, info)
                edge.append(id)
                edge_info.append(info)
        draw.draw_networkx(node, edge, node_info=node_info, edge_info=edge_info)
        if pydotpath:
            draw.draw_pydotplus(node, edge, node_info=node_info, edge_info=edge_info, path=pydotpath)

    def draw_graph_activity_and_state(self, pydotpath=None):
        node = [n for n in self.state]
        node_to_sub = {}
        for n in node:
            node_to_sub[n] = self.state[n]['activity']
        edge = []
        edge_info = []
        for e in self.state_edge_json:
            src = e['source']
            dst = e['target']
            trigger = e['action'] 
            info = str(trigger)
            id = []
            for i in [src, dst]:
                id.append(node.index(i))
            if id in edge:
                print('repeat edge', id, info)
            edge.append(id)
            edge_info.append(info)
        draw.draw_networkx(node, edge, edge_info=edge_info)
        if pydotpath:
            draw.draw_pydotplus_withsubgraph(node, edge, node_to_sub, edge_info=edge_info, path=pydotpath)

    def get_text(self):
        res_dict = {}
        similarity = 0.8
        for act in self.activity:
            res_text = []
            for state in self.activity[act]['state']:
                state_text = self.state[state]['more']['viewText']
                for old in res_text:
                    if jaccard_similarity(state_text, old) > similarity:
                        continue    
                res_text.append(state_text) 
            res_flat_text = []
            for i in res_text:
                res_flat_text.extend(i)
            res_activity_text = '。'.join(list(set(res_flat_text)))
            res_dict[act] = res_activity_text
        return res_dict

    def get_webview_num(self, to_a):
        res_dict = {}
        for act in self.activity:
            try:
                if to_a:
                    tact = to_a[act]
                else:
                    tact = act
            except:
                continue
            res_dict[tact] = 0  
            try:
                for state in self.activity[act]['state']:
                    if self.state[state]['isweb']:
                        res_dict[tact] += 1
            except Exception as e:
                print(e)
                pass
        return res_dict

class merge_graph():
    def __init__(self, s_graph: static_graph, d_graph: dynamic_graph):
        self.state = 0          
        self.activity = {}
        self.edge = {}         
        self.s_graph = s_graph
        self.d_graph = d_graph
        for act in s_graph.activity:
            self.activity.setdefault(act, {})['static'] = s_graph.activity[act]
            if act in d_graph.activity:
                self.activity.setdefault(act, {})['dynamic'] = d_graph.activity[act]

        for edge in s_graph.edge:
            src = edge[0]
            dst = edge[1]
            if src in s_graph.activity and dst in s_graph.activity: 
                s_edge_info = s_graph.activity[src]['src'][dst]     
                self.edge.setdefault(src, {})[dst] = {'state': 1,               
                                                      'static': s_edge_info}  
        for edge in d_graph.edge:
            src = edge[0]
            dst = edge[1]
            if src in s_graph.activity and dst in s_graph.activity:
                d_edge_info = d_graph.activity[src]['src'][dst] 
                if dst in self.edge.setdefault(src, {}):   
                    self.edge[src][dst]['dynamic'] = d_edge_info
                    self.edge[src][dst]['state'] = 3               
                else:   
                    self.edge.setdefault(src, {})[dst] = {'state': 2,           
                                                        'dynamic': d_edge_info}  
        
    def draw_graph(self, pydotpath=None):
        node = [n for n in self.activity]
        edge = []
        edge_info = []
        for src in self.edge:
            for dst in self.edge[src]:
                info = self.edge[src][dst]['state']
                edge_info.append(info)
                id = []
                for i in [src, dst]:
                    id.append(node.index(i))
                if id in edge:
                    print('repeat edge', id, info)
                edge.append(id)
        draw.draw_networkx(node, edge, edge_info=edge_info)
        if pydotpath:
            draw.draw_pydotplus(node, edge, edge_info=edge_info, path=pydotpath)

    def get_activity(self):
        res_dict = {}
        for act in self.activity:
            res_dict[act] = act
        return res_dict
        

class new_merge_graph():
    def __init__(self, s_graph: static_graph, d_graph: new_dynamic_graph, act_set=None):
        self.state = 0          
        self.activity = {}
        self.edge = {}          
        self.s_graph = s_graph
        self.d_graph = d_graph

        self.to_a = {}      
        if act_set:
            for act in s_graph.activity:
                if act in act_set:  
                    self.to_a[act] = act     
                    self.activity.setdefault(act, {})['static'] = s_graph.activity[act]
                    if act in d_graph.activity:
                        self.activity.setdefault(act, {})['dynamic'] = d_graph.activity[act]
                else:
                    for ract in act_set:
                        if ract.endswith(act):
                            self.to_a[act] = ract
                            self.activity.setdefault(ract, {})['static'] = s_graph.activity[act]
                            if act in d_graph.activity:
                                self.activity.setdefault(ract, {})['dynamic'] = d_graph.activity[act]
                            break
            for act in d_graph.activity:
                if act in act_set:
                    self.to_a[act] = act
                    self.activity.setdefault(act, {})['dynamic'] = d_graph.activity[act]
                else: 
                    for ract in act_set:
                        if ract.endswith(act):
                            self.to_a[act] = ract
                            self.activity.setdefault(ract, {})['dynamic'] = d_graph.activity[act]
        else:
            for act in s_graph.activity:
                self.to_a[act] = act
                self.activity.setdefault(act, {})['static'] = s_graph.activity[act]
                if act in d_graph.activity:
                    self.activity.setdefault(act, {})['dynamic'] = d_graph.activity[act]
            for act in d_graph.activity:
                self.to_a[act] = act
                self.activity.setdefault(act, {})['dynamic'] = d_graph.activity[act]
                
        for edge in s_graph.edge:
            src = edge[0]
            dst = edge[1]
            if src in self.to_a and dst in self.to_a and src in s_graph.activity and dst in s_graph.activity:
                s_edge_info = s_graph.activity[src]['src'][dst]    
                self.edge.setdefault(self.to_a[src], {})[self.to_a[dst]] = {'state': 1,      
                                                      'static': s_edge_info}    
        for edge in d_graph.edge:
            src = edge[0]
            dst = edge[1]
            if src in self.to_a and dst in self.to_a and self.to_a[src] in self.activity and self.to_a[dst] in self.activity:    
                d_edge_info = d_graph.activity[src]['src'][dst]
                if self.to_a[dst] in self.edge.setdefault(self.to_a[src], {}):    
                    self.edge[self.to_a[src]][self.to_a[dst]]['dynamic'] = d_edge_info
                    self.edge[self.to_a[src]][self.to_a[dst]]['state'] = 3              
                else:  
                    self.edge.setdefault(self.to_a[src], {})[self.to_a[dst]] = {'state': 2,             
                                                        'dynamic': d_edge_info} 
        
    def draw_graph(self, pydotpath=None):
        node = [n for n in self.activity]
        edge = []
        edge_info = []
        for src in self.edge:
            for dst in self.edge[src]:
                info = self.edge[src][dst]['state']
                edge_info.append(info)
                id = []
                for i in [src, dst]:
                    id.append(node.index(i))
                if id in edge:
                    print('repeat edge', id, info)
                edge.append(id)
        draw.draw_networkx(node, edge, edge_info=edge_info)
        if pydotpath:
            draw.draw_pydotplus(node, edge, edge_info=edge_info, path=pydotpath)

    def get_text(self):
        static_dict = {}
        dynamic_dict = {}
        res_dict = {}
        static_text = self.s_graph.get_text()
        dynamic_text = self.d_graph.get_text()
        for act in static_text:
            static_dict[self.to_a[act]] = static_text[act]
            if act in self.to_a:
                res_dict[self.to_a[act]] = res_dict.setdefault(self.to_a[act], "") + static_text[act]
        for act in dynamic_text:
            dynamic_dict[self.to_a[act]] = dynamic_text[act]
            if act in self.to_a:
                res_dict[self.to_a[act]] = res_dict.setdefault(self.to_a[act], "") + dynamic_text[act]
        for act in self.activity:
            if act not in static_dict:
                static_dict[act] = ""
            if act not in dynamic_dict:
                dynamic_dict[act] = ""
            if act not in res_dict:
                res_dict[act] = ""
        return static_dict, dynamic_dict, res_dict
    
    def get_info(self):
        node_list = []
        stext_dict, dtext_dict, mtext_dict = self.get_text()
        fdm = self.s_graph.get_fragment_dialog_menu_num(self.to_a)
        web = self.d_graph.get_webview_num(self.to_a)
        for act in self.activity:
            act_dict = {}
            act_dict['activity'] = act
            act_dict['stext'] = stext_dict[act]
            act_dict['dtext'] = dtext_dict[act]
            act_dict['mtext'] = mtext_dict[act]
            act_dict['fragment'] = fdm.setdefault(act, [0, 0, 0])[0]
            act_dict['dialog'] = fdm.setdefault(act, [0, 0, 0])[1]
            act_dict['menu'] = fdm.setdefault(act, [0, 0, 0])[2]
            act_dict['webview'] = web.setdefault(act, 0)
            node_list.append(act_dict)
        act_id = [i['activity'] for i in node_list]
        edge_list_src = []     
        edge_list_dst = []     
        edge_list_type = []     
        for src in self.activity:
            src_id = act_id.index(src)
            for dst in self.edge.setdefault(src, {}):
                dst_id = act_id.index(dst)
                edge_type = self.edge[src][dst]['state']
                edge_list_src.append(src_id)
                edge_list_dst.append(dst_id)
                edge_list_type.append(edge_type)
        edge_list = [edge_list_src, edge_list_dst]
        return node_list, edge_list, edge_list_type 


if __name__ == '__main__':
    
    import pandas as pd
    target_folder = [r'D:\Code\ML\APP\sample\正常',
                     r'D:\Code\ML\APP\sample\赌博',
                     r'D:\Code\ML\APP\sample\色情',
                     r'D:\Code\ML\APP\sample\欺诈']
    all_text_list = []
    for i in range(len(target_folder)):
        for folder in os.listdir(target_folder[i]):
            folder_path = os.path.join(target_folder[i], folder)
            androguard_path = os.path.join(folder_path, 'androguard.txt')
            icon_path = os.path.join(folder_path, 'icon.png')
            static_path =  os.path.join(folder_path, 'static_graph.json')
            dynamic_path = os.path.join(folder_path, 'dynamic_graph')
            text_path = os.path.join(folder_path, 'text.txt')
            [name, pack, perm, acti, serv, rece, prov, cert] = [None, None, None, None, None, None, None, None]
            try:
                # androguard.txt
                with open(androguard_path, encoding='utf-8') as f:
                    [name, pack, perm, acti, serv, rece, prov, cert] = [kr.strip() for kr in f.readlines()]
                    perm = perm.split(',')
                    acti = acti.split(',')
                    serv = serv.split(',')
                    rece = rece.split(',')
                    prov = prov.split(',')
            except Exception as e:
                print(e)
                print('unable to get androguard info', folder_path)
            try:
                # icon.png
                with open(icon_path, 'rb') as f:
                    icon = f.read()
            except Exception as e:
                print(e)
                print('unable to get icon info', folder_path)
            try:
                # static_graph.json
                stg = static_graph(static_path)
            except Exception as e:
                print(e)
                print('unable to get static graph', folder_path)
                stg = None
            try:
                dtg = new_dynamic_graph(os.path.join(dynamic_path, pack + '.json'), dynamic_path)
            except Exception as e:
                print(e)
                print('unable to get dynamic graph', folder_path)
                dtg = None
            try:
                mtg = new_merge_graph(stg, dtg, acti)
                stext, dtext, mtext = mtg.get_text()
                for t in mtext:
                    all_text_list.append([i, mtext[t]])
            except Exception as e:
                print(e)
                print('unable to merge graph', folder_path)
        
        tl_csv = pd.DataFrame(all_text_list)
        tl_csv.to_csv('textlist.csv', index=None, header=None)
