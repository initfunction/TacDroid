import os

import pandas as pd

import align as al
import embedding as em
import prediction as lp
import globalfeature as gl
import classification as cl

def load_dataset(target_folder):
    info_dict = []
    for i in range(len(target_folder)):
        dic = {}
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
                # with open(icon_path, 'rb') as f:
                #     icon = f.read()
                icon = icon_path
            except Exception as e:
                print(e)
                print('unable to get icon info', folder_path)
            try:
                # static_graph.json
                stg = al.static_graph(static_path)
            except Exception as e:
                print(e)
                print('unable to get static graph', folder_path)
                stg = None
            try:
                dtg = al.new_dynamic_graph(os.path.join(dynamic_path, pack + '.json'), dynamic_path)
            except Exception as e:
                print(e)
                print('unable to get dynamic graph', folder_path)
                dtg = None
            try:
                mtg = al.new_merge_graph(stg, dtg, acti)
                mtext = mtg.get_text()
            except Exception as e:
                print(e)
                print('unable to merge graph', folder_path)
                mtg = None

            dic[pack] = {'perm': perm, 'acti': acti, 'name': name, 'pack': pack, 'perm': perm, 'acti': acti, 'serv': serv, 'rece': rece, 'prov': prov, 'cert': cert,
                         'stg': stg, 'dtg': dtg, 'mtg': mtg,
                         'icon': icon,
                         'cls': i}
        info_dict.append(dic)
    return info_dict

def split_dataset(info_dict, test_size=0.15):
    train_dict = []
    train_label = []
    test_dict = []
    test_label = []
    test_size = 0.15
    for i in range(len(info_dict)):
        traindic = {}
        testdic = {}
        dic = info_dict[i]
        total_len = len(dic)
        train_len = total_len * (1 - test_size)
        l = 0
        for q in dic:
            if l < train_len:
                traindic[q] = dic[q]
                train_label.append(i)
            else:
                testdic[q] = dic[q]
                test_label.append(i)
            l += 1
        train_dict.append(traindic)
        test_dict.append(testdic)
    return train_dict, train_label, test_dict, test_label


target_folder = [r'APP\sample\正常',
                 r'APP\sample\赌博',
                 r'APP\sample\色情',
                 r'APP\sample\欺诈']
info_dict = load_dataset(target_folder)
train_dict, train_label, test_dict, test_label = split_dataset(info_dict, test_size=0.15)

import pickle
with open('base/train_feature_list.pickle', 'rb') as f:
    train_feature_list = pickle.load(f)  
with open('base/train_edge_list.pickle', 'rb') as f:
    train_edge_list = pickle.load(f)  
with open('base/train_edge_list_type.pickle', 'rb') as f:
    train_edge_list_type = pickle.load(f)
with open('base/train_label.pickle', 'rb') as f:
    train_label = pickle.load(f)
with open('base/test_feature_list.pickle', 'rb') as f:
    test_feature_list = pickle.load(f)  
with open('base/test_edge_list.pickle', 'rb') as f:
    test_edge_list = pickle.load(f)  
with open('base/test_edge_list_type.pickle', 'rb') as f:
    test_edge_list_type = pickle.load(f)
with open('base/test_label.pickle', 'rb') as f:
    test_label = pickle.load(f)

graph_num = len(train_feature_list)
input_dim = len(train_feature_list[0][0])
hidden_channels = 128
out_channels = 64
epoch = 50
batch = 32
trainloader = lp.build_data_loader(train_feature_list, train_edge_list, batch=batch)
mod = lp.MyGravityAEGCN(input_dim, hidden_channels, out_channels, 0)
mod = lp.MyGravityAEGCN.load_model(r'model\lp_model_128_64_50_32.pth', input_dim, hidden_channels, out_channels)

h_threshold = 0.9
l_threshold = 0.1
predictloader = lp.build_data_loader(train_feature_list, train_edge_list, None, batch=1, shuffle=False)
train_lp_result_edge_list = []
train_lp_result_node_list = []
fl = 0
for pre in predictloader:
    if pre.x.shape[0]:
        this_result_edge_list = [[], []]
        p_all_edge, p_score = mod.predict_score(pre)
        s_edge = [[train_edge_list[0][i], train_edge_list[1][i]] for i in range(len(train_edge_list[0])) if train_edge_list_type[i] == 1]
        d_edge = [[train_edge_list[0][i], train_edge_list[1][i]] for i in range(len(train_edge_list[0])) if train_edge_list_type[i] != 1]
        for ii in range(len(p_all_edge[0])):
            src = p_all_edge[0][ii]
            dst = p_all_edge[1][ii]
            if [src, dst] in d_edge:   
                this_result_edge_list[0].append(src)
                this_result_edge_list[1].append(dst)
            elif p_score[ii] > h_threshold and [src, dst] not in s_edge: 
                this_result_edge_list[0].append(src)
                this_result_edge_list[1].append(dst)
            elif p_score[ii] < h_threshold and [src, dst] in s_edge:  
                pass
        train_lp_result_edge_list.append(this_result_edge_list)
        train_lp_result_node_list.append(train_feature_list[fl])
    else:
        train_lp_result_edge_list.append([[0], [0]])
        train_lp_result_node_list.append([[0 for i in range(input_dim)]])
    fl += 1

num_features = input_dim    
hidden_channels = 256
out_channels = 64
num_classes = 4
epoch = 100
batch = 64
clloader_final = cl.build_data_loader(train_lp_result_node_list, train_lp_result_edge_list, train_label, batch)
clmod_final = cl.Func_GCN(num_features, hidden_channels, out_channels, num_classes)
clmod_final.train(clloader_final, epoch)
clmod_final.save_model(r'model\gcn_final_model_256_64_100_64.pth')
res_list = []
clpreloader = cl.build_data_loader(train_lp_result_node_list, train_lp_result_edge_list, None, batch=1, shuffle=False)
for pre in clpreloader:
    res_list.append(clmod_final.predict(pre))
sum = 0
for i in range(len(train_label)):
    sum += train_label[i] == int(res_list[i])
print(sum / len(train_label))
predictloader = lp.build_data_loader(test_feature_list, test_edge_list, None, batch=1, shuffle=False)
test_lp_result_edge_list = []
test_lp_result_node_list = []
fl = 0
for pre in predictloader:
    if pre.x.shape[0]:
        this_result_edge_list = [[], []]
        p_all_edge, p_score = mod.predict_score(pre)
        s_edge = [[test_edge_list[0][i], test_edge_list[1][i]] for i in range(len(test_edge_list[0])) if test_edge_list_type[i] == 1]
        d_edge = [[test_edge_list[0][i], test_edge_list[1][i]] for i in range(len(test_edge_list[0])) if test_edge_list_type[i] != 1]
        for ii in range(len(p_all_edge[0])):
            src = p_all_edge[0][ii]
            dst = p_all_edge[1][ii]
            if [src, dst] in d_edge:   
                this_result_edge_list[0].append(src)
                this_result_edge_list[1].append(dst)
            elif p_score[ii] > h_threshold and [src, dst] not in s_edge: 
                this_result_edge_list[0].append(src)
                this_result_edge_list[1].append(dst)
            elif p_score[ii] < h_threshold and [src, dst] in s_edge: 
                pass
        test_lp_result_edge_list.append(this_result_edge_list)
        test_lp_result_node_list.append(test_feature_list[fl])
    else:
        test_lp_result_edge_list.append([[0], [0]])
        test_lp_result_node_list.append([[0 for i in range(input_dim)]])
    fl += 1
clloader_final = cl.build_data_loader(test_lp_result_node_list, test_lp_result_edge_list, test_label, batch)
predicted_labels, acc = clmod_final.test(clloader_final)
print('final acc: ', acc)
result_pd = pd.DataFrame({'Column1': predicted_labels, 'Column2': test_label})
result_pd.to_csv('base/result/final.csv')

num_features = input_dim   
hidden_channels = 256
out_channels = 64
num_classes = 4
epoch = 100
batch = 32
clloader_final = cl.build_data_loader(train_lp_result_node_list, train_edge_list, train_label, batch)
clmod_seed = cl.Func_GCN(num_features, hidden_channels, out_channels, num_classes)
clmod_seed.train(clloader_final, epoch)
clmod_seed.save_model(r'model\gcn_seed_model_256_64_100_32.pth')
res_list = []
clpreloader = cl.build_data_loader(train_lp_result_node_list, train_edge_list, None, batch=1, shuffle=False)

for pre in clpreloader:
    res_list.append(clmod_final.predict(pre))
sum = 0
for i in range(len(train_label)):
    sum += train_label[i] == int(res_list[i])
print(sum / len(train_label))
predictloader = lp.build_data_loader(test_lp_result_node_list, test_edge_list, None, batch=1, shuffle=False)
clloader_final = cl.build_data_loader(test_lp_result_node_list, test_edge_list, test_label, batch)
predicted_labels, acc = clmod_final.test(clloader_final)
print('seed acc: ', acc)
result_pd = pd.DataFrame({'Column1': predicted_labels, 'Column2': test_label})
result_pd.to_csv('base/result/seed.csv')

gf_dim = 192
gf_input_dim = input_dim + gf_dim  
with open('base/train_GF_result_node_list.pickle', 'rb') as f:
    train_GF_result_node_list = pickle.load(f)  
with open('base/test_GF_result_node_list.pickle', 'rb') as f:
    test_GF_result_node_list = pickle.load(f)  

try:
    num_features = gf_input_dim   
    hidden_channels = 256
    out_channels = 64
    num_classes = 4
    epoch = 100
    batch = 32
    clloader_final = cl.build_data_loader(train_GF_result_node_list, train_edge_list, train_label, batch)
    clmod_seed = cl.Func_GCN(num_features, hidden_channels, out_channels, num_classes)
    clmod_seed.train(clloader_final, epoch)
    clmod_seed.save_model(r'D:\t\tct\gcn_seed_GF_model_256_64_100_32.pth')
    res_list = []
    clpreloader = cl.build_data_loader(train_GF_result_node_list, train_edge_list, None, batch=1, shuffle=False)
    for pre in clpreloader:
        res_list.append(clmod_final.predict(pre))
    sum = 0
    for i in range(len(train_label)):
        sum += train_label[i] == int(res_list[i])
    print(sum / len(train_label))
    predictloader = lp.build_data_loader(test_GF_result_node_list, test_edge_list, None, batch=1, shuffle=False)
    clloader_final = cl.build_data_loader(test_GF_result_node_list, test_edge_list, test_label, batch)
    predicted_labels, acc = clmod_final.test(clloader_final)
    print('seed+GF acc: ', acc)
    result_pd = pd.DataFrame({'Column1': predicted_labels, 'Column2': test_label})
    result_pd.to_csv('base/result/seed+GF.csv')

except:
    pass

gf_dim = 192
gf_input_dm = input_dim + gf_dim    # 132 + 192
train_GF = [i[132:] for i in train_GF_result_node_list]
test_GF = [i[132:] for i in test_GF_result_node_list]

train_lp_GF_result_node_list = [train_lp_result_node_list[i] + train_GF[i] for i in range(len(train_GF))]
test_lp_GF_result_node_list = [test_lp_result_node_list[i] + test_GF[i] for i in range(len(test_GF))]

num_features = gf_input_dim   
hidden_channels = 256
out_channels = 64
num_classes = 4
epoch = 100
batch = 32
clloader_final = cl.build_data_loader(train_lp_GF_result_node_list, train_edge_list, train_label, batch)
clmod_seed = cl.Func_GCN(num_features, hidden_channels, out_channels, num_classes)
clmod_seed.train(clloader_final, epoch)
clmod_seed.save_model(r'model\gcn_final_GF_model_256_64_100_32.pth')
res_list = []
clpreloader = cl.build_data_loader(train_lp_GF_result_node_list, train_edge_list, None, batch=1, shuffle=False)
for pre in clpreloader:
    res_list.append(clmod_final.predict(pre))
sum = 0
for i in range(len(train_label)):
    sum += train_label[i] == int(res_list[i])
print(sum / len(train_label))
predictloader = lp.build_data_loader(test_lp_GF_result_node_list, test_edge_list, None, batch=1, shuffle=False)
clloader_final = cl.build_data_loader(test_lp_GF_result_node_list, test_edge_list, test_label, batch)
predicted_labels, acc = clmod_final.test(clloader_final)
print('final+GF acc: ', acc)
result_pd = pd.DataFrame({'Column1': predicted_labels, 'Column2': test_label})
result_pd.to_csv('base/result/final+GF.csv')
