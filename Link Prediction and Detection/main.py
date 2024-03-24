import os

import pandas as pd
import numpy as np

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
                stext, dtext, mtext = mtg.get_text()
            except Exception as e:
                print(e)
                print('unable to merge graph', folder_path)
                mtg = None

            dic[pack] = {'perm': perm, 'acti': acti, 'name': name, 'pack': pack, 'serv': serv, 'rece': rece, 'prov': prov, 'cert': cert,
                         'stg': stg, 'dtg': dtg, 'mtg': mtg,
                         'icon': icon,
                         'cls': i}
        info_dict.append(dic)
    return info_dict

def build_dataset(info_dict, test_size=0.15):
    train_dict = []
    train_label = []
    test_dict = []
    test_label = []
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

def split_dataset_k_fold(data_list: list, label:list, k=5, split_list=None):
    features = [[data_list[l][i] for l in range(len(data_list))] for i in range(len(data_list[0]))]
    result = [[] for i in range(k)]   
    if not split_list:
        cls = 4   
        first_occurrence = [label.index(i) for i in range(cls)]
        last_occurrence = [first_occurrence[i + 1] if i != 3 else len(label) for i in range(cls)]
        chosen_index_list = [[] for i in range(k)]
    else:
        chosen_index_list = split_list
    for i in range(cls):
        f = first_occurrence[i]
        l = last_occurrence[i]
        numbers = np.arange(f, l)
        split_res = np.array_split(np.random.choice(numbers, size=len(numbers), replace=False), k)
        for c in range(k):
            chosen_index_list[c].extend(split_res[c].tolist())
    
    for i in range(k):
        train_f, train_l = [], []
        for t in range(k):
            if t != i:
                train_f.extend(features[l] for l in chosen_index_list[t])
                train_l.extend(label[l] for l in chosen_index_list[t])
        test_f = [features[l] for l in chosen_index_list[i]]
        test_l = [label[l] for l in chosen_index_list[i]]
        train_df, test_df = [[train_f[i][l] for i in range(len(train_f))] for l in range(len(train_f[0]))], \
                            [[test_f[i][l] for i in range(len(test_f))] for l in range(len(test_f[0]))]
        result[i] = [[train_df, train_l], [test_df, test_l]]
    return result, chosen_index_list

def check_accuracy(csv_path):
    csv = pd.read_csv(csv_path)
    result = csv['Column1'].to_list()
    label = csv['Column2'].to_list()
    result_2 = [1 if i > 0 else 0 for i in result]
    label_2 = [1 if i > 0 else 0 for i in label]
    correct = sum(r == l for r, l in zip(result, label)) / len(label)
    correct_2 = sum(r == l for r, l in zip(result_2, label_2)) / len(label)

    return correct, correct_2


feature_node_s_list = []             
feature_node_d_list = []               
feature_node_sd_list = []              
feature_node_s_GF_list = []             
feature_node_d_GF_list = []           
feature_node_sd_GF_list = []       
feature_edge_s_list = []     
feature_edge_d_list = []               
feature_edge_sd_list = []             
feature_edgetype_sd_list = []           
label_list = []                         

# for i in range(len(all_dict)):
#     dic = all_dict[i]
#     for j in dic:      
#         gf = gl.get_global_feature(dic[j])  
#         mtg = dic[j]['mtg']
#         if mtg:
#             node_list, edge_list, edge_list_type = mtg.get_info()
#             sfeature_list, dfeature_list, mfeature_list = em.get_feature_sdm(node_list)    
#             feature_node_s_list.append(sfeature_list)
#             feature_node_s_GF_list.append([sfeature_list[i] + gf for i in range(len(mfeature_list))])
#             feature_node_d_list.append(dfeature_list)
#             feature_node_d_GF_list.append([dfeature_list[i] + gf for i in range(len(mfeature_list))])
#             feature_node_sd_list.append(mfeature_list)
#             feature_node_sd_GF_list.append([mfeature_list[i] + gf for i in range(len(mfeature_list))])
#             feature_edge_s_list.append([[edge_list[0][i] for i in range(len(edge_list_type)) if edge_list_type[i] in [1, 3]], [edge_list[1][i] for i in range(len(edge_list_type)) if edge_list_type[i] in [1, 3]]])
#             feature_edge_d_list.append([[edge_list[0][i] for i in range(len(edge_list_type)) if edge_list_type[i] in [2, 3]], [edge_list[1][i] for i in range(len(edge_list_type)) if edge_list_type[i] in [2, 3]]])
#             feature_edge_sd_list.append(edge_list)
#             feature_edgetype_sd_list.append(edge_list_type)
#             label_list.append(i)

# # 存储~
# import pickle
# with open('base/feature_node_s_list.pickle', 'wb') as f:
#     pickle.dump(feature_node_s_list, f)  
# with open('base/feature_node_s_GF_list.pickle', 'wb') as f:
#     pickle.dump(feature_node_s_GF_list, f)  
# with open('base/feature_node_d_list.pickle', 'wb') as f:
#     pickle.dump(feature_node_d_list, f) 
# with open('base/feature_node_d_GF_list.pickle', 'wb') as f:
#     pickle.dump(feature_node_d_GF_list, f) 
# with open('base/feature_node_sd_list.pickle', 'wb') as f:
#     pickle.dump(feature_node_sd_list, f)        
# with open('base/feature_node_sd_GF_list.pickle', 'wb') as f:
#     pickle.dump(feature_node_sd_GF_list, f)    
# with open('base/feature_edge_s_list.pickle', 'wb') as f:
#     pickle.dump(feature_edge_s_list, f)    
# with open('base/feature_edge_d_list.pickle', 'wb') as f:
#     pickle.dump(feature_edge_d_list, f)    
# with open('base/feature_edge_sd_list.pickle', 'wb') as f:
#     pickle.dump(feature_edge_sd_list, f)    
# with open('base/feature_edgetype_sd_list.pickle', 'wb') as f:
#     pickle.dump(feature_edgetype_sd_list, f)    
# with open('base/label_list.pickle', 'wb') as f:
#     pickle.dump(label_list, f)    

import pickle
with open('base/feature_node_s_list.pickle', 'rb') as f:
    feature_node_s_list = pickle.load(f)  
with open('base/feature_node_s_GF_list.pickle', 'rb') as f:
    feature_node_s_GF_list = pickle.load(f)  
with open('base/feature_node_d_list.pickle', 'rb') as f:
    feature_node_d_list = pickle.load(f)  
with open('base/feature_node_d_GF_list.pickle', 'rb') as f:
    feature_node_d_GF_list = pickle.load(f)  
with open('base/feature_node_sd_list.pickle', 'rb') as f:
    feature_node_sd_list = pickle.load(f)
with open('base/feature_node_sd_GF_list.pickle', 'rb') as f:
    feature_node_sd_GF_list = pickle.load(f)
with open('base/feature_edge_s_list.pickle', 'rb') as f:
    feature_edge_s_list = pickle.load(f)  
with open('base/feature_edge_d_list.pickle', 'rb') as f:
    feature_edge_d_list = pickle.load(f)  
with open('base/feature_edge_sd_list.pickle', 'rb') as f:
    feature_edge_sd_list = pickle.load(f)
with open('base/feature_edgetype_sd_list.pickle', 'rb') as f:
    feature_edgetype_sd_list = pickle.load(f)
with open('base/label_list.pickle', 'rb') as f:
    label_list = pickle.load(f)


def one_round_sd(train_feature_node_sd_list,      
                 train_feature_node_sd_GF_list,     
                 train_feature_edge_sd_list,     
                 train_edgetype_sd_list,        
                 train_label,                   

                 test_feature_node_sd_list, 
                 test_feature_node_sd_GF_list,
                 test_feature_edge_sd_list, 
                 test_edgetype_sd_list,
                 test_label,

                 train_feature_edge_after_lp_list = None,  
                 train_edgetype_after_lp_list = None,    
                 test_feature_edge_after_lp_list = None, 
                 test_edgetype_after_lp_list = None
                 ):
    res_acc = {'lp': 0, 'lp_GF': 0, 'seed': 0, 'seed_GF': 0}
    ###########################################
    # LP_train

    lp_graph_num = len(train_feature_node_sd_list)
    lp_input_dim = len(train_feature_node_sd_list[0][0])
    lp_hidden_channels = 128
    lp_out_channels = 64
    lp_epoch = 50
    lp_batch = 32
    lp_name = '_'.join([str(lp_hidden_channels), str(lp_out_channels), str(lp_epoch), str(lp_batch)])
    lp_trainloader = lp.build_data_loader(train_feature_node_sd_list, train_feature_edge_sd_list, batch=lp_batch)
    lp_mod = lp.MyGravityAEGCN(lp_input_dim, lp_hidden_channels, lp_out_channels, 0)
    lp_mod.train(lp_trainloader, lp_epoch)
    lp_mod.save_model(r'model/temp/lp_model_' + lp_name + '.pth')
    lp_mod = lp.MyGravityAEGCN.load_model(r'model/temp/lp_model_' + lp_name + '.pth', lp_input_dim, lp_hidden_channels, lp_out_channels)

    ###########################################
    # lp_train_graph    -> GC_final(t)  -> train_class_0  
    # lp_test_graph     -> GC_final     -> test_class_0   
    h_threshold = 0.408113  # 0.524964
    l_threshold = 0.170088  # 0.202414  # 0.174192
    predictloader = lp.build_data_loader(train_feature_node_sd_list, train_feature_edge_sd_list, None, batch=1, shuffle=False)
    train_feature_edge_after_lp_list = []
    fl = 0
    sco_oo = []
    sco_label = []
    for pre in predictloader:
        if pre.x.shape[0]:
            this_result_edge_list = [[], []]
            p_all_edge, p_score = lp_mod.predict_score(pre)
            temp_edge_couple = [[train_feature_edge_sd_list[fl][0][i], train_feature_edge_sd_list[fl][1][i]] for i in range(len(train_feature_edge_sd_list[fl][0]))]
            temp_all_couple = [[int(p_all_edge[0][i]), int(p_all_edge[1][i])] for i in range(len(p_all_edge[0]))]
            sco_oo.extend([i.item() for i in p_score])
            sco_label.extend([1 if temp_all_couple[i] in temp_edge_couple else 0 for i in range(len(temp_all_couple))])

            s_edge = [[train_feature_edge_sd_list[fl][0][i], train_feature_edge_sd_list[fl][1][i]] for i in range(len(train_feature_edge_sd_list[fl][0])) if train_edgetype_sd_list[fl][i] == 1]
            d_edge = [[train_feature_edge_sd_list[fl][0][i], train_feature_edge_sd_list[fl][1][i]] for i in range(len(train_feature_edge_sd_list[fl][0])) if train_edgetype_sd_list[fl][i] != 1]
            for ii in range(len(p_all_edge[0])):
                src = int(p_all_edge[0][ii])
                dst = int(p_all_edge[1][ii])
                if [src, dst] in d_edge:    
                    this_result_edge_list[0].append(src)
                    this_result_edge_list[1].append(dst)
                elif p_score[ii] > h_threshold and [src, dst] not in s_edge: 
                    this_result_edge_list[0].append(src)
                    this_result_edge_list[1].append(dst)
                elif p_score[ii] < l_threshold and [src, dst] in s_edge: 
                    pass
                elif [src, dst] in s_edge:  
                    this_result_edge_list[0].append(src)
                    this_result_edge_list[1].append(dst)
            # lp_result_edge_list.append(mod.predict(pre, threshold=0.5))
            train_feature_edge_after_lp_list.append(this_result_edge_list)
        else:
            train_feature_edge_after_lp_list.append([[0], [0]])
        fl += 1

    csv_sco_oo = pd.DataFrame([[sco_oo[i], sco_label[i]] for i in range(len(sco_oo))])
    csv_sco_oo.to_csv('base/lp_threshold.csv', header=None, index=None)
    length = len(sco_oo)
    sco_oo.sort()
    print(sco_oo[length//4], sco_oo[length//4*3])
    
    cl_lp_num_features = lp_input_dim    
    cl_hidden_channels = 256
    cl_out_channels = 64
    cl_num_classes = 4
    cl_epoch = 100
    cl_batch = 64
    cl_name = '_'.join([str(cl_hidden_channels), str(cl_out_channels), str(cl_epoch), str(cl_batch)])
    clloader_lp = cl.build_data_loader(train_feature_node_sd_list, train_feature_edge_after_lp_list, train_label, cl_batch)
    clmod_lp = cl.MyGCN(cl_lp_num_features, cl_hidden_channels, cl_out_channels, cl_num_classes)
    # clmod_lp.train(clloader_lp, cl_epoch)
    # clmod_lp.save_model(r'model/temp/gcn_lp_model_' + cl_name + '.pth')
    clmod_lp = cl.MyGCN.load_model(r'model/temp/gcn_lp_model_' + cl_name + '.pth', cl_lp_num_features, cl_hidden_channels, cl_out_channels, cl_num_classes)
    cl_lp_res_list = []
    clpreloader_lp = cl.build_data_loader(train_feature_node_sd_list, train_feature_edge_after_lp_list, None, batch=1, shuffle=False)
    for pre in clpreloader_lp:
        cl_lp_res_list.append(clmod_lp.predict(pre))
    sum = 0
    for i in range(len(train_label)):
        sum += train_label[i] == int(cl_lp_res_list[i])
    print(sum / len(train_label))
    predictloader_lp = lp.build_data_loader(test_feature_node_sd_list, test_feature_edge_sd_list, None, batch=1, shuffle=False)
    test_feature_edge_after_lp_list = []
    fl = 0
    for pre in predictloader_lp:
        if pre.x.shape[0]:
            this_result_edge_list = [[], []]
            p_all_edge, p_score = lp_mod.predict_score(pre)
            s_edge = [[test_feature_edge_sd_list[fl][0][i], test_feature_edge_sd_list[fl][1][i]] for i in range(len(test_feature_edge_sd_list[fl][0])) if test_edgetype_sd_list[fl][i] == 1]
            d_edge = [[test_feature_edge_sd_list[fl][0][i], test_feature_edge_sd_list[fl][1][i]] for i in range(len(test_feature_edge_sd_list[fl][0])) if test_edgetype_sd_list[fl][i] != 1]
            for ii in range(len(p_all_edge[0])):
                src = int(p_all_edge[0][ii])
                dst = int(p_all_edge[1][ii])
                if [src, dst] in d_edge:  
                    this_result_edge_list[0].append(src)
                    this_result_edge_list[1].append(dst)
                elif p_score[ii] > h_threshold and [src, dst] not in s_edge: 
                    this_result_edge_list[0].append(src)
                    this_result_edge_list[1].append(dst)
                elif p_score[ii] < l_threshold and [src, dst] in s_edge: 
                    pass
                elif [src, dst] in s_edge: 
                    this_result_edge_list[0].append(src)
                    this_result_edge_list[1].append(dst)
            test_feature_edge_after_lp_list.append(this_result_edge_list)
        else:
            test_feature_edge_after_lp_list.append([[0], [0]])
        fl += 1
    clloader_lp_t = cl.build_data_loader(test_feature_node_sd_list, test_feature_edge_after_lp_list, test_label, cl_batch, shuffle=False)
    predicted_labels, acc = clmod_lp.test(clloader_lp_t)
    print('lp: ', acc)
    result_pd = pd.DataFrame({'Column1': predicted_labels, 'Column2': test_label})
    result_pd.to_csv('base/result/lp.csv')
    res_acc['lp'] = acc

    ###########################################
    # train_graph   -> GC_seed(t)   -> train_class_1    
    # test_graph    -> GC_seed      -> test_class_1     

    cl_seed_num_features = lp_input_dim    
    clloader_seed = cl.build_data_loader(train_feature_node_sd_list, train_feature_edge_sd_list, train_label, cl_batch)
    clmod_seed = cl.MyGCN(cl_seed_num_features, cl_hidden_channels, cl_out_channels, cl_num_classes)
    clmod_seed.train(clloader_seed, cl_epoch)
    clmod_seed.save_model(r'model/temp/gcn_seed_model_' + cl_name + '.pth')
    clmod_seed = cl.MyGCN.load_model(r'model/temp/gcn_seed_model_' + cl_name + '.pth', cl_seed_num_features, cl_hidden_channels, cl_out_channels, cl_num_classes)
    cl_seed_res_list = []
    clpreloader_seed = cl.build_data_loader(train_feature_node_sd_list, train_feature_edge_sd_list, None, batch=1, shuffle=False)

    for pre in clpreloader_seed:
        cl_seed_res_list.append(clmod_seed.predict(pre))
    sum = 0
    for i in range(len(train_label)):
        sum += train_label[i] == int(cl_seed_res_list[i])
    print(sum / len(train_label))
    clloader_seed_t = cl.build_data_loader(test_feature_node_sd_list, test_feature_edge_sd_list, test_label, cl_batch, shuffle=False)
    predicted_labels, acc = clmod_seed.test(clloader_seed_t)
    print('seed: ', acc)
    result_pd = pd.DataFrame({'Column1': predicted_labels, 'Column2': test_label})
    result_pd.to_csv('base/result/seed.csv')
    res_acc['seed'] = acc

    ###########################################
    # train_graph + train_GF -> GC_seed_GF(t) -> train_class_2  
    # test_graph  + test_GF  -> GC_seed_GF    -> test_class_2   

    gf_dim = 192
    gf_input_dim = lp_input_dim + gf_dim   

    cl_seed_GF_num_features = gf_input_dim
    clloader_seed_GF = cl.build_data_loader(train_feature_node_sd_GF_list, train_feature_edge_sd_list, train_label, cl_batch)
    clmod_seed_GF = cl.MyGCN(cl_seed_GF_num_features, cl_hidden_channels, cl_out_channels, cl_num_classes)
    clmod_seed_GF.train(clloader_seed_GF, cl_epoch)
    clmod_seed_GF.save_model(r'model/temp/gcn_seed_GF_model' + cl_name + '.pth')
    clmod_seed_GF = cl.MyGCN.load_model(r'model/temp/gcn_seed_GF_model' + cl_name + '.pth', cl_seed_GF_num_features, cl_hidden_channels, cl_out_channels, cl_num_classes)
    cl_seed_GF_res_list = []
    clpreloader_seed_GF = cl.build_data_loader(train_feature_node_sd_GF_list, train_feature_edge_sd_list, None, batch=1, shuffle=False)
    for pre in clpreloader_seed_GF:
        cl_seed_GF_res_list.append(clmod_seed_GF.predict(pre))
    sum = 0
    for i in range(len(train_label)):
        sum += train_label[i] == int(cl_seed_GF_res_list[i])
    print(sum / len(train_label))
    clloader_seed_GF_t = cl.build_data_loader(test_feature_node_sd_GF_list, test_feature_edge_sd_list, test_label, cl_batch, shuffle=False)
    predicted_labels, acc = clmod_seed_GF.test(clloader_seed_GF_t)
    print('seed+GF: ', acc)
    result_pd = pd.DataFrame({'Column1': predicted_labels, 'Column2': test_label})
    result_pd.to_csv('base/result/seed+GF.csv')
    res_acc['seed_GF'] = acc

    ###########################################
    # lp_train_graph + train_GF -> GC_final_GF(t) -> train_class_3 
    # lp_test_graph  + test_GF  -> GC_final_GF    -> test_class_3 

    gf_dim = 192
    gf_input_dim = lp_input_dim + gf_dim
    
    cl_lp_GF_num_features = gf_input_dim   
    clloader_lp_GF = cl.build_data_loader(train_feature_node_sd_GF_list, train_feature_edge_after_lp_list, train_label, cl_batch)
    clmod_lp_GF = cl.MyGCN(cl_lp_GF_num_features, cl_hidden_channels, cl_out_channels, cl_num_classes)
    clmod_lp_GF.train(clloader_lp_GF, cl_epoch)
    clmod_lp_GF.save_model(r'model/temp/gcn_lp_GF_model_' + cl_name + '.pth')
    clmod_lp_GF = cl.MyGCN.load_model(r'model/temp/gcn_lp_GF_model_' + cl_name + '.pth', cl_lp_GF_num_features, cl_hidden_channels, cl_out_channels, cl_num_classes)
    cl_lp_GF_res_list = []
    clpreloader_lp_GF = cl.build_data_loader(train_feature_node_sd_GF_list, train_feature_edge_after_lp_list, None, batch=1, shuffle=False)
    for pre in clpreloader_lp_GF:
        cl_lp_GF_res_list.append(clmod_lp_GF.predict(pre))
    sum = 0
    for i in range(len(train_label)):
        sum += train_label[i] == int(cl_lp_GF_res_list[i])
    print(sum / len(train_label))
    clloader_lp_GF_t = cl.build_data_loader(test_feature_node_sd_GF_list, test_feature_edge_after_lp_list, test_label, cl_batch, shuffle=False)
    predicted_labels, acc = clmod_lp_GF.test(clloader_lp_GF_t)
    print('lp+GF: ', acc)
    result_pd = pd.DataFrame({'Column1': predicted_labels, 'Column2': test_label})
    result_pd.to_csv('base/result/lp+GF.csv')
    res_acc['lp_GF'] = acc

    return res_acc


def GF(gf_list,   
       label):   
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    train_gf_list, test_gf_list, train_label, test_label = train_test_split(gf_list, label)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(train_gf_list, train_label)    
    y_pred = rf_classifier.predict(test_gf_list)
    accuracy = accuracy_score(test_label, y_pred)
    y_pred_2 = [0 if i == 0 else 1 for i in y_pred]
    test_label_2 = [0 if i == 0 else 1 for i in test_label]
    accuracy_2 = accuracy_score(test_label_2, y_pred_2)
    print("Accuracy:", accuracy, accuracy_2)

###########################################

def run():
    train_test_set, split_list = split_dataset_k_fold( \
        [feature_node_sd_list, feature_node_sd_GF_list, feature_edge_sd_list, feature_edgetype_sd_list, 
        feature_node_s_list, feature_node_s_GF_list, feature_edge_s_list,
        feature_node_d_list, feature_node_d_GF_list, feature_edge_d_list], label_list, \
        k = 5)

    res_sd_acc = {'lp': [], 'lp_GF': [], 'seed': [], 'seed_GF': []}
    res_s_acc = {'lp': [], 'lp_GF': [], 'seed': [], 'seed_GF': []}
    res_d_acc = {'lp': [], 'lp_GF': [], 'seed': [], 'seed_GF': []}


    for st in train_test_set:
        train_data, train_label = st[0][0], st[0][1]
        test_data, test_label = st[1][0], st[1][1]
        [train_fnsd, train_fnsdGF, train_fesd, train_fetsd,
        train_fns, train_fnsGF, train_fes, train_fnd, train_fndGF, train_fed] = train_data
        [test_fnsd, test_fnsdGF, test_fesd, test_fetsd,
        test_fns, test_fnsGF, test_fes, test_fnd, test_fndGF, test_fed] = test_data

        res = one_round_sd(train_feature_node_sd_list=train_fnsd, 
                        train_feature_node_sd_GF_list=train_fnsdGF,
                        train_feature_edge_sd_list=train_fesd, 
                        train_edgetype_sd_list=train_fetsd, 
                        train_label=train_label, 
                        test_feature_node_sd_list=test_fnsd, 
                        test_feature_node_sd_GF_list=test_fnsdGF, 
                        test_feature_edge_sd_list=test_fesd, 
                        test_edgetype_sd_list=test_fetsd, 
                        test_label=test_label)
        
        for i in res:
            res_sd_acc[i].append(res[i].item())

        res = one_round_sd(train_feature_node_sd_list=train_fns, 
                        train_feature_node_sd_GF_list=train_fnsGF,
                        train_feature_edge_sd_list=train_fes, 
                        train_edgetype_sd_list=[[1 for l in range(len(i[0]))] for i in train_fes], 
                        train_label=train_label, 
                        test_feature_node_sd_list=test_fns, 
                        test_feature_node_sd_GF_list=test_fnsGF, 
                        test_feature_edge_sd_list=test_fes, 
                        test_edgetype_sd_list=[[1 for l in range(len(i[0]))] for i in test_fes], 
                        test_label=test_label)

        for i in res:
            res_s_acc[i].append(res[i].item())
        
        res = one_round_sd(train_feature_node_sd_list=train_fnd, 
                        train_feature_node_sd_GF_list=train_fndGF,
                        train_feature_edge_sd_list=train_fed, 
                        train_edgetype_sd_list=[[2 for l in range(len(i[0]))] for i in train_fed], 
                        train_label=train_label, 
                        test_feature_node_sd_list=test_fnd, 
                        test_feature_node_sd_GF_list=test_fndGF, 
                        test_feature_edge_sd_list=test_fed, 
                        test_edgetype_sd_list=[[2 for l in range(len(i[0]))] for i in test_fed], 
                        test_label=test_label)
        
        for i in res:
            res_d_acc[i].append(res[i].item())

    print('sd:')
    print(res_sd_acc)
    print([[i, np.mean(res_sd_acc[i])] for i in res_sd_acc])

    print('s:')
    print(res_s_acc)
    print([[i, np.mean(res_s_acc[i])] for i in res_s_acc])

    print('d:')
    print(res_d_acc)
    print([[i, np.mean(res_d_acc[i])] for i in res_d_acc])


def check():
    base_path = r'base/result/20240317/'
    type_list = ['sd', 's', 'd']
    name_list = ['seed', 'lp', 'seed+GF', 'lp+GF']

    for tpe in type_list:
        res_acc_4 = {'lp': [], 'lp+GF': [], 'seed': [], 'seed+GF': []}
        res_acc_2 = {'lp': [], 'lp+GF': [], 'seed': [], 'seed+GF': []}
        for name in name_list:
            for i in range(1, 6):
                path = os.path.join(base_path, tpe, str(i), name + '.csv')
                a4, a2 = check_accuracy(path)
                res_acc_4[name].append(a4)
                res_acc_2[name].append(a2)
            
        print('////////////////////' + '\n' + tpe)
        for i in res_acc_4:
            print('4 class', i, np.mean(res_acc_4[i]))
            print('2 class', i, np.mean(res_acc_2[i]))

# check()
# run()

# GF(gfdata, label)



