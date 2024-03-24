import csv
import string
import pickle
import pypinyin
import torch
import copy
import time
import csv
import gzip
from torch.utils.data import DataLoader, Dataset
import datetime
import matplotlib.pyplot as plt
import numpy as np
import ast
from sklearn.feature_extraction.text import HashingVectorizer
from torch.utils.tensorboard import SummaryWriter



class manifest_text_model(object):
    # Parameters
    HIDDEN_SIZE = 64
    BATCH_SIZE = 128    # 256
    N_LAYER = 2
    N_EPOCHS = 30
    N_LABEL_NUM = 4
    N_CHARS = 128
    USE_GPU = False
    Hashingvector_len = 256  # 256
    hashing_vectorizer = HashingVectorizer(n_features=Hashingvector_len, alternate_sign=False)

    @staticmethod
    def pinyin(word):
        s = ''
        for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
            s += ''.join(i)
        return s
    
    @staticmethod
    def toregular(pinyin):
        rw = ''
        for i in pinyin:
            if i not in string.printable:
                rw += '$'
            else:
                rw += i
        return rw

    def __init__(self, x_text, y=None):
        self.names = [self.toregular(self.pinyin(x)) for x in x_text]  
        self.len = len(self.names)  
        if y is not None:
            self.labels = y             
            self.label_list = list(sorted(set(self.labels)))   
            self.label_dict = self.getLabelDict()              
            self.label_num = len(self.label_list)               
        else:
            self.labels = ['test' for i in x_text]
            self.label_list = ['test']
            self.label_dict = self.getLabelDict()
            self.label_num = len(self.label_list)

    def train(self, path):
        self.loader = DataLoader(self, batch_size=self.BATCH_SIZE, shuffle=True)
        self.classifier = RNNClassifier(self.N_CHARS, self.HIDDEN_SIZE, self.label_num, self.N_LAYER)
        self.criterion = torch.nn.CrossEntropyLoss()  
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001) 

        acc_list = []
        los_list = []
        start = time.time()
        for epoch in range(1, self.N_EPOCHS + 1):
            print('%d / %d:' % (epoch, self.N_EPOCHS))
            acc, loss = self.trainModel()
            acc_list.append(acc)
            los_list.append(loss)
        end = time.time()
        print(datetime.timedelta(seconds=(end - start) // 1))
        self.classifier.save(path)
        return acc
    
    def test(self, path):
        self.loader = DataLoader(self, batch_size=self.BATCH_SIZE, shuffle=True)
        self.classifier = RNNClassifier.load(path)

        start = time.time()
        res, acc = self.testModel()
        end = time.time()
        print(datetime.timedelta(seconds=(end - start) // 1))
        return np.array(res.cpu()), acc
    
    def test_embedding(self, path):
        self.loader = DataLoader(self, batch_size=self.BATCH_SIZE, shuffle=False)
        self.classifier = RNNClassifier.load(path)

        start = time.time()
        res, acc, emb = self.testModel_Embedding()
        end = time.time()
        print(datetime.timedelta(seconds=(end - start) // 1))
        return np.array(res.cpu()), acc, np.array(emb.cpu())

    def __getitem__(self, index):
        return self.names[index], self.label_dict[self.labels[index]]

    def __len__(self):
        return self.len

    def getLabelDict(self):
        Label_dict = dict()  
        for idx, label_name in enumerate(self.label_list, 0): 
            Label_dict[label_name] = idx  
        return Label_dict

    def idx2Label(self, index):
        return self.label_list(index)

    def getLabelsNum(self):
        return self.label_num
    
    @staticmethod
    def name2list(name):  
        arr = [ord(c) for c in name]
        return arr, len(arr)

    def make_tensors(self, names, labels):  
        sequences_and_lengths = [self.name2list(names[i]) for i in range(len(names))]  
        name_sequences = [sl[0] for sl in sequences_and_lengths]                       
        seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])         
        labels = labels.long()

        # make tensor of name, BatchSize x SeqLen
        seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()    
        for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):  
            seq_tensor[idx, :seq_len] = torch.LongTensor(seq)                      

        # sort by length to use pack_padded_sequence
        seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)    
        seq_tensor = seq_tensor[perm_idx]                                  
        labels = labels[perm_idx]                           

        inv_perm_idx = torch.argsort(perm_idx)
        return self.classifier.create_tensor(seq_tensor), self.classifier.create_tensor(seq_lengths), self.classifier.create_tensor(labels), inv_perm_idx

    def trainModel(self):
        correct = 0
        total_loss = 0
        total = len(self)
        for i, (names, labels) in enumerate(self.loader, 1):
            # print(permission)
            self.optimizer.zero_grad()
            inputs, seq_lengths, target, _ = self.make_tensors(names, labels)  
            output, _ = self.classifier(inputs, seq_lengths)  
            loss = self.criterion(output, target) 

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            pred = output.max(dim=1, keepdim=True)[1]              
            correct += pred.eq(target.view_as(pred)).sum().item()  
            percent = '%.2f' % (100 * correct / total)
            print(f'Accuracy {correct}/{total} {percent}%')

            if i == len(self) // self.BATCH_SIZE:
                print(f'loss={total_loss / (i * len(inputs))}')
        return correct / total, total_loss

    def testModel(self):
        correct = 0
        res = []
        total = len(self)
        with torch.no_grad():
            for i, (names, labels) in enumerate(self.loader, 1):
                inputs, seq_lengths, target, inv_perm_index = self.make_tensors(names, labels)  # 返回处理后的名字ASCII码 重新排序的长度和标签列表
                output, _ = self.classifier(inputs, seq_lengths)          
                res.append(output[inv_perm_index])
                pred = output.max(dim=1, keepdim=True)[1]              
                correct += pred.eq(target.view_as(pred)).sum().item()  

        percent = '%.2f' % (100 * correct / total)
        print(f'Accuracy {correct}/{total} {percent}%')
        result = torch.cat(res)
        return result, percent
    
    def testModel_Embedding(self):
        res_0 = []
        res_1 = []
        total = len(self)
        with torch.no_grad():
            for i, (names, labels) in enumerate(self.loader, 1):
                inputs, seq_lengths, target, inv_perm_index = self.make_tensors(names, labels)  # 返回处理后的名字ASCII码 重新排序的长度和标签列表
                output, emb = self.classifier(inputs, seq_lengths)         
               
                res_0.append(output[inv_perm_index])
                res_1.append(emb[inv_perm_index])
                pred = output.max(dim=1, keepdim=True)[1]               
                correct += pred.eq(target.view_as(pred)).sum().item() 

        percent = '%.2f' % (100 * correct / total)
        print(f'Accuracy {correct}/{total} {percent}%')
        result = torch.cat(res_0)
        result_2 = torch.cat(res_1)
        return result, percent, result_2

class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size  
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        self.USE_GPU = manifest_text_model.USE_GPU

        self.embedding = torch.nn.Embedding(input_size,
                                            hidden_size) 
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size) 
        if self.USE_GPU:
            self.to("cuda:0")

    def create_tensor(self, tensor): 
        if self.USE_GPU:
            device = torch.device("cuda:0")
            tensor = tensor.to(device)
        return tensor

    def forward(self, input, seq_lengths):
        input = input.t() 
        batch_size = input.size(1)
        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(input)

        if self.USE_GPU:
            seq_lengths = seq_lengths.cpu() 
        # pack them up
        gru_input = torch.nn.utils.rnn.pack_padded_sequence(embedding, seq_lengths) 

        output, hidden = self.gru(gru_input, hidden)  
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output, hidden_cat

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return self.create_tensor(hidden)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    @staticmethod
    def load(path):
        a = torch.load(path)
        b = RNNClassifier(manifest_text_model.N_CHARS, manifest_text_model.HIDDEN_SIZE, manifest_text_model.N_LABEL_NUM, manifest_text_model.N_LAYER)
        b.load_state_dict(a)
        return b

import re
import io
import cv2
import shutil
import easyocr
from PIL import Image
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

def text_re(image_array):
    global reader
    result = reader.readtext(image_array)  
    return result

def get_color_histogram_image(img, hsv_img):
    try:
        r, g, b, h, s = img[:,:,0], img[:,:,1], img[:,:,2], hsv_img[:,:,0], hsv_img[:,:,1]
        color_groups = np.array_split(np.arange(256), 16)
        hist = []
        for i in [r, g, b, h, s]:
            histogram = []
            for group in color_groups:
                group_histogram = 0
                for color_value in group:
                    group_histogram += np.sum(i == color_value)
                histogram.append(group_histogram)
            hist = hist + histogram
        v_groups = np.array_split(np.arange(180), 10)
        histogram = []
        for group in v_groups:
            group_histogram = 0
            for v_value in group:
                group_histogram += np.sum(i == v_value)
            histogram.append(group_histogram)
        hist = hist + histogram
        modify_hist = hist / (np.sum(hist) / 6)
        return modify_hist
    except Exception as e:
        print(e)
        return [0 for i in range(90)]
    
def get_text_feature_image(img):
    txt_re = text_re(img)
    txt = [t[1] for t in txt_re]
    i0 = len(''.join(txt))
    i1 = len(txt)
    
    match_com = r'com|cc|con|cn|cm|net|xyz'
    matchres_com = [s for s in txt if re.findall(match_com, s)]
    match_gamb = r'彩票|娱乐|体育|福利|威尼斯|世界杯|棋牌|新葡京|游戏'
    matchres_gamb = [s for s in txt if re.findall(match_gamb, s)]
    match_porn = r'fuck|av|色|撸|约炮'
    matchres_porn = [s for s in txt if re.findall(match_porn, s)]
    match_scam = r'交易|兼职'
    matchres_scam = [s for s in txt if re.findall(match_scam, s)]
    i2 = len(matchres_com) if len(matchres_com) else 0
    i3 = len(matchres_gamb) if len(matchres_gamb) else 0
    i4 = len(matchres_porn) if len(matchres_porn) else 0
    i5 = len(matchres_scam) if len(matchres_scam) else 0
    res = [i0, i1, i2, i3, i4, i5]
    return res


class icon_embedding_fc(torch.nn.Module):
    def __init__(self, in_channels=96, hidden_channels=128, out_channels=64, final_class=4):
        super(icon_embedding_fc, self).__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.norm = torch.nn.BatchNorm1d(in_channels)
        self.fh = torch.nn.Linear(in_channels, hidden_channels)
        self.relu1 = torch.nn.Sigmoid()
        self.norm2 = torch.nn.BatchNorm1d(hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)
        self.relu2 = torch.nn.Sigmoid()
        self.fn = torch.nn.Linear(out_channels, final_class)


    def forward(self, input):
        input = self.norm(input)
        hidden_state = self.fh(input)
        hidden_state = self.norm2(hidden_state)
        x = self.dropout(hidden_state)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        feature = self.fc(x)
        x = self.norm2(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        logits = self.fn(feature)
        return logits, feature

class icon_model():

    model_path = r'\model\mymodel\icon_model_96.pth'
    def __init__(self, model=None):
        if model:
            self.model = model
        else:
            self.model = icon_embedding_fc()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class icon_dataset(Dataset):
        def __init__(self, icons, labels, ignore_none=False):
            self.icons = []
            self.labels = []
            for i in range(len(icons)):
                try:
                    img = Image.open(io.BytesIO(eval(icons[i])))
                    numpy_image = np.array(img)
                    cv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
                    rgb_image_matrix = np.array(cv_image)
                    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
                    hsv_image_matrix = np.array(hsv_image)

                except Exception as e:
                    print(e)
                    print('error when opening', icons[i])
                    if ignore_none:    
                        pass
                    else:
                        self.labels.append(labels[i])
                        self.icons.append(torch.tensor(list(np.zeros(96)), dtype=torch.float32))
                    continue
                try:   
                    c_90 = get_color_histogram_image(rgb_image_matrix, hsv_image_matrix)
                except:
                    c_90 = np.zeros(90)
                try:
                    c_6 = get_text_feature_image(rgb_image_matrix)
                except:
                    c_6 = np.zeros(6)
                self.labels.append(labels[i])
                self.icons.append(torch.tensor(list(c_90) + list(c_6), dtype=torch.float32))

        def __len__(self):
            return len(self.icons)

        def __getitem__(self, idx):
            icon = self.icons[idx]
            label = self.labels[idx]
            return icon, label

    def train(self, icon_paths, labels, batch_size=32, epoch=100):
        train_dataset = self.icon_dataset(icon_paths, labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # move to GPU (if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        # initialize the optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        # summarywriter
        writer = SummaryWriter('D:\Code\log')
        print(writer.log_dir)
        for e in range(1, epoch + 1):
            e_loss = []
            e_acc = []
            for data, label in train_loader:
                try:
                    self.model.train()
                    optimizer.zero_grad()
                    out, feature = self.model(data)
                    loss = criterion(out, label)
                    # acc
                    print(loss)
                    loss.backward()
                    optimizer.step()
                    out_label = torch.argmax(out, dim=-1)
                    acc = torch.sum(out_label == label) / out_label.shape[0]
                    e_loss.append(loss)
                    e_acc.append(acc)
                    print('Epoch: {:03d}, LOSS: {:.4f}, ACC: {:.4f}'.format(e, loss, acc))
                except Exception as exc:
                    print(exc)
            e_loss_num = torch.mean(torch.tensor(e_loss))
            e_acc_num = torch.mean(torch.tensor(e_acc))
            print('E_print', e_loss_num, e_acc_num)
            writer.add_scalar('Loss', e_loss_num.item(), e)
            writer.add_scalar('Accuracy', e_acc_num.item(), e)
        writer.close()    

    def test(self, icon_paths, label_set, batch_size=32):
        test_dataset = self.icon_dataset(icon_paths, label_set)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        device = self.device
        model = self.model.to(device)
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits, features = model(inputs)
                _, predicted = torch.max(logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print(correct / total, correct, total)
            accuracy = correct / total
            print('Test accuracy: {:.2f}%'.format(accuracy * 100))
            return accuracy

    def embedding(self, icon_paths, batch_size=32):

        label_set = [0 for i in icon_paths]
        # model = BertForMaskedLM()
        test_dataset = self.icon_dataset(icon_paths, label_set, ignore_none=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        device = self.device
        model = self.model.to(device)
        result = []
        with torch.no_grad():
            model.eval()
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits, feature = model(inputs)
                # print(predicted.tolist())
                result += feature.tolist()
            return result

    def save(self, model_path=model_path):
        torch.save(self.model.state_dict(), model_path)
    
    @staticmethod
    def load(model_path=model_path):
        state_dict = torch.load(model_path)
        model = icon_model()
        model.model.load_state_dict(state_dict)
        return model

def get_global_feature(global_list):
    
    npc = [global_list['name'] + '; ' + global_list['pack'] + '; ' + global_list['cert']]
    ico = [global_list['icon']]
    # lbl_list = [i['label'] for i in global_list]
    npc_model = manifest_text_model(npc)
    ico_model = icon_model.load('model/mymodel/icon_model.pth')

    res, acc, r_npc = npc_model.test_embedding('model/mymodel/npc_model_128.pth')
    r_ico = ico_model.embedding(ico)
    res_feature = r_npc[0].tolist() + r_ico[0]
    return res_feature

if __name__ == '__main__':

    from sklearn.model_selection import StratifiedKFold
    import pandas as pd
    import numpy as np

    icon_csv = pd.read_csv('base\icon.csv')
    labels = icon_csv['类别'].tolist()
    icons = icon_csv['图标'].tolist()
    from sklearn.model_selection import train_test_split
    train_icons, test_icons, train_labels, test_labels = train_test_split(icons, labels, test_size=0.2, random_state=42)

    res = []
    E_icon = icon_model()
    E_icon.train(train_icons, train_labels, epoch=10)
    # E_icon.load(r'model\temp\3.pth')
    a = E_icon.test(test_icons, test_labels)
    res.append(a)
    E_icon.save(model_path=r'model\temp\1.pth')

    E_icon.train(train_icons, train_labels, epoch=10)
    E_icon.test(test_icons, test_labels)
    res.append(a)
    E_icon.save(model_path=r'model\temp\2.pth')

    E_icon.train(train_icons, train_labels, epoch=10)
    a = E_icon.test(test_icons, test_labels)
    res.append(a)
    E_icon.save(model_path=r'model\temp\3.pth')

    print(res)
