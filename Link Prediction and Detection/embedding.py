import os
import ast
import copy
import json
import draw
import align
import torch
import pickle
import random
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import pandas as pd
import numpy as np
from ast import literal_eval
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import BertModel
from torch.utils.tensorboard import SummaryWriter


SummaryWriterNum = 50
writer = SummaryWriter('D:\Code\log')
print(writer.log_dir)
current_SummaryWriterNum = 0
current_SummaryWriterTotal = 0
current_ScoreList = []

class app_folder():
    def __init__(self, folder_path):
        self.name = ''
        self.pack = ''
        self.perm = []
        self.acti = []
        self.serv = []
        self.rece = []
        self.prov = []
        self.cert = None
        self.static_graph = None
        self.dynamic_graph = None
        androguard_path = os.path.join(folder_path, 'androguard.txt')
        icon_path = os.path.join(folder_path, 'icon.png')
        static_path =  os.path.join(folder_path, 'static_graph.json')
        dynamic_path = os.path.join(folder_path, 'dynamic_graph.json')
        text_path = os.path.join(folder_path, 'text.txt')
        try:
            # androguard.txt
            with open(androguard_path, encoding='gbk') as f:
                [self.name, self.pack, perm, acti, serv, rece, prov, self.cert] = [kr.strip() for kr in f.readlines()]
                self.perm = perm.split(',')
                self.acti = acti.split(',')
                self.serv = serv.split(',')
                self.rece = rece.split(',')
                self.prov = prov.split(',')
        except:
            print('unable to get androguard info', folder_path)
        try:
            # icon.png
            with open(icon_path, 'rb') as f:
                self.icon = f.read()
        except:
            print('unable to get icon info', folder_path)
        try:
            # static_graph.json
            align.static_graph(static_path)
        except:
            print('unable to get static graph', folder_path)
        try:
            align.new_dynamic_graph(dynamic_path)
        except:
            print('unable to get dynamic graph', folder_path)
    
    def get_name_and_pack(self):
        res = self.name + '[SEP]' + self.pack
        return res


def activity_bert_embedding_and_classification():
    tokenizer = BertTokenizer.from_pretrained(r'model\transformer\bert-base-cased')
    class BertClassifier(nn.Module):
        def __init__(self, model_path, num_classes):
            super(BertClassifier, self).__init__()
            tokenizer = BertTokenizer.from_pretrained(model_path)
            model = BertModel.from_pretrained(model_path)
            self.bert = model
            self.dropout = nn.Dropout(0.1)
            self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

        def forward(self, input_ids, attention_mask):
            pooler_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            x = self.dropout(pooler_output)
            logits = self.fc(x)
            return logits
    
    class MyDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            tokens = tokenizer.tokenize(text)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            if len(input_ids) > 512:    
                i = input_ids
                input_ids = i[len(i)//2 - 256 : len(i)//2 + 256]
            elif len(input_ids) < 512:   
                input_ids = input_ids + [0 for i in range(512 - len(input_ids))]
            attention_mask = [1] * len(input_ids)
            return input_ids, attention_mask, label
            
    data = pd.read_csv('base\manifest.csv')
    labels = list(data['类别'])
    act = [literal_eval(i) for i in data['活动'].fillna(value="[]")]
    act_string = ['[SEP]'.join(i) for i in act]
    from sklearn.model_selection import train_test_split

    train_texts, test_texts, train_labels, test_labels = train_test_split(act_string, labels, test_size=0.2, random_state=42)

    train_dataset = MyDataset(train_texts, train_labels)
    test_dataset = MyDataset(test_texts, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_dict = torch.load('model/model_bertclassification.pth')
    model = BertClassifier(num_classes=4)
    model.load_state_dict(state_dict)

    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for inputs, _, labels in test_loader:
            inputs = torch.stack(inputs, dim=0).permute(1, 0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            attention_mask = (inputs != 0).float().to(device)
            logits = model(inputs, attention_mask)
            _, predicted = torch.max(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(correct / total, correct, total)
        accuracy = correct / total
        print('Test accuracy: {:.2f}%'.format(accuracy * 100))

def activity_bert_embedding_test():
    data = pd.read_csv('base\manifest.csv')
    cls = list(data['类别'])
    act = [literal_eval(i) for i in data['活动'].fillna(value="[]")]

    tokenizer = BertTokenizer.from_pretrained(r'\huggingface\hub\models--bert-base-uncased\snapshots\1dbc166cf8765166998eff31ade2eb64c8a40076')
    model = BertModel.from_pretrained(r'\huggingface\hub\models--bert-base-uncased\snapshots\1dbc166cf8765166998eff31ade2eb64c8a40076')

    token_ids = pd.read_csv('base/token_ids.csv', header=None)
    outputs = []
    for _, i in token_ids.iterrows():
        i = i.fillna(0)
        al = i.to_list()
        i = [int(ai) for ai in al]
        aes = model(torch.tensor(tokenizer.build_inputs_with_special_tokens(i)).unsqueeze(0))
        x1 = aes.pooler_output[0].tolist()
        outputs.append(x1)
    X = outputs
    tem_X_pd = pd.DataFrame(X, index=None)
    tem_X_pd.to_csv('base/X.csv', index=None, header=None)

    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, cls, test_size=0.5)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("Accuracy:", acc)


class name_bert_model(nn.Module):
    def __init__(self, base_model_path=r'model\transformer\bert-base-multilingual-uncased', num_classes=4):
        super(name_bert_model, self).__init__()
        model = BertModel.from_pretrained(base_model_path)
        self.bert = model
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        pooler_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        x = self.dropout(pooler_output)
        logits = self.fc(x)
        return logits

class name_bert_embedding():
    maxlength = 64     
    model_path = r'model\mymodel\name_bert\name_bert_model.pth' 
    base_model_path = r'model\mymodel\name_bert\transformer'   

    class name_bert_dataset(Dataset):
        def __init__(self, texts, labels, maxlength=64, base_tokenizer_path=r'model\transformer\bert-base-multilingual-cased'):
            self.texts = texts
            self.labels = labels
            self.maxlength = maxlength
            self.tokenizer = BertTokenizer.from_pretrained(base_tokenizer_path)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            tokens = self.tokenizer.tokenize(text)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids_sp = self.tokenizer.build_inputs_with_special_tokens(input_ids)
            maxl = self.maxlength          
            half = maxl // 2
            if len(input_ids_sp) > maxl:    
                input_ids_sp = input_ids_sp[len(input_ids_sp)//2 - half : len(input_ids_sp)//2 + half]
            elif len(input_ids_sp) < maxl:
                input_ids_sp = input_ids_sp + [0 for i in range(maxl - len(input_ids_sp))]
            attention_mask = [1] * len(input_ids_sp)
            return input_ids_sp, attention_mask, label    

    def __init__(self, model=None):
        if model:
            self.model = model
        else:
            self.model = name_bert_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def train(self, train_set, label_set, batch_size=32, epoch=5):
        train_dataset = self.name_bert_dataset(train_set, label_set)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        device = self.device
        model = self.model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        for ep in range(epoch):
            for inputs, _, labels in train_loader:
                inputs = torch.stack(inputs, dim=0).permute(1, 0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                attention_mask = (inputs != 0).float().to(device)
                optimizer.zero_grad()
                logits = model(inputs, attention_mask)
                loss = criterion(logits, labels)
                print(ep, loss)
                loss.backward()
                optimizer.step()
                self.save()

    def test(self, test_set, label_set, batch_size=32):
        test_dataset = self.name_bert_dataset(test_set, label_set)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        device = self.device
        model = self.model.to(device)
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for inputs, _, labels in test_loader:
                inputs = torch.stack(inputs, dim=0).permute(1, 0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                attention_mask = (inputs != 0).float().to(device)
                logits = model(inputs, attention_mask)
                _, predicted = torch.max(logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print(correct / total, correct, total)
            accuracy = correct / total
            print('Test accuracy: {:.2f}%'.format(accuracy * 100))

    def predict(self, predict_set, batch_size=32):
        label_set = [0 for i in predict_set]
        test_dataset = self.name_bert_dataset(predict_set, label_set)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        device = self.device
        model = self.model.to(device)
        result = []
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for inputs, _, labels in test_loader:
                inputs = torch.stack(inputs, dim=0).permute(1, 0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                attention_mask = (inputs != 0).float().to(device)
                logits = model(inputs, attention_mask)
                _, predicted = torch.max(logits, dim=1)
                print(predicted.tolist())
                result += predicted.tolist()
            return result

    def save(self, model_path=model_path, base_model_path=base_model_path):
        torch.save(self.model.state_dict(), model_path)
        self.model.bert.save_pretrained(base_model_path)
    
    @staticmethod
    def load(model_path=model_path, base_model_path=base_model_path):
        state_dict = torch.load(model_path)
        model = name_bert_model(base_model_path)
        model.load_state_dict(state_dict)
        cls = name_bert_embedding(model)
        return cls


class activity_bert_model(nn.Module):
    def __init__(self, base_model_path=r'model\transformer\bert-base-cased', out_channels=64, final_class=4):
        super(activity_bert_model, self).__init__()
        model = BertForMaskedLM.from_pretrained(base_model_path)
        self.bert = model
        for param in self.bert.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, out_channels)
        self.fn = nn.Linear(out_channels, final_class)

    def forward(self, input_ids, attention_mask):
        hidden_state = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.dropout(hidden_state[:, 0, :])
        feature = self.fc(x)
        logits = self.fn(feature)
        return logits, feature
    
class activity_bert_embedding():
    maxlength = 64      
    modified_base_model_path = r'model\mymodel\activity_bert\transformer'
    base_model_path = r'model\transformer\bert-base-cased'  
    model_path = r'model\mymodel\activity_bert\acti_bert_model.pth' 

    class activity_bert_dataset(Dataset):
        def __init__(self, texts, labels, maxlength=64, base_tokenizer_path=r'model\transformer\bert-base-cased'):
            self.texts = texts
            self.labels = labels
            self.maxlength = maxlength
            self.tokenizer = BertTokenizer.from_pretrained(base_tokenizer_path)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            tokens = self.tokenizer.tokenize(text)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids_sp = self.tokenizer.build_inputs_with_special_tokens(input_ids)
            maxl = self.maxlength       
            half = maxl // 2
            if len(input_ids_sp) > maxl:   
                input_ids_sp = input_ids_sp[len(input_ids_sp)//2 - half : len(input_ids_sp)//2 + half]
            elif len(input_ids_sp) < maxl: 
                input_ids_sp = input_ids_sp + [0 for i in range(maxl - len(input_ids_sp))]
            attention_mask = [1] * len(input_ids_sp)
            return input_ids_sp, attention_mask, label    

    class activity_bert_dataset_LM(Dataset):
        def __init__(self, texts, maxlength=64, base_tokenizer_path=r'model\transformer\bert-base-cased'):
            self.texts = texts
            self.maxlength = maxlength
            self.tokenizer = BertTokenizer.from_pretrained(base_tokenizer_path)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            tokens = self.tokenizer.tokenize(text)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids_sp = self.tokenizer.build_inputs_with_special_tokens(input_ids)
            maxl = self.maxlength         
            half = maxl // 2
            if len(input_ids_sp) > maxl:   
                input_ids_sp = input_ids_sp[len(input_ids_sp)//2 - half : len(input_ids_sp)//2 + half]
            elif len(input_ids_sp) < maxl: 
                input_ids_sp = input_ids_sp + [0 for i in range(maxl - len(input_ids_sp))]
            
            rm = random.randint(1, len(input_ids_sp) - 2)  
            input_ids_mask = copy.deepcopy(input_ids_sp)
            input_ids_mask[rm] = self.tokenizer.mask_token_id
            return input_ids_mask, input_ids_sp, rm   

    def __init__(self, model=None):
        if model:
            self.model = model
        else:
            # self.model = BertForMaskedLM.from_pretrained(self.base_model_path)
            self.model = activity_bert_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def train(self, train_set, label_set, batch_size=32, epoch=5):
        global SummaryWriterNum
        global writer
        global current_SummaryWriterNum
        global current_SummaryWriterTotal
        global current_ScoreList
        
        train_dataset = self.activity_bert_dataset(train_set, label_set)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        device = self.device
        model = self.model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        for ep in range(epoch):
            for inputs, _, labels in train_loader:
                inputs = torch.stack(inputs, dim=0).permute(1, 0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                attention_mask = (inputs != 0).float().to(device)
                optimizer.zero_grad()
                logits, feature = model(inputs, attention_mask)
                loss = criterion(logits, labels)
                corr = (torch.max(logits, dim=1)[1] == labels).sum().item() / batch_size
                print(ep, loss, corr)
                loss.backward()
                optimizer.step()
                self.save()
                current_SummaryWriterNum += 1
                current_ScoreList.append(loss)
                if current_SummaryWriterNum == SummaryWriterNum:
                    writer.add_scalar('Loss', torch.mean(torch.tensor(current_ScoreList)).item(), current_SummaryWriterTotal)
                    current_SummaryWriterTotal += 1
                    current_SummaryWriterNum = 0
                    current_ScoreList = []
            current_SummaryWriterTotal += 1
            writer.add_scalar('Loss', torch.mean(torch.tensor(current_ScoreList)).item(), current_SummaryWriterTotal)
    
    def train_with_MaskedLM(self, train_set, batch_size=32, epoch=5):
        global SummaryWriterNum
        global writer
        global current_SummaryWriterNum
        global current_SummaryWriterTotal
        global current_ScoreList

        train_dataset = self.activity_bert_dataset_LM(train_set)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        device = self.device
        model = self.model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        for ep in range(epoch):
            for inputs, actual, labels in train_loader:
                inputs = torch.stack(inputs, dim=0).permute(1, 0)
                actual = torch.stack(actual, dim=0).permute(1, 0)
                inputs = inputs.to(device)
                actual = actual.to(device)
                attention_mask = (inputs != 0).float().to(device)
                optimizer.zero_grad()
                for param in model.bert.parameters():
                    param.requires_grad = True
                output = model.bert(inputs, attention_mask, labels=actual)
                loss = output.loss
                print(ep, loss)
                loss.backward()
                optimizer.step()
                self.save()
                current_SummaryWriterNum += 1
                current_ScoreList.append(loss)
                if current_SummaryWriterNum == SummaryWriterNum:
                    writer.add_scalar('Loss', torch.mean(torch.tensor(current_ScoreList)).item(), current_SummaryWriterTotal)
                    current_SummaryWriterTotal += 1
                    current_SummaryWriterNum = 0
                    current_ScoreList = []
            current_SummaryWriterTotal += 1
            writer.add_scalar('Loss', torch.mean(torch.tensor(current_ScoreList)).item(), current_SummaryWriterTotal)

    def test(self, test_set, label_set, batch_size=32):
        test_dataset = self.activity_bert_dataset(test_set, label_set)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        device = self.device
        model = self.model.to(device)
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for inputs, _, labels in test_loader:
                inputs = torch.stack(inputs, dim=0).permute(1, 0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                attention_mask = (inputs != 0).float().to(device)
                logits = model(inputs, attention_mask)
                _, predicted = torch.max(logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print(correct / total, correct, total)
            accuracy = correct / total
            print('Test accuracy: {:.2f}%'.format(accuracy * 100))

    def predict(self, predict_set, batch_size=32):
        label_set = [0 for i in predict_set]
        test_dataset = self.activity_bert_dataset(predict_set, label_set)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        device = self.device
        model = self.model.to(device)
        result = []
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for inputs, _, labels in test_loader:
                inputs = torch.stack(inputs, dim=0).permute(1, 0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                attention_mask = (inputs != 0).float().to(device)
                logits, feature = model(inputs, attention_mask)
                _, predicted = torch.max(logits, dim=1)
                print(predicted.tolist())
                result += predicted.tolist()
            return result
    
    def predict_with_MaskedLM(self, predict_set, batch_size=32):
        label_set = [0 for i in predict_set]
        # model = BertForMaskedLM()
        test_dataset = self.activity_bert_dataset(predict_set, label_set)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        device = self.device
        model = self.model.to(device)
        result = []
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for inputs, _, labels in test_loader:
                inputs = torch.stack(inputs, dim=0).permute(1, 0)
                attention_mask = (inputs != 0).float().to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits, feature = model(inputs, attention_mask)
                # print(predicted.tolist())
                result += feature.tolist()
            return result
    
    def embedding(self, predict_set, batch_size=32):
        label_set = [0 for i in predict_set]
        test_dataset = self.activity_bert_dataset(predict_set, label_set)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        device = self.device
        model = self.model.to(device)
        result = []
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for inputs, _, labels in test_loader:
                inputs = torch.stack(inputs, dim=0).permute(1, 0)
                attention_mask = (inputs != 0).float().to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits, feature = model(inputs, attention_mask)
                # print(predicted.tolist())
                result += feature.tolist()
            return result
    
    def save(self, model_path=model_path, base_model_path=modified_base_model_path):
        torch.save(self.model.state_dict(), model_path)
        self.model.bert.save_pretrained(base_model_path)
    
    @staticmethod
    def load(model_path=model_path, base_model_path=base_model_path):
        state_dict = torch.load(model_path)
        model = activity_bert_model(base_model_path)
        model.load_state_dict(state_dict)
        cls = activity_bert_embedding(model)
        return cls


class text_bert_model(nn.Module):
    def __init__(self, base_model_path=r'model\transformer\bert-base-chinese', out_channels=64, final_class=4):
        super(text_bert_model, self).__init__()
        model = BertForMaskedLM.from_pretrained(base_model_path)
        self.bert = model
        for param in self.bert.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, out_channels)
        self.fn = nn.Linear(out_channels, final_class)

    def forward(self, input_ids, attention_mask):
        hidden_state = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.dropout(hidden_state[:, 0, :])
        feature = self.fc(x)
        logits = self.fn(feature)
        return logits, feature

class text_bert_embedding():
    maxlength = 256    
    modified_base_model_path = r'model\mymodel\text_bert\transformer' 
    base_model_path = r'model\transformer\bert-base-chinese'    
    model_path = r'model\mymodel\text_bert\text_bert_model.pth' 

    class text_bert_dataset(Dataset):
        def __init__(self, texts, labels, maxlength=64, base_tokenizer_path=r'model\transformer\bert-base-chinese'):
            self.texts = texts
            self.labels = labels
            self.maxlength = maxlength
            self.tokenizer = BertTokenizer.from_pretrained(base_tokenizer_path)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            tokens = self.tokenizer.tokenize(text)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids_sp = self.tokenizer.build_inputs_with_special_tokens(input_ids)
            maxl = self.maxlength         
            half = maxl // 2
            if len(input_ids_sp) > maxl:   
                input_ids_sp = input_ids_sp[len(input_ids_sp)//2 - half : len(input_ids_sp)//2 + half]
            elif len(input_ids_sp) < maxl:
                input_ids_sp = input_ids_sp + [0 for i in range(maxl - len(input_ids_sp))]
            attention_mask = [1] * len(input_ids_sp)
            return input_ids_sp, attention_mask, label    

    class text_bert_dataset_LM(Dataset):
        def __init__(self, texts, maxlength=256, base_tokenizer_path=r'model\transformer\bert-base-chinese'):
            self.texts = texts
            self.maxlength = maxlength
            self.tokenizer = BertTokenizer.from_pretrained(base_tokenizer_path)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            tokens = self.tokenizer.tokenize(text)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids_sp = self.tokenizer.build_inputs_with_special_tokens(input_ids)
            maxl = self.maxlength           
            half = maxl // 2
            if len(input_ids_sp) > maxl:   
                input_ids_sp = input_ids_sp[len(input_ids_sp)//2 - half : len(input_ids_sp)//2 + half]
            elif len(input_ids_sp) < maxl: 
                input_ids_sp = input_ids_sp + [0 for i in range(maxl - len(input_ids_sp))]
            
            change_prob = 0.15
            mask_prob = 0.8
            replace_prob = 0.1
            input_ids_mask = copy.deepcopy(input_ids_sp)
            for id in range(1, len(input_ids_mask) - 1):
                if input_ids_mask[id] == 0:   
                    continue
                prob = random.random()
                if prob < change_prob:
                    prob = random.random()
                    if prob < mask_prob: 
                        input_ids_mask[id] = self.tokenizer.mask_token_id
                    elif prob < mask_prob + replace_prob:
                        vocab = self.tokenizer.get_vocab()
                        tokens = list(vocab.keys())
                        random_index = random.randint(0, len(tokens) - 1)
                        # replace_token = tokens[random_index]
                        input_ids_mask[id] = random_index
            return input_ids_mask, input_ids_sp, 1   

    def __init__(self, model=None):
        if model:
            self.model = model
        else:
            self.model = text_bert_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def train(self, train_set, label_set, batch_size=32, epoch=5):
        train_dataset = self.text_bert_dataset(train_set, label_set)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        device = self.device
        model = self.model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        for ep in range(epoch):global SummaryWriterNum
        global writer
        global current_SummaryWriterNum
        global current_SummaryWriterTotal
        global current_ScoreList
        
        train_dataset = self.text_bert_dataset(train_set, label_set)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        device = self.device
        model = self.model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        for ep in range(epoch):
            for inputs, _, labels in train_loader:
                inputs = torch.stack(inputs, dim=0).permute(1, 0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                attention_mask = (inputs != 0).float().to(device)
                optimizer.zero_grad()
                logits, feature = model(inputs, attention_mask)
                loss = criterion(logits, labels)
                corr = (torch.max(logits, dim=1)[1] == labels).sum().item() / batch_size
                print(ep, loss, corr)
                loss.backward()
                optimizer.step()
                self.save()
                current_SummaryWriterNum += 1
                current_ScoreList.append(loss)
                if current_SummaryWriterNum == SummaryWriterNum:
                    writer.add_scalar('Loss', torch.mean(torch.tensor(current_ScoreList)).item(), current_SummaryWriterTotal)
                    current_SummaryWriterTotal += 1
                    current_SummaryWriterNum = 0
                    current_ScoreList = []
            current_SummaryWriterTotal += 1
            writer.add_scalar('Loss', torch.mean(torch.tensor(current_ScoreList)).item(), current_SummaryWriterTotal)
    
    def train_with_MaskedLM(self, train_set, batch_size=32, epoch=5):
        global SummaryWriterNum
        global writer
        global current_SummaryWriterNum
        global current_SummaryWriterTotal
        global current_ScoreList

        train_dataset = self.text_bert_dataset_LM(train_set)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        device = self.device
        model = self.model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        for ep in range(epoch):
            for inputs, actual, _ in train_loader:
                inputs = torch.stack(inputs, dim=0).permute(1, 0)
                actual = torch.stack(actual, dim=0).permute(1, 0)
                inputs = inputs.to(device)
                actual = actual.to(device)
                attention_mask = (inputs != 0).float().to(device)
                optimizer.zero_grad()
                output = model(inputs, attention_mask, labels=actual)
                logits = output.logits
                loss = output.loss
                print(ep, loss)
                loss.backward()
                optimizer.step()
                self.save()
                current_SummaryWriterNum += 1
                current_ScoreList.append(loss)
                if current_SummaryWriterNum == SummaryWriterNum:
                    writer.add_scalar('Loss', torch.mean(torch.tensor(current_ScoreList)).item(), current_SummaryWriterTotal)
                    current_SummaryWriterTotal += 1
                    current_SummaryWriterNum = 0
                    current_ScoreList = []
            current_SummaryWriterTotal += 1
            writer.add_scalar('Loss', torch.mean(torch.tensor(current_ScoreList)).item(), current_SummaryWriterTotal)

    def test(self, test_set, label_set, batch_size=32):
        test_dataset = self.text_bert_dataset(test_set, label_set)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        device = self.device
        model = self.model.to(device)
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for inputs, _, labels in test_loader:
                inputs = torch.stack(inputs, dim=0).permute(1, 0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                attention_mask = (inputs != 0).float().to(device)
                logits = model(inputs, attention_mask)
                _, predicted = torch.max(logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print(correct / total, correct, total)
            accuracy = correct / total
            print('Test accuracy: {:.2f}%'.format(accuracy * 100))

    def predict(self, predict_set, batch_size=32):
        label_set = [0 for i in predict_set]
        test_dataset = self.text_bert_dataset(predict_set, label_set)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        device = self.device
        model = self.model.to(device)
        result = []
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for inputs, _, labels in test_loader:
                inputs = torch.stack(inputs, dim=0).permute(1, 0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                attention_mask = (inputs != 0).float().to(device)
                logits = model(inputs, attention_mask)
                _, predicted = torch.max(logits, dim=1)
                print(predicted.tolist())
                result += predicted.tolist()
            return result
    
    def predict_with_MaskedLM(self, predict_set, batch_size=32):
        test_dataset = self.text_bert_dataset_LM(predict_set)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        device = self.device
        model = self.model.to(device)
        result = []
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for inputs, _, labels in test_loader:
                inputs = torch.stack(inputs, dim=0).permute(1, 0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                attention_mask = (inputs != 0).float().to(device)
                logits = model(inputs, attention_mask).logits
                _, predicted = torch.max(logits, dim=2)
                # print(predicted.tolist())
                result += predicted.tolist()
            return result
    
    def embedding(self, predict_set, batch_size=32):
        label_set = [0 for i in predict_set]
        # model = BertForMaskedLM()
        test_dataset = self.text_bert_dataset(predict_set, label_set)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        device = self.device
        model = self.model.to(device)
        result = []
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for inputs, _, labels in test_loader:
                inputs = torch.stack(inputs, dim=0).permute(1, 0)
                attention_mask = (inputs != 0).float().to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits, feature = model(inputs, attention_mask)
                # print(predicted.tolist())
                result += feature.tolist()
            return result
    
    def save(self, model_path=model_path, base_model_path=modified_base_model_path):
        torch.save(self.model.state_dict(), model_path)
        self.model.bert.save_pretrained(base_model_path)
    
    @staticmethod
    def load(model_path=model_path, base_model_path=base_model_path):
        state_dict = torch.load(model_path)
        model = text_bert_model(base_model_path)
        model.load_state_dict(state_dict)
        cls = text_bert_embedding(model)
        return cls

def get_feature(node_list):
    
    if len(node_list) == 0:
        return [[0 for i in range(132)]]
    norm_list = [[i['fragment'], i['dialog'], i['menu'], i['webview']] for i in node_list]
    acti_list = [i['activity'] for i in node_list]
    text_list = [i['mtext'] for i in node_list]
    E_acti = activity_bert_embedding.load(activity_bert_embedding.model_path)
    E_text = text_bert_embedding.load(text_bert_embedding.model_path)

    r_acti = E_acti.embedding(acti_list)
    r_text = E_text.embedding(text_list)

    feature_list = []
    for i in range(len(acti_list)):
        feature_list.append(norm_list[i] + r_acti[i] + r_text[i])
    return feature_list

def get_feature_sdm(node_list):
   
    if len(node_list) == 0:
        return [[0 for i in range(131)]], [[0 for i in range(129)]], [[0 for i in range(132)]]
    snorm_list = [[i['fragment'], i['dialog'], i['menu']] for i in node_list]
    dnorm_list = [[i['webview']] for i in node_list]
    mnorm_list = [[i['fragment'], i['dialog'], i['menu'], i['webview']] for i in node_list]
    acti_list = [i['activity'] for i in node_list]
    stext_list = [i['stext'] for i in node_list]
    dtext_list = [i['dtext'] for i in node_list]
    mtext_list = [i['mtext'] for i in node_list]
    E_acti = activity_bert_embedding.load(activity_bert_embedding.model_path)
    E_stext = text_bert_embedding.load(text_bert_embedding.model_path)
    E_dtext = text_bert_embedding.load(text_bert_embedding.model_path)
    E_mtext = text_bert_embedding.load(text_bert_embedding.model_path)

    r_acti = E_acti.embedding(acti_list)
    r_stext = E_stext.embedding(stext_list)
    r_dtext = E_dtext.embedding(dtext_list)
    r_mtext = E_mtext.embedding(mtext_list)

    sfeature_list = []
    dfeature_list = []
    mfeature_list = []
    for i in range(len(acti_list)):
        sfeature_list.append(snorm_list[i] + r_acti[i] + r_stext[i])
        dfeature_list.append(dnorm_list[i] + r_acti[i] + r_dtext[i])
        mfeature_list.append(mnorm_list[i] + r_acti[i] + r_mtext[i])
    return sfeature_list, dfeature_list, mfeature_list

    
def build_stellargraph(feature_list, edge_list, edge_type):
    import networkx as nx
    import matplotlib.pyplot as plt
    import stellargraph as sg

    n = len(feature_list)
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)

    src_list = edge_list[0]
    dst_list = edge_list[1]
    for i in range(len(edge_list[0])):
        G.add_edge(src_list[i], dst_list[i], type=edge_type[i])
    plt.figure()
    nx.draw(G, with_labels=True)
    plt.show()
    
    stellar_graph = sg.StellarGraph.from_networkx(G, node_features=pd.DataFrame(feature_list), edge_type_attr='type')
    print(stellar_graph.info())
    return stellar_graph

def build_combine_stellargraph(graph_list):
    import networkx as nx
    import matplotlib.pyplot as plt
    import stellargraph as sg

    nx_list = []
    all_feature_list = []
    now_n = 0
    for feature_list, edge_list, edge_type in graph_list:
        n = len(feature_list)
        G = nx.DiGraph()
        for i in range(n):
            G.add_node(i + now_n)

        src_list = edge_list[0]
        dst_list = edge_list[1]
        for i in range(len(edge_list[0])):
            G.add_edge(src_list[i] + now_n, dst_list[i] + now_n, type=edge_type[i])
        nx_list.append(G)
        all_feature_list.extend(feature_list)
        now_n += n
        
    all_G = nx.compose_all(nx_list)
    stellar_graph = sg.StellarGraph.from_networkx(all_G, node_features=pd.DataFrame(all_feature_list), edge_type_attr='type')
    print(stellar_graph.info())
    return stellar_graph
    
def build_tensor(feature_list, edge_list, edge_type):
    x = torch.tensor(feature_list, dtype=torch.float)
    edge_index = torch.tensor(edge_list, dtype=torch.long)
    edge_attr = torch.tensor([[i] for i in edge_type], dtype=torch.long)

    return x, edge_index, edge_attr

def build_combine_tensor(graph_list):
    all_feature_list = []
    all_edge_list = []
    all_edge_type = []
    now_n = 0
    for feature_list, edge_list, edge_type in graph_list:
        n = len(feature_list)
        all_feature_list.extend(feature_list)
        all_edge_type.extend(edge_type)
        all_edge_list = all_edge_list + [[i + now_n for i in edge_list[0]], [i + now_n for i in edge_list[1]]]
        now_n += n    

    x = torch.tensor(all_feature_list, dtype=torch.float)
    edge_index = torch.tensor(all_edge_list, dtype=torch.long)
    edge_attr = torch.tensor([[i] for i in all_edge_type], dtype=torch.long)
    return x, edge_index, edge_attr



if __name__ == '__main__':

    pass
