# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import numpy as np
import random
import torch
import torch.nn as nn
import pandas as pd

from ERC_dataset import MELD_loader, Emory_loader, IEMOCAP_loader, DD_loader
from model import ERC_model

from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import pdb
import argparse, logging
from sklearn.metrics import precision_recall_fscore_support
from utils import encode_right_truncated, padding
from utils import make_batch_roberta
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.utils import Bunch
from sklearn.manifold import TSNE



def CELoss(pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val
    
## finetune RoBETa-large
def main():  


    """Dataset Loading"""
    dataclass = 'emotion'
    model_type = args.pretrained
    dataset = args.dataset
    dataType = 'multi'
    
    #for dataset, dataclass in zip(dataset_list, dataclass_list):
    if dataset == 'MELD':
        if args.dyadic:
            dataType = 'dyadic'
        else:
            dataType = 'multi'
        data_path = './dataset/MELD/'+dataType+'/'
        DATA_loader = MELD_loader
    elif dataset == 'EMORY':
        data_path = './dataset/EMORY/'
        DATA_loader = Emory_loader
    elif dataset == 'iemocap':
        data_path = './dataset/iemocap/'
        DATA_loader = IEMOCAP_loader
    elif dataset == 'dailydialog':
        data_path = './dataset/dailydialog/'
        DATA_loader = DD_loader

    if model_type == 'roberta-large':
        make_batch = make_batch_roberta
    

    test_path = data_path + dataset+'_test.txt'

    test_dataset = DATA_loader(test_path, 'emotion')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)

    """logging and path"""
    save_path = os.path.join(dataset+'_models', model_type, args.loss)

    print("###Save Path### ", save_path)
    log_path = os.path.join(save_path, 'test.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)      

    """Model Loading"""
    if 'gpt2' in model_type:
        last = True
    else:
        last = False

    print('DataClass: ', 'emotion', '!!!') # emotion    
    if dataclass == 'emotion':
        clsNum = len(test_dataset.emoList)
    else:
        clsNum = len(test_dataset.sentiList)
    label_list = test_dataset.labelList
    layer_set = [-1]
    
    model = ERC_model(model_type, clsNum, last,'')
    modelfile = os.path.join(save_path, 'model.bin')
    model.load_state_dict(torch.load(modelfile))
    model = model.cuda()
    model.eval() 

    """Dev & Test evaluation"""
    cand_labels = []
    for k, label in enumerate(label_list):
        if dataclass == 'sentiment':
            cand_labels.append(k)
        elif label != 'neutral': # emotion
            cand_labels.append(k)
    
    logger.info("Data: {}, label: {}".format(dataset, dataclass))
    if dataset == 'dailydialog': # micro & macro
        test_prek, test_pred_list, test_label_list = _CalACC(model, test_dataloader, args.loss,dataset, save_path)
        test_pre_macro, test_rec_macro, test_fbeta_macro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='macro')
        test_pre_micro, test_rec_micro, test_fbeta_micro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, labels=cand_labels, average='micro') # neutral x
    else: # weight
        test_prek, test_pred_list, test_label_list = _CalACC(model, test_dataloader,args.loss,dataset, save_path)
        test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list,  average='weighted')#labels=cand_labels,

    if dataset == 'dailydialog': # micro & macro
        logger.info('test-precision: {}, test-macro: {}, test-micro: {}'.format(test_prek, test_fbeta_macro, test_fbeta_micro)) 
    else:
        logger.info('test-precision: {}, test-fscore: {}'.format(test_prek, test_fbeta))  


    
def _CalACC(model, dataloader, loss_method, dataset, directory):
    eval_model = model.eval()
    correct = 0
    label_list = []
    pred_list = []
    
    p1num, p2num, p3num = 0, 0, 0    
    # label arragne
    tsne_embedding = []
    tsne_label = []
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):            
            """Prediction"""
            batch_input_tokens, batch_labels,_,_ = data
            batch_input_tokens, batch_labels = batch_input_tokens.cuda(), batch_labels.cuda()
            
            pred_logits,last_layer_embedding,_ = model(batch_input_tokens) # (1, clsNum)
            
            tsne_embedding.append(last_layer_embedding.squeeze().detach().cpu().numpy())
            tsne_label.append(int(batch_labels.detach().cpu()))


            """Calculation"""    
            pred_logits_sort = pred_logits.sort(descending=True)
            indices = pred_logits_sort.indices.tolist()[0]
            
            pred_label = indices[0] # pred_logits.argmax(1).item()
            true_label = batch_labels.item()
            
            pred_list.append(pred_label)
            label_list.append(true_label)
            if pred_label == true_label:
                correct += 1
                
            """Calculation precision"""
            if true_label in indices[:1]:
                p1num += 1
            if true_label in indices[:2]:
                p2num += 1/2
            if true_label in indices[:3]:
                p3num += 1/3
            
        p1 = round(p1num/len(dataloader)*100, 2)
        p2 = round(p2num/len(dataloader)*100, 2)
        p3 = round(p3num/len(dataloader)*100, 2)
        

    return [p1, p2, p3], pred_list, label_list
            

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Emotion Classifier" )    
    parser.add_argument( "--pretrained", help = 'roberta-large or bert-large-uncased or gpt2 or gpt2-large or gpt2-medium', default = 'roberta-large')
    parser.add_argument('-dya', '--dyadic', action='store_true', help='dyadic conversation')
    parser.add_argument('--loss',  default='ce')
    parser.add_argument('--dataset', default='MELD')
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()
    