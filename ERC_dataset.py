from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import random
    
class MELD_loader(Dataset):
    def __init__(self, txt_file, dataclass):
        self.dialogs = []
        
        f = open(txt_file, 'r')
        dataset = f.readlines()
        f.close()
        
        temp_speakerList = []
        context = []
        context_speaker = []
        self.speakerNum = []
        # 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'
        emodict = {'anger': "anger", 'disgust': "disgust", 'fear': "fear", 'joy': "joy", 'neutral': "neutral", 'sadness': "sad", 'surprise': 'surprise'}
        self.sentidict = {'positive': ["joy"], 'negative': ["anger", "disgust", "fear", "sadness"], 'neutral': ["neutral", "surprise"]}
        self.emoSet = set()
        self.sentiSet = set()
        emo = 100
        for i, data in enumerate(dataset):
            if i < 2:
                continue
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                previous_emo = 100
                continue
            previous_emo = emo
            if previous_emo != 100:
                previous_emo = emodict[previous_emo]
            speaker, utt, emo, senti = data.strip().split('\t')
            context.append(utt)
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)
            
            self.dialogs.append([context_speaker[:], context[:], emodict[emo] ,previous_emo])#
            self.emoSet.add(emodict[emo])
            self.sentiSet.add(senti)
            
        self.emoList = sorted(self.emoSet)  
        self.sentiList = sorted(self.sentiSet)
        if dataclass == 'emotion':
            self.labelList = self.emoList
        else:
            self.labelList = self.sentiList        
        self.speakerNum.append(len(temp_speakerList))
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList, self.sentidict
    
    
class Emory_loader(Dataset):
    def __init__(self, txt_file, dataclass):
        self.dialogs = []
        
        f = open(txt_file, 'r')
        dataset = f.readlines()
        f.close()
        """sentiment"""
        # 'Joyful', 'Mad', 'Neutral', 'Peaceful', 'Powerful', 'Sad', 'Scared'
        pos = ['Joyful', 'Peaceful', 'Powerful']
        neg = ['Mad', 'Sad', 'Scared']
        neu = ['Neutral']
        emodict = {'Joyful': "joy", 'Mad': "mad", 'Peaceful': "peaceful", 'Powerful': "powerful", 'Neutral': "neutral", 'Sad': "sad", 'Scared': 'scared'}
        self.sentidict = {'positive': pos, 'negative': neg, 'neutral': neu}
        temp_speakerList = []
        context = []
        context_speaker = []        
        self.speakerNum = []
        self.emoSet = set()
        self.sentiSet = set()
        emo = 100
        for i, data in enumerate(dataset):
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                previous_emo = 100
                continue
            previous_emo = emo
            if previous_emo != 100:
                previous_emo = emodict[previous_emo]
            speaker, utt, emo = data.strip().split('\t')
            context.append(utt)
            
            if emo in pos:
                senti = "positive"
            elif emo in neg:
                senti = "negative"
            elif emo in neu:
                senti = "neutral"
            else:
                print('ERROR emotion&sentiment')
                
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)
            
            self.dialogs.append([context_speaker[:], context[:], emodict[emo],previous_emo])#,previous_emo
            self.emoSet.add(emodict[emo])
            self.sentiSet.add(senti)
            
        self.emoList = sorted(self.emoSet)
        self.sentiList = sorted(self.sentiSet)
        if dataclass == 'emotion':
            self.labelList = self.emoList
        else:
            self.labelList = self.sentiList        
        self.speakerNum.append(len(temp_speakerList))
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList, self.sentidict
    
    
class IEMOCAP_loader(Dataset):
    def __init__(self, txt_file, dataclass):
        self.dialogs = []
        
        f = open(txt_file, 'r')
        dataset = f.readlines()
        f.close()
        
        temp_speakerList = []
        context = []
        context_speaker = []
        self.speakerNum = []
        pos = ['exc', 'hap']
        neg = ['ang', 'fru', 'sad']
        neu = ['neu']
        emodict = {'ang': "angry", 'exc': "excited", 'fru': "frustrated", 'hap': "happy", 'neu': "neutral", 'sad': "sad"}
        self.sentidict = {'positive': pos, 'negative': neg, 'neutral': neu}
        # use: 'hap', 'sad', 'neu', 'ang', 'exc', 'fru'
        # discard: disgust, fear, other, surprise, xxx        
        self.emoSet = set()
        self.sentiSet = set()
        emo = 100
        for i, data in enumerate(dataset):
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                previous_emo = 100
                continue
            previous_emo = emo
            if previous_emo != 100:
                previous_emo = emodict[previous_emo]
            speaker = data.strip().split('\t')[0]
            utt = ' '.join(data.strip().split('\t')[1:-1])
            emo = data.strip().split('\t')[-1]
            context.append(utt)
            
            if emo in pos:
                senti = "positive"
            elif emo in neg:
                senti = "negative"
            elif emo in neu:
                senti = "neutral"
            else:
                print('ERROR emotion&sentiment')                        
            
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)
            
            self.dialogs.append([context_speaker[:], context[:], emodict[emo],previous_emo])#,previous_emo
            self.emoSet.add(emodict[emo])
        
        self.emoList = sorted(self.emoSet)   
        self.sentiList = sorted(self.sentiSet)
        if dataclass == 'emotion':
            self.labelList = self.emoList
        else:
            self.labelList = self.sentiList        
        self.speakerNum.append(len(temp_speakerList))
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList, self.sentidict
    
class DD_loader(Dataset):
    def __init__(self, txt_file, dataclass):
        self.dialogs = []
        
        f = open(txt_file, 'r')
        dataset = f.readlines()
        f.close()
        
        temp_speakerList = []
        context = []
        context_speaker = []
        self.speakerNum = []      
        self.emoSet = set()
        self.sentiSet = set()
        # {'anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'}
        pos = ['happiness']
        neg = ['anger', 'disgust', 'fear', 'sadness']
        neu = ['neutral', 'surprise']
        emodict = {'anger': "anger", 'disgust': "disgust", 'fear': "fear", 'happiness': "happy", 'neutral': "neutral", 'sadness': "sad", 'surprise': "surprise"}
        self.sentidict = {'positive': pos, 'negative': neg, 'neutral': neu}
        emo = 100
        for i, data in enumerate(dataset):
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                previous_emo = 100
                continue
            previous_emo = emo
            if previous_emo != 100:
                previous_emo = emodict[previous_emo]
            speaker = data.strip().split('\t')[0]
            utt = ' '.join(data.strip().split('\t')[1:-1])
            emo = data.strip().split('\t')[-1]
            
            if emo in pos:
                senti = "positive"
            elif emo in neg:
                senti = "negative"
            elif emo in neu:
                senti = "neutral"
            else:
                print('ERROR emotion&sentiment')                
            
            context.append(utt)
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)
            
            self.dialogs.append([context_speaker[:], context[:], emodict[emo],previous_emo]) #,previous_emo
            self.emoSet.add(emodict[emo])
        
        self.emoList = sorted(self.emoSet)   
        self.sentiList = sorted(self.sentiSet)
        if dataclass == 'emotion':
            self.labelList = self.emoList
        else:
            self.labelList = self.sentiList        
        self.speakerNum.append(len(temp_speakerList))
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList, self.sentidict