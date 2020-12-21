from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from nltk.tokenize import sent_tokenize


import random
import time
from transformers import AutoTokenizer

from OurTransformers import EncoderLayer
from OurBertTokenizer import MyTokenizer
from OurBertModel import MyBertModel
from OurBertModelTrainer import BertModelTrainer


if __name__ == "__main__":
    f = open("data_low.txt", 'r', encoding='utf8')
    data = f.read().rstrip()
    #data = re.sub(r"\[[0-9]+\]","",data)
    #data = re.sub(r"\([^)]*\)"," ",data)
    data = re.sub(r"\n"," ",data)
    #data = re.sub(r","," ",data)
    data = re.sub(r" +"," ",data)
    #data = data[10:] # Remove title
    data = data.lower()
    #data = data[:len(data)//10]
    f.close()
    
    SEQ_LEN = 64
      
    Mytokenize = MyTokenizer(data,SEQ_LEN,True,True)

    masked_id,mask_index,mask_label,NSP_label,segment = Mytokenize(data,reverse=False,to_lower=True)
    mask_index = mask_index
    
    NB_ENCODER = 12
    FFN_UNITS = 768
    NB_ATTENTION_HEAD = 8
    HIDDEN_UNITS = 512
    VOCAB_SIZE = Mytokenize.get_vocab_size()
    DROPOUT = 0.10
    
    
    masked_id = np.array(masked_id)
    segment = np.array(segment)
    NSP_label = np.array(NSP_label)
    
    #tvars = tf.compat.v1.trainable_variables()   
    OUR_BERT = MyBertModel(nb_encoder_layers=NB_ENCODER,
                           FFN_units=FFN_UNITS,
                           nb_attention_head=NB_ATTENTION_HEAD,
                           nb_hidden_units=HIDDEN_UNITS,
                           dropout_rate=DROPOUT,
                           vocab_size=VOCAB_SIZE
                           )
    
    
    MaxSeq = 0
    length = [len(i) for i in mask_index]
    MaxSeq = max(length)
    
    #deneme = OUR_BERT(masked_id,segment,True)
    #deneme_MLM = OUR_BERT.train_for_MLM(masked_id,segment,True)
    #deneme_NSP = OUR_BERT.train_for_NSP(masked_id,segment,True)
    
    Trainer = BertModelTrainer(HIDDEN_UNITS) 
    trainingModel =  Trainer(BertModel=OUR_BERT,
                            epochs=100,
                            inputs=masked_id,
                            NSP_label=NSP_label,
                            mask_index=mask_index,
                            mask_label=mask_label,
                            segment=segment,
                            checkpoint_path="Bert_Checkpoint/",
                            max2keep=2,
                            batch2Show=8)
    
    
    
    
     
    
   