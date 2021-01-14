import numpy as np
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import AutoTokenizer

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import time

from OurTransformers import DecoderLayer
from OurBertTokenizer import MyTokenizer
from OurBertModel import MyBertModel
from WithBERT_MyTranslator import MyBertTranslator
from WithBERT_MyTranslator import TranslatorTrainer
from OurBertModelTrainer import BertModelTrainer

import pandas as pd

def findfitseq(data):
    token_len = [len(token) for token in data]
    token_len = np.array(token_len)
        
    max_tokens = np.mean(token_len) + 2*np.std(token_len)
    max_tokens = int(max_tokens)
    return max_tokens

def return_token_type_ids(input_id):
        segment = []
        s = 1
        for sentence_ids in input_id:
            sentence_segment = []
            for ids in sentence_ids:
                if ids == 0:
                    s=0
                else:
                    s=1   
                sentence_segment.append(s)
  
            segment.append(sentence_segment)
            
        return segment

# Türkçe yazı yazacağım, bana ingilizce çıktısını verecek
if __name__ == "__main__":
    data_src = []
    data_dest = []
        
    for line in open('EnglishTurkishCorpus.txt', encoding='UTF-8'):
        en_text, tr_text = line.rstrip().lower().split('\t')
            
        data_src.append(tr_text)
        data_dest.append(en_text)

    slice_part = 200000
    data_src = data_src[:slice_part]
    data_dest = data_dest[:slice_part]
    
    # remove punctuation
    for i in range(len(data_src)):
        data_src[i] = re.sub(r'[^\w\s]', '', data_src[i])
    for i in range(len(data_dest)):
        data_dest[i] = re.sub(r'[^\w\s]', '', data_dest[i])
            
    EnglishTokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    turkishTokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
    
    turkishBatch = turkishTokenizer(data_src)
    englishBatch = EnglishTokenizer(data_dest)
    
    turkish_fit = findfitseq(turkishBatch['input_ids'])
    english_fit = findfitseq(englishBatch['input_ids'])
    fit_len = turkish_fit if turkish_fit>english_fit else english_fit
    
    turkishBatch_Id = turkishBatch['input_ids']
    turkishBatch_S = turkishBatch['token_type_ids'] # of course all member is 0
    englishBatch = englishBatch['input_ids']
    
    # removing long sentences
    idx_to_remove = [count for count, sent in enumerate(turkishBatch_Id)
                     if len(sent) > fit_len]
    for idx in reversed(idx_to_remove):
        del turkishBatch_Id[idx]
        del turkishBatch_S[idx]
        del englishBatch[idx]
    idx_to_remove = [count for count, sent in enumerate(englishBatch)
                     if len(sent) > fit_len]
    for idx in reversed(idx_to_remove):
        del turkishBatch_Id[idx]
        del turkishBatch_S[idx]
        del englishBatch[idx]
    
    turkishBatch_Id = tf.keras.preprocessing.sequence.pad_sequences(turkishBatch_Id,
                                                                    value=0,
                                                                    padding='post',
                                                                    maxlen=fit_len)

    englishBatch = tf.keras.preprocessing.sequence.pad_sequences(englishBatch,
                                                                 value=0,
                                                                 padding='post',
                                                                 maxlen=fit_len)
    
    turkishBatch_S = np.array(return_token_type_ids(turkishBatch_Id))
    
    BATCH_SIZE = 32
    BUFFER_SIZE = 1500
        
    dataset = tf.data.Dataset.from_tensor_slices((turkishBatch_Id,turkishBatch_S,englishBatch))
        
    dataset = dataset.cache() # Just increase speed
    dataset = dataset.shuffle(BUFFER_SIZE,seed=3).batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) # same things as the cache 
    
    NB_ENCODER = 12
    NB_DECODER = 6
    FFN_UNITS = 768
    NB_ATTENTION_HEAD = 8 #12
    HIDDEN_UNITS = 512 #768
    VOCAB_SIZE_TURKISH = turkishTokenizer.vocab_size
    VOCAB_SIZE_ENGLISH = EnglishTokenizer.vocab_size
    DROPOUT = 0.10
    SEQ_LEN = 64
    
    OUR_BERT = MyBertModel(nb_encoder_layers=NB_ENCODER,
                           FFN_units=FFN_UNITS,
                           nb_attention_head=NB_ATTENTION_HEAD,
                           nb_hidden_units=HIDDEN_UNITS,
                           dropout_rate=DROPOUT,
                           vocab_size=VOCAB_SIZE_TURKISH
                           )
    print("Downloading BERT model...")
    Trainer = BertModelTrainer(HIDDEN_UNITS) 
    OUR_BERT =  Trainer(BertModel=OUR_BERT,
                        epochs=0,
                        inputs=[],
                        NSP_label=[],
                        mask_index=[],
                        mask_label=[],
                        segment=[],
                        checkpoint_path="Bert_Checkpoint/",
                        max2keep=2,
                        batch2Show=8)
    print("************************************************")
    

    
    FFN_UNITS = 512
    MyTranslator = MyBertTranslator(BERT_Model=OUR_BERT,
    								vocab_size_dec=VOCAB_SIZE_ENGLISH,
                                    d_model=HIDDEN_UNITS,
                                    nb_decoders=NB_DECODER,
                                    FFN_units=FFN_UNITS,
                                    nb_proj=NB_ATTENTION_HEAD,
                                    dropout_rate=DROPOUT,
                                    )
    
    
    
    MyTranslatorTrainer = TranslatorTrainer(HIDDEN_UNITS)
    """
    trainingTranslator = MyTranslatorTrainer(TranslatorModel=MyTranslator,
                                             epochs=1000,
                                             dataset=dataset,
                                             checkpoint_path = "Translator_Checkpoint1/",
                                             max2keep=2,
                                             batch2Show=64,
                                             )
    """
    
    
    
    
"""
from MyTranslator import TranslatorTrainer

MyTranslatorTrainer = TranslatorTrainer(HIDDEN_UNITS)
    
trainingTranslator = MyTranslatorTrainer(TranslatorModel=MyTranslator,
                                         epochs=1000,
                                         dataset=dataset,
                                         checkpoint_path = "Translator_Checkpoint1/",
                                         max2keep=2,
                                         batch2Show=64)   
"""    
    
    
    
    
    
    
    
    
    
    
    
    