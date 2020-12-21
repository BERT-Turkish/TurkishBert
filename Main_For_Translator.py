import numpy as np
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import AutoTokenizer

import random
import time

from OurTransformers import DecoderLayer
from OurBertTokenizer import MyTokenizer
from OurBertModel import MyBertModel
from MyTranslator import MyBertTranslator
from MyTranslator import TranslatorTrainer
from MyTranslator import CustomSchedule

import pandas as pd

# Türkçe yazı yazacağım, bana ingilizce çıktısını verecek
if __name__ == "__main__":
    data_src = []
    data_dest = []
    
    for line in open('EnglishTurkishCorpus.txt', encoding='UTF-8'):
        en_text, tr_text = line.rstrip().split('\t')
        
        data_src.append(tr_text)
        data_dest.append(en_text)

    EnglishTokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    turkishTokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
    fix_length = 40
    
    turkishBatch = turkishTokenizer(data_src,padding=True,truncation=True,max_length=fix_length)
    englishBatch = EnglishTokenizer(data_dest,padding=True,truncation=True,max_length=fix_length)
    
    turkishBatch_Id = np.array(turkishBatch['input_ids'])
    turkishBatch_S = np.array(turkishBatch['token_type_ids']) # of course all member is 0
    englishBatch = np.array(englishBatch['input_ids'])
    
    BATCH_SIZE = 64
    BUFFER_SIZE = 20000
    
    dataset = tf.data.Dataset.from_tensor_slices((turkishBatch_Id,turkishBatch_S,englishBatch))
    
    dataset = dataset.cache() # Just increase speed
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) # same things as the cache
    
    NB_ENCODER = 12
    NB_DECODER = 4
    FFN_UNITS = 128
    NB_ATTENTION_HEAD = 8
    HIDDEN_UNITS = 512
    VOCAB_SIZE_TURKISH = turkishTokenizer.vocab_size
    VOCAB_SIZE_ENGLISH = EnglishTokenizer.vocab_size
    DROPOUT = 0.10
    SEQ_LEN = 70
    
    OUR_BERT = MyBertModel(nb_encoder_layers=NB_ENCODER,
                           FFN_units=FFN_UNITS,
                           nb_attention_head=NB_ATTENTION_HEAD,
                           nb_hidden_units=HIDDEN_UNITS,
                           dropout_rate=DROPOUT,
                           vocab_size=VOCAB_SIZE_TURKISH
                           )
    
    ################## Load Bert ################## 
    leaning_rate = CustomSchedule(HIDDEN_UNITS)   
    optimizer = tf.keras.optimizers.Adam(leaning_rate,beta_1=0.9,beta_2=0.98,epsilon=1e-9)
    
    checkpoint_path = "Bert_Checkpoint/"
    ckpt_translator = tf.train.Checkpoint(OUR_BERT=OUR_BERT,optimizer=optimizer)
    ckpt_manager_translator = tf.train.CheckpointManager(ckpt_translator, checkpoint_path, max_to_keep=5)
    
    if ckpt_manager_translator.latest_checkpoint:
        ckpt_translator.restore(ckpt_manager_translator.latest_checkpoint)
        print("Latest checkpoint Bert restored!!")
    ############################################3
    
    MyTranslator = MyBertTranslator(vocab_size_dec=VOCAB_SIZE_ENGLISH,
                                    d_model=HIDDEN_UNITS,
                                    nb_decoders=NB_DECODER,
                                    FFN_units=FFN_UNITS,
                                    nb_proj=NB_ATTENTION_HEAD,
                                    dropout_rate=DROPOUT)
    
    TranslatorTrainer = TranslatorTrainer(HIDDEN_UNITS)
    
    trainingTranslator = TranslatorTrainer(TranslatorModel=MyTranslator,
                                           FineTunedBert=OUR_BERT,
                                           epochs=10,
                                           dataset=dataset,
                                           checkpoint_path = "MyTranslator_Checkpoint/",
                                           max2keep=3,
                                           batch2Show=50)
    
    
    