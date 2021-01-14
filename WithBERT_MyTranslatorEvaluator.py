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


EnglishTokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
turkishTokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')

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
print("Downloading Translator...")  
MyTranslatorTrainer = TranslatorTrainer(HIDDEN_UNITS)
trainingTranslator = MyTranslatorTrainer(TranslatorModel=MyTranslator,
                                         epochs=0,
                                         dataset=[],
                                         checkpoint_path = "Translator_Checkpoint1/",
                                         max2keep=2,
                                        batch2Show=64,
                                        )
print("************************************************")


TurkishEncode_Id = lambda  x: np.array(turkishTokenizer(x)['input_ids'])
TurkishEncode_segment = lambda  x: np.array(turkishTokenizer(x)['token_type_ids'])+1
TurkishDecode = lambda  x: np.array(turkishTokenizer.decode(x))

EnglishEncode = lambda  x: np.array(EnglishTokenizer(x)['input_ids'])
EnglishDecode = lambda  x: np.array(EnglishTokenizer.decode(x))

initial_tok_english = 101
end_tok_english = 102
MAX_LENGTH = 20


def evaluate(inp_sentence_id,inp_sentence_segment):
    bert_input = tf.expand_dims(inp_sentence_id, axis=0)
    bert_input_segment = tf.expand_dims(inp_sentence_segment, axis=0)
    
    output = tf.expand_dims([initial_tok_english], axis=0)
    
    for _ in range(MAX_LENGTH):
        predictions = MyTranslator(output,bert_input,bert_input_segment,False)
        
        prediction = predictions[:, -1:, :]
        
        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
        
        if predicted_id == end_tok_english:
            return tf.squeeze(output, axis=0)
        
        output = tf.concat([output, predicted_id], axis=-1)
        print(EnglishDecode(tf.squeeze(output, axis=0)))
        
    return tf.squeeze(output, axis=0)


def translate(sentence):
    sentence = sentence.lower()
    sentence_id = TurkishEncode_Id(sentence)
    sentence_segment = TurkishEncode_segment(sentence)
    output = evaluate(sentence_id,sentence_segment).numpy()
    
    predicted_sentence = EnglishDecode(output)
    
    print("Input: {}".format(sentence))
    print("Predicted translation: {}".format(predicted_sentence))


#translate("merhaba")
#translate("doktor olacaksın")
#translate("Tom hala mutfakta")
#translate("Hastanede olmalısın")
#translate("neden hala okuldayız")







