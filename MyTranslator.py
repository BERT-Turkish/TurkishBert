import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import time

from OurTransformers import DecoderLayer


############### TRANSLATOR CREATION #################################

class PositionalEncoding(layers.Layer):

    def __init__(self):
        super(PositionalEncoding, self).__init__()
    
    def get_angles(self, pos, i, d_model):
        angles = 1 / np.power(10000., (2*(i//2)) / np.float32(d_model))
        return pos * angles

    def call(self, inputs):
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]
        angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],
                                 np.arange(d_model)[np.newaxis, :],
                                 d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        pos_encoding = angles[np.newaxis, ...]
        return inputs + tf.cast(pos_encoding, tf.float32)
    
class MyBertTranslator(tf.keras.Model):
    
    def __init__(self,
                 vocab_size_dec,
                 d_model,
                 nb_decoders,
                 FFN_units,
                 nb_proj,
                 dropout_rate,
                 name="Translator"):
        super(MyBertTranslator, self).__init__(name=name)
        self.nb_decoders = nb_decoders
        self.d_model = d_model
        self.nb_decoders = nb_decoders
        
        self.embedding = layers.Embedding(vocab_size_dec,self.d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)
        
        self.dec_layers = [DecoderLayer(FFN_units,nb_proj,dropout_rate) 
                           for i in range(nb_decoders)]
        
        self.last_linear = layers.Dense(units=vocab_size_dec, name="lin_ouput")
        
    def create_padding_mask(self, seq):
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, seq):
        seq_len = tf.shape(seq)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return look_ahead_mask
        
    def call(self,inputs,bert_embedded_outputs,bert_inputs,training):
        mask_1 = tf.maximum(self.create_padding_mask(inputs),
                            self.create_look_ahead_mask(inputs)
                            )
        
        mask_2 = self.create_padding_mask(bert_inputs)
            
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)
            
        for i in range(self.nb_decoders):
            outputs = self.dec_layers[i](outputs,
                                         bert_embedded_outputs,
                                         mask_1,
                                         mask_2,
                                         training)
            
        outputs = self.last_linear(outputs)
            
        return outputs
    
############### TRAIN PART #################################

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
class TranslatorTrainer():
    def __init__(self,HIDDEN_UNITS):
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction="none")
        
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")

        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
        
        leaning_rate = CustomSchedule(HIDDEN_UNITS)
        
        self.optimizer = tf.keras.optimizers.Adam(leaning_rate,
                                                  beta_1=0.9,
                                                  beta_2=0.98,
                                                  epsilon=1e-9)
    
    def loss_function(self,target, pred):
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss_ = self.loss_object(target, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)
    
    def CheckPoint_Model(self,TranslatorModel,checkpoint_path,max_to_keep):
        self.ckpt = tf.train.Checkpoint(TranslatorModel=TranslatorModel,optimizer=self.optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=max_to_keep)

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored!!")
        
    def __call__(self,
                 TranslatorModel,
                 FineTunedBert,
                 epochs,
                 dataset,
                 checkpoint_path = "",
                 max2keep=0,
                 batch2Show=1):
        
        self.CheckPoint_Model(TranslatorModel,checkpoint_path,max2keep)
        
        for epoch in range(epochs):
            print("Start of epoch {}".format(epoch+1))
            start = time.time()
            
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            
            for (batch,(turkishSentence,turkishSegment,englishSentence)) in enumerate(dataset):
                Inputs = englishSentence[:,:-1]
                Real_Inputs = englishSentence[:,1:]
                turkishBertEmbedded = FineTunedBert(turkishSentence,turkishSegment,False)
                
                with tf.GradientTape() as tape:
                    predictions = TranslatorModel(Inputs,turkishBertEmbedded,turkishSentence,True)
                    loss = self.loss_function(Real_Inputs, predictions)
                    
                gradients = tape.gradient(loss, TranslatorModel.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, TranslatorModel.trainable_variables))
                
                self.train_loss(loss)
                self.train_accuracy(Real_Inputs, predictions)
                
                if batch % batch2Show == 0:
                    print("Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                        epoch+1, batch, self.train_loss.result(), self.train_accuracy.result()))
                    grad_list = [grad for grad in gradients if grad is not None]
                    print("Number of not None grads MLM is: {} for MLM".format(len(grad_list)))
                    print("ALL Trainable Variables:{}".format(len(TranslatorModel.trainable_variables)))
                    print("*************************************************************")
                    
            
            ckpt_save_path = self.ckpt_manager.save()
            print("Saving checkpoint for epoch {} at {}".format(epoch+1,ckpt_save_path))
            print("Time taken for 1 epoch: {} secs\n".format(time.time() - start))
    
        return TranslatorModel    
    
    
    
    
    