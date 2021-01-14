import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import time
import re

from OurTransformers import DecoderLayer
from tensorflow.keras import initializers


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
    			 BERT_Model,
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
        
        self.embedding = layers.Embedding(vocab_size_dec,self.d_model,
                                          embeddings_initializer=initializers.RandomNormal(stddev=0.01,seed=3))
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)

        self.BERT_Model = BERT_Model
        
        self.dec_layers = [DecoderLayer(FFN_units,nb_proj,dropout_rate) 
                           for i in range(nb_decoders)]
        
        self.last_linear = layers.Dense(units=vocab_size_dec, name="lin_output",
                                        kernel_initializer=initializers.RandomNormal(stddev=0.01,seed=3),
                                        bias_initializer=initializers.Zeros())
        
    def create_padding_mask(self, seq):
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, seq):
        seq_len = tf.shape(seq)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return look_ahead_mask
        
    def call(self,inputs,bert_inputs,bert_segment,training):
        mask_1 = tf.maximum(self.create_padding_mask(inputs),
                            self.create_look_ahead_mask(inputs)
                            )
        
        mask_2 = self.create_padding_mask(bert_inputs)
        
        bert_output = self.BERT_Model(bert_inputs,bert_segment,training)

        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)
            
        for i in range(self.nb_decoders):
            outputs = self.dec_layers[i](outputs,
                                         bert_output,
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
        self.loss_object = tf.compat.v1.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction="none")
        
        self.train_loss = tf.compat.v1.keras.metrics.Mean(name="train_loss")

        self.train_accuracy = tf.compat.v1.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
        
        
        #leaning_rate = CustomSchedule(HIDDEN_UNITS)
        leaning_rate = 3e-5
        self.optimizer = tf.compat.v1.keras.optimizers.Adam(leaning_rate,
                                                            beta_1=0.9,
                                                            beta_2=0.98,
                                                            epsilon=1e-9)
        """
        self.learning_rate = 3e-5
        self.num_train_steps = 1000
        self.num_warmup_steps = None#100
        self.global_step = 1
        
        self.optimizer = create_optimizer(self.learning_rate,
                                          self.num_train_steps,
                                          self.num_warmup_steps,
                                          self.global_step)
        """
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
            
            """
            self.optimizer = create_optimizer(self.learning_rate,
                                              self.num_train_steps,
                                              self.num_warmup_steps,
                                              self.global_step)
            self.global_step = self.global_step+1
            """
            for (batch,(turkishSentence,turkishSegment,englishSentence)) in enumerate(dataset):
                Inputs = englishSentence[:,:-1]
                Real_Inputs = englishSentence[:,1:]
                
                with tf.GradientTape() as tape:
                    predictions = TranslatorModel(Inputs,turkishSentence,turkishSegment,True)
                    loss = self.loss_function(Real_Inputs, predictions)
                    
                gradients = tape.gradient(loss, TranslatorModel.trainable_variables)
                
                #(self.gradients, _) = tf.clip_by_global_norm(gradients, clip_norm=1.0)
                #tvars = TranslatorModel.trainable_variables
                
                self.optimizer.apply_gradients(zip(gradients, TranslatorModel.trainable_variables))
                """
                train_op = self.optimizer.apply_gradients(zip(self.gradients, tvars), 
                                                              global_step=self.global_step,
                                                              name=None)

                train_op = tf.group(train_op, tf.convert_to_tensor([self.global_step],tf.float32))
                """
                
                self.train_loss(loss)
                self.train_accuracy(Real_Inputs, predictions)
                
                if (batch % batch2Show == 0 and batch != 0) or batch == len(dataset)-1:
                    print("Epoch {} Batch {} Loss {:.6f} Accuracy {:.6f}".format(
                        epoch+1, batch, self.train_loss.result(), self.train_accuracy.result()))
                    grad_list = [grad for grad in gradients if grad is not None]
                    print("Number of not None grads is: {} ".format(len(grad_list)))
                    print("ALL Trainable Variables:{}".format(len(TranslatorModel.trainable_variables)))
                    print("*************************************************************")
                    
            
            ckpt_save_path = self.ckpt_manager.save()
            print("Saving checkpoint for epoch {} at {}".format(epoch+1,ckpt_save_path))
            print("Time taken for 1 epoch: {} secs\n".format(time.time() - start))
    
        return TranslatorModel    
    
    
    
def create_optimizer(init_lr, num_train_steps, num_warmup_steps,global_step):
    
    #global_step = tf.compat.v1.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)


    learning_rate = tf.compat.v1.train.polynomial_decay(learning_rate,
                                        global_step,
                                        num_train_steps,
                                        end_learning_rate=8e-8,
                                        power=1.0,
                                        cycle=False)
       
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
        
        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
        
        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done
        
        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = ((1.0 - is_warmup) * learning_rate.func(global_step-1) + is_warmup * warmup_learning_rate)
        
    else:
        learning_rate = learning_rate.func(global_step-1)
        

    optimizer = AdamWeightDecayOptimizer(learning_rate=learning_rate,
                                         weight_decay_rate=0.01,
                                         beta_1=0.9,
                                         beta_2=0.999,
                                         epsilon=3e-7,
                                         exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    
    return optimizer    

class AdamWeightDecayOptimizer(tf.compat.v1.train.Optimizer):
    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)
    
            m = tf.compat.v1.get_variable(name=param_name + "/adam_m",
                                          shape=param.shape.as_list(),
                                          dtype=tf.float32,
                                          trainable=False,
                                          initializer=tf.zeros_initializer())
            v = tf.compat.v1.get_variable(name=param_name + "/adam_v",
                                          shape=param.shape.as_list(),
                                          dtype=tf.float32,
                                          trainable=False,
                                          initializer=tf.zeros_initializer())
            
            # Standard Adam update.
            next_m = (tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,tf.square(grad)))
              

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend([param.assign(next_param),
                               m.assign(next_m),
                               v.assign(next_v)])
            
        return tf.group(*assignments, name=name)
    
    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name