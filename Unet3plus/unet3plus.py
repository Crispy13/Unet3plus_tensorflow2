import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow_probability as tfp


###
class DiceCoefficient(keras.metrics.Metric):
    def __init__(self, calc_axis = [-3, -2], epsilon = 1e-8, name='dice_coefficient', **kwargs):
        super().__init__(name=name, **kwargs)
        self.dice_coef = self.add_weight(name='dc', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.calc_axis = calc_axis
        self.epsilon = epsilon
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
#         print("dicemetric: input shapes = ", y_true.shape, y_pred.shape)
        
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis = self.calc_axis)
        dn = tf.reduce_sum(tf.square(y_true) + tf.square(y_pred), axis = self.calc_axis)
        
        dice = tf.reduce_mean((2 * intersection + self.epsilon) / (dn + self.epsilon), axis = None)
#         print("dicemetric: dice values, shape = ", dice, dice.shape)
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)

        self.dice_coef.assign_add(dice)
        self.count.assign_add(1)
#         print("dicemetric: self.count = ", self.count)
        
    def result(self):
        return self.dice_coef / self.count

    
###    
def no_regularizer(x):
    return None

    
###
class Unet3plus(keras.Model):
    """
    Unet3plus implementation
    
    The encoders have 2 convolution layers each.
    
    """
    def __init__(self, kernel_regularizer = None, bias_regularizer = None, data_format = 'channels_last', kernel_initializer = "he_normal", 
                 acc_threshold = 0.5, reg_lambda = None, loss_weights = None, encoder_conv_n = 2, 
                 **kwargs):
        """
        kernel_regularizer : a regularizer class of keras
            e.g. keras.regularizers.l2
            
        bias_regularizer : a regularizer class of keras
            e.g. keras.regularizers.l2
            
        reg_lambda : a float
            lambda value for regularization.
        """
        super().__init__(**kwargs)
        
        
        ### Check passed arguments
        if kernel_regularizer:
            assert reg_lambda, "`reg_lambda` should be designated when using `kernel_regularizer`."
        
        
        ### Parameter settings
        if kernel_regularizer is None:
            kernel_regularizer = no_regularizer
            
        if bias_regularizer is None:
            bias_regularizer = no_regularizer
        
        
        # loss weights
        if loss_weights is None:
            self.loss_weights = [1., 1., 1., 1., 1.]
        else:
            assert len(loss_weights) == 5
            self.loss_weights = loss_weights
        
        ### losses and metrics
        self.hybrid_loss = HybridLoss()
        self.bce_loss = keras.losses.BinaryCrossentropy()
        
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.dice_metric = DiceCoefficient(name="dice")
        self.bloss_tracker = keras.metrics.Mean(name="bce_loss")
        self.accuracy_metric = keras.metrics.BinaryAccuracy(name="accuracy", threshold = acc_threshold)
        
        self.hloss_tracker = keras.metrics.Mean(name="hybrid_loss")
        self.hloss1_tracker = keras.metrics.Mean(name="hybrid_loss1")
        self.hloss2_tracker = keras.metrics.Mean(name="hybrid_loss2")
        self.hloss3_tracker = keras.metrics.Mean(name="hybrid_loss3")
        self.hloss4_tracker = keras.metrics.Mean(name="hybrid_loss4")
        self.hloss5_tracker = keras.metrics.Mean(name="hybrid_loss5")
    
    
        self.dice1_metric = DiceCoefficient(name="dice1")
        self.dice2_metric = DiceCoefficient(name="dice2")
        self.dice3_metric = DiceCoefficient(name="dice3")
        self.dice4_metric = DiceCoefficient(name="dice4")
        self.dice5_metric = DiceCoefficient(name="dice5")
        
        
        ### layers definition 
        self.ld = dict()
        
        ## enc1
        self.conv1 = UnetConv2(filters = 64, kernel_size = (3, 3), data_format = data_format, kernel_initializer = "he_normal",
                                 kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer, strides = (1, 1), n = encoder_conv_n, 
                               reg_lambda = reg_lambda,
                                 name = 'conv1')
        self.conv1_maxpool = keras.layers.MaxPool2D(pool_size = 2, data_format = data_format, name = f"conv1_maxpool")
        
    
        ## enc2
        self.conv2 = UnetConv2(filters = 128, kernel_size = (3, 3), data_format = data_format, kernel_initializer = "he_normal",
                                 kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer, strides = (1, 1), n = encoder_conv_n, 
                               reg_lambda = reg_lambda,
                                 name = 'conv2')
        self.conv2_maxpool = keras.layers.MaxPool2D(pool_size = 2, data_format = data_format, name = f"conv2_maxpool")
            
            
        ## enc3
        self.conv3 = UnetConv2(filters = 256, kernel_size = (3, 3), data_format = data_format, kernel_initializer = "he_normal",
                                 kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer, strides = (1, 1), n = encoder_conv_n, 
                               reg_lambda = reg_lambda,
                                 name = 'conv3')
        self.conv3_maxpool = keras.layers.MaxPool2D(pool_size = 2, data_format = data_format, name = f"conv3_maxpool")
        
        
        ## enc4
        self.conv4 = UnetConv2(filters = 512, kernel_size = (3, 3), data_format = data_format, kernel_initializer = "he_normal",
                                 kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer, strides = (1, 1), n = encoder_conv_n, 
                               reg_lambda = reg_lambda,
                                 name = 'conv4')
        self.conv4_maxpool = keras.layers.MaxPool2D(pool_size = 2, data_format = data_format, name = f"conv4_maxpool")
        
        
        ## enc5
        self.conv5 = UnetConv2(filters = 1024, kernel_size = (3, 3), data_format = data_format, kernel_initializer = "he_normal",
                                 kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer, strides = (1, 1), n = encoder_conv_n, 
                               reg_lambda = reg_lambda,
                                 name = 'conv5')
        
        
        self.enc5_ds = Unet3plus_deep_supervision(upsample_size = 2 ** 4, name = "enc5_ds",
                                                 kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer, reg_lambda = reg_lambda
                                                 )
        
        # cgm
        self.cgm_dropout = keras.layers.SpatialDropout2D(rate = 0.5, data_format = data_format)
        self.cgm_conv = keras.layers.Conv2D(filters = 1, kernel_size = (1,1), strides = (1, 1), padding = 'same',
                                                kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer,
                                                data_format = data_format, kernel_initializer = kernel_initializer,
                                                name = f"cgm_conv")
        self.cgm_pool = tfa.layers.AdaptiveMaxPooling2D(output_size = 1, data_format = data_format, name = "cgm_adaptive_pool")
        self.cgm_sig = keras.layers.Activation('sigmoid', name = 'cgm_sigmoid', dtype='float32')
        
        ## dec 4
        self.dec4_fullscale = FullScaleBlock(num_smaller_scale = 3, name = "dec4",
                                            kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer, reg_lambda = reg_lambda)
        
        ## dec 3
        self.dec3_fullscale = FullScaleBlock(num_smaller_scale = 2, name = "dec3",
                                            kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer, reg_lambda = reg_lambda)
        
        ## dec 2
        self.dec2_fullscale = FullScaleBlock(num_smaller_scale = 1, name = "dec2",
                                            kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer, reg_lambda = reg_lambda)
        
        ## dec 1
        self.dec1_fullscale = FullScaleBlock(num_smaller_scale = 0, name = "dec1",
                                            kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer, reg_lambda = reg_lambda)
        
        
    def build(self, batch_input_shape):
        super().build(batch_input_shape)

        
    def call(self, inputs):
        ### ouput node connection
        Z = inputs
            
            
        ## conv1    
        Z1 = self.conv1(Z)
        Zo = self.conv1_maxpool(Z1)
        
        ## conv2    
        Z2 = self.conv2(Zo)
        Zo = self.conv2_maxpool(Z2)
        
        ## conv3    
        Z3 = self.conv3(Zo)
        Zo = self.conv3_maxpool(Z3)
        
        ## conv4    
        Z4 = self.conv4(Zo)
        Zo = self.conv4_maxpool(Z4)
        
        ## conv5    
        Zd5 = self.conv5(Zo)
        
        Zsd5 = self.enc5_ds(Zd5)
        
        # CGM
        Z = self.cgm_dropout(Zd5)
        Z = self.cgm_conv(Z)
        Z = self.cgm_pool(Z)
        Zco = self.cgm_sig(Z)
        Zc = tf.where(Zco > 0.5, 1., 0.)
        Zco = tf.squeeze(Zco, axis = [-3, -2, -1], name = "Zco_squeeze")
        
        Zsd5 = tf.multiply(Zsd5, Zc, name = "enc5_multiply")
        
        
        ## dec4
#         print(Z1.shape, Z2.shape, Z3.shape, Z4.shape, Zd5.shape)
        Zd4, Zsd4 = self.dec4_fullscale([Z1, Z2, Z3, Z4, Zd5])
        Zsd4 = tf.multiply(Zsd4, Zc, name = "enc4_multiply")
        
        ## dec3
        Zd3, Zsd3 = self.dec3_fullscale([Z1, Z2, Z3, Zd4, Zd5]) 
        Zsd3 = tf.multiply(Zsd3, Zc, name = "enc3_multiply")
        
        ## dec2
        Zd2, Zsd2 = self.dec2_fullscale([Z1, Z2, Zd3, Zd4, Zd5]) 
        Zsd2 = tf.multiply(Zsd2, Zc, name = "enc2_multiply")
        
        ## dec1
        Zd1, Zsd1 = self.dec1_fullscale([Z1, Zd2, Zd3, Zd4, Zd5]) 
        Zsd1 = tf.multiply(Zsd1, Zc, name = "enc1_multiply")
        
        
        return Zsd1, Zsd2, Zsd3, Zsd4, Zsd5, Zco
    
    
    def train_step(self, data):
        x, y = data
        y_true_cls = y[-1]
        y_true_seg = tf.stack((y[0], ) * 5, axis = 0)
        y_true_seg = tf.transpose(y_true_seg, perm = [1, 0, 2, 3, 4])
        
        with tf.GradientTape(persistent = True) as tape:
            y_pred = self(x, training = True)
            
            y_pred_seg = y_pred[:-1]
            y_pred_cls = y_pred[-1]
            
            y_pred_seg = tf.stack(y_pred_seg, axis = 0)
            y_pred_seg = tf.transpose(y_pred_seg, perm = [1, 0, 2, 3, 4])
            
            hloss1 = self.hybrid_loss(y[0], y_pred[0]) 
            hloss2 = self.hybrid_loss(y[0], y_pred[1]) 
            hloss3 = self.hybrid_loss(y[0], y_pred[2]) 
            hloss4 = self.hybrid_loss(y[0], y_pred[3]) 
            hloss5 = self.hybrid_loss(y[0], y_pred[4]) 
            hloss = hloss1 * self.loss_weights[0] + hloss2 * self.loss_weights[1] + hloss3 * self.loss_weights[2] + hloss4 * self.loss_weights[3] + hloss5 * self.loss_weights[4]
            bloss = self.bce_loss(y_true_cls, y_pred_cls)
            loss = hloss + bloss + sum(self.losses)
            
            
        #compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
#         gradients_b = tape.gradient(bloss, trainable_vars)
        
        del tape
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#         self.optimizer.apply_gradients(zip(gradients_b, trainable_vars))

        # l2 reg
#         for variable in self.variables:
#             if variable.constraint is not None: 
#                 variable.assign(variable.constraint(variable))
        
        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.hloss_tracker.update_state(hloss)
        
        self.hloss1_tracker.update_state(hloss1)
        self.hloss2_tracker.update_state(hloss2)
        self.hloss3_tracker.update_state(hloss3)
        self.hloss4_tracker.update_state(hloss4)
        self.hloss5_tracker.update_state(hloss5)
        
        self.bloss_tracker.update_state(bloss)
        self.dice_metric.update_state(y_true_seg, y_pred_seg)
        self.accuracy_metric.update_state(y_true_cls, y_pred_cls)
        self.dice1_metric.update_state(y[0], y_pred[0])
        self.dice2_metric.update_state(y[0], y_pred[1])
        self.dice3_metric.update_state(y[0], y_pred[2])
        self.dice4_metric.update_state(y[0], y_pred[3])
        self.dice5_metric.update_state(y[0], y_pred[4])
        
        return {"loss": self.loss_tracker.result(), "hybrid_loss": self.hloss_tracker.result(),
                "hybrid_loss1": self.hloss1_tracker.result(), "hybrid_loss2": self.hloss2_tracker.result(), "hybrid_loss3": self.hloss3_tracker.result(),
                "hybrid_loss4": self.hloss4_tracker.result(),  "hybrid_loss5": self.hloss5_tracker.result(), "binary_crossentropy": self.bloss_tracker.result(),
                "dice": self.dice_metric.result(), 
                "dice1": self.dice1_metric.result(), "dice2": self.dice2_metric.result(), "dice3": self.dice3_metric.result(),
                "dice4": self.dice4_metric.result(), "dice5": self.dice5_metric.result(), "accuracy": self.accuracy_metric.result()}
        
        
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.hloss_tracker, 
                self.hloss1_tracker, self.hloss2_tracker, self.hloss3_tracker, self.hloss4_tracker, self.hloss5_tracker, self.bloss_tracker,
                self.dice_metric,
                self.dice1_metric, self.dice2_metric, self.dice3_metric, self.dice4_metric, self.dice5_metric, self.accuracy_metric]
    
    
    
    def test_step(self, data):
        x, y = data
        y_true_cls = y[1]
        y_true_seg = tf.stack((y[0], ) * 5, axis = 0)
        y_true_seg = tf.transpose(y_true_seg, perm = [1, 0, 2, 3, 4])
        
        ##
        y_pred = self(x, training = False)
        
        y_pred_seg = y_pred[:-1]
        y_pred_cls = y_pred[-1]

        y_pred_seg = tf.stack(y_pred_seg, axis = 0)
        y_pred_seg = tf.transpose(y_pred_seg, perm = [1, 0, 2, 3, 4])
        
        ##
        hloss1 = self.hybrid_loss(y[0], y_pred[0]) 
        hloss2 = self.hybrid_loss(y[0], y_pred[1]) 
        hloss3 = self.hybrid_loss(y[0], y_pred[2]) 
        hloss4 = self.hybrid_loss(y[0], y_pred[3]) 
        hloss5 = self.hybrid_loss(y[0], y_pred[4]) 
        hloss = hloss1 * self.loss_weights[0] + hloss2 * self.loss_weights[1] + hloss3 * self.loss_weights[2] + hloss4 * self.loss_weights[3] + hloss5 * self.loss_weights[4]
        bloss = self.bce_loss(y_true_cls, y_pred_cls)
        loss = hloss + bloss + sum(self.losses)
        
        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.hloss_tracker.update_state(hloss)
        
        self.hloss1_tracker.update_state(hloss1)
        self.hloss2_tracker.update_state(hloss2)
        self.hloss3_tracker.update_state(hloss3)
        self.hloss4_tracker.update_state(hloss4)
        self.hloss5_tracker.update_state(hloss5)
        
        self.bloss_tracker.update_state(bloss)
        self.dice_metric.update_state(y_true_seg, y_pred_seg)
        self.accuracy_metric.update_state(y_true_cls, y_pred_cls)
        self.dice1_metric.update_state(y[0], y_pred[0])
        self.dice2_metric.update_state(y[0], y_pred[1])
        self.dice3_metric.update_state(y[0], y_pred[2])
        self.dice4_metric.update_state(y[0], y_pred[3])
        self.dice5_metric.update_state(y[0], y_pred[4])
        
        return {m.name: m.result() for m in self.metrics}
    
    
###
class UnetConv2(keras.layers.Layer):
    def __init__(self,
                 filters, kernel_size = (3, 3), data_format = 'channels_last', kernel_initializer = "he_normal",
                 kernel_regularizer = None, bias_regularizer = None, strides = (1, 1), n = 2, reg_lambda = None,
                 name = '', **kwargs):
        
        super().__init__(name, **kwargs)
        
        self.ld = dict()
        self.n = n
        
        for i in range(1, n+1):
            self.ld['conv{}'.format(i)] = keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides=strides, padding = "same",
                                                     kernel_regularizer = kernel_regularizer(reg_lambda), bias_regularizer = bias_regularizer(reg_lambda),
                                                        data_format = data_format, kernel_initializer = kernel_initializer, name = f"{name}_unetconv2_conv{i}")
            self.ld['bn{}'.format(i)] = keras.layers.BatchNormalization(axis = -1, name = f"{name}_unetconv2_bn{i}")
            self.ld['relu{}'.format(i)] = keras.layers.Activation('relu', name=f"{name}_unetconv2_relu{i}")
        
        self.hidden = [l for l in self.ld.values()]
        
    def build(self, batch_input_size):
        super().build(batch_input_size)
        
    def call(self, inputs):
        Z = inputs
        
        for i in range(1, self.n + 1):
            Z = self.ld['conv{}'.format(i)](Z)
            Z = self.ld['bn{}'.format(i)](Z)
            Z = self.ld['relu{}'.format(i)](Z)
        
        return Z
    
### DiceLoss
class HybridLoss(keras.losses.Loss):
    def __init__(self, calc_axis = [-3, -2], epsilon = 1e-8, **kwargs):
        """
        calc_axis : axis when calculating intersection and union. 
                    e.g. when using 2d images and the data format is 'channels_last', axis could be (-3, -2)
        """
        super().__init__(**kwargs)
        
        self.fl = tfa.losses.SigmoidFocalCrossEntropy()
        self.iou = IoULoss_()
        self.ms_ssim = MS_SSIM_Loss()
        
        self.calc_axis = calc_axis
        self.epsilon = epsilon
        
    
    def call(self, y_true, y_pred):
        """
        y_true : segmentation ground truth
        y_pred : 5 segmentation predictions (Deep supervision)
        """
#         y_true = tf.stack((y_true, ) * 5, axis = 0)
#         y_pred = tf.stack(y_pred, axis = 0)
        
#         y_true = tf.transpose(y_true, perm = [1, 0, 2, 3, 4])
#         y_pred = tf.transpose(y_pred, perm = [1, 0, 2, 3, 4])
        
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        
        # Focal loss
        fl = tf.reduce_mean(self.fl(y_true, y_pred), axis = (-3, -2, -1))
        
        # IoU loss
        il = tf.reduce_mean(self.iou(y_true, y_pred), axis = -1)
            
        # MS-SSIM loss
        ml = self.ms_ssim(y_true, y_pred)
        
        return fl + il + ml
    
    
###
class Unet3plus_deep_supervision(keras.layers.Layer):
    def __init__(self, upsample_size,
                 filters = 1, kernel_size = (3, 3), data_format = 'channels_last', kernel_initializer = "he_normal",
                 kernel_regularizer = None, bias_regularizer = None, strides = (1, 1), reg_lambda = None,
                 name = '', **kwargs):
        
        super().__init__(name, **kwargs)
        
        self.ld = dict()
        self.ld['conv'] = keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides=strides, padding = "same",
                                         kernel_regularizer = kernel_regularizer(reg_lambda), bias_regularizer = bias_regularizer(reg_lambda),
                                            data_format = data_format, kernel_initializer = kernel_initializer, name = f"{name}_ds_conv")
        self.ld['bilinear_upsample'] = keras.layers.UpSampling2D(size = upsample_size, data_format = data_format,
                                                                                  interpolation = 'bilinear', name = f"{name}_ds_bilinear_upsample")
        self.ld['sigmoid'] = keras.layers.Activation('sigmoid', dtype= 'float32', name=f"{name}_sigmoid")
        
        self.hidden = [l for l in self.ld.values()]
        
    def build(self, batch_input_size):
        super().build(batch_input_size)
        
    def call(self, inputs):
        Z = inputs
        
        for layer in self.hidden:
            Z = layer(Z)
        
        return Z
        

        
class FullScaleBlock(keras.layers.Layer):

    def __init__(self, num_smaller_scale,
                 filters = 64, kernel_size = (3, 3), data_format = 'channels_last', kernel_initializer = "he_normal",
                 kernel_regularizer = None, bias_regularizer = None, reg_lambda = None, strides = (1, 1),
                 first_conv2d_strides = (1, 1), name = '', **kwargs):

        super().__init__(**kwargs)
        
        self.ld = {}
        self.nsc = num_smaller_scale
        
        for i in range(1, num_smaller_scale + 1):
            self.ld['maxpool_{}'.format(i)] = keras.layers.MaxPool2D(pool_size = 2 ** (num_smaller_scale - i + 1),
                                                                     data_format = data_format, name = f"{name}_fsb_maxpool_{i}")
        
        for i in range(1, 5 + 1):
            self.ld['conv_{}'.format(i)] = keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same',
                                                                kernel_regularizer = kernel_regularizer(reg_lambda), bias_regularizer = bias_regularizer(reg_lambda),
                                                                data_format = data_format, kernel_initializer = kernel_initializer,
                                                                name = f"{name}_fsb_conv_{i}")
            self.ld['bn_{}'.format(i)] = keras.layers.BatchNormalization(axis = -1, name = f"{name}_fsb_bn_{i}")
            self.ld['relu_{}'.format(i)] = keras.layers.ReLU(name = f"{name}_fsb_relu_{i}")
            
        for i in range(num_smaller_scale + 2, 5 + 1):
            self.ld['bilinear_upsample_{}'.format(i)] = keras.layers.UpSampling2D(size = 2 ** (i - self.nsc - 1), data_format = data_format,
                                                                                  interpolation = 'bilinear', name = f"{name}_fsb_bilinear_upsample_{i}")
        
        self.ld['concat'] = keras.layers.Concatenate(axis = -1, name = f"{name}_fsb_concat")
        
        self.ld['fusion_conv'] = keras.layers.Conv2D(filters = 320, kernel_size = kernel_size, strides = strides, padding = 'same',
                                                kernel_regularizer = kernel_regularizer(reg_lambda), bias_regularizer = bias_regularizer(reg_lambda),
                                                data_format = data_format, kernel_initializer = kernel_initializer,
                                                name = f"{name}_fsb_fusion_conv")
        self.ld['fusion_bn'] = keras.layers.BatchNormalization(axis = -1, name = f"{name}_fusion_bn")
        self.ld['fusion_relu'] = keras.layers.ReLU(name = f"{name}_fusion_relu")
        
        ## side output path
        self.ld['sd_conv'] = keras.layers.Conv2D(filters = 1, kernel_size = (3, 3), strides = strides, padding = 'same',
                                                        kernel_regularizer = kernel_regularizer(reg_lambda), bias_regularizer = bias_regularizer(reg_lambda),
                                                        data_format = data_format, kernel_initializer = kernel_initializer,
                                                        name = f"{name}_fsb_sd_conv")
        self.ld['sd_biliear_upsample'] = keras.layers.UpSampling2D(size = 2 ** (self.nsc), data_format = data_format,
                                                                                  interpolation = 'bilinear', name = f"{name}_fsb_sd_bilinear_upsample")
        self.ld['sd_sigmoid'] = keras.layers.Activation('sigmoid', dtype = 'float32', name = f"{name}_sd_sigmoid")
        
        
        self.hidden = [l for l in self.ld.values()]
    
    def build(self, batch_input_shape):
        super().build(batch_input_shape)
    
        
    def call(self, inputs):
        inp = inputs
        inp_d = dict(('inp{}'.format(i+1), inp[i]) for i in range(len(inputs)))
        
        for i in range(1, self.nsc + 1):
            inp_d['inp{}'.format(i)] = self.ld['maxpool_{}'.format(i)](inp_d['inp{}'.format(i)])
            inp_d['inp{}'.format(i)] = self.ld['conv_{}'.format(i)](inp_d['inp{}'.format(i)])
            inp_d['inp{}'.format(i)] = self.ld['bn_{}'.format(i)](inp_d['inp{}'.format(i)])
            inp_d['inp{}'.format(i)] = self.ld['relu_{}'.format(i)](inp_d['inp{}'.format(i)])
            
        inp_d['inp{}'.format(self.nsc + 1)] = self.ld['conv_{}'.format(self.nsc + 1)](inp_d['inp{}'.format(self.nsc + 1)])
        inp_d['inp{}'.format(self.nsc + 1)] = self.ld['bn_{}'.format(self.nsc + 1)](inp_d['inp{}'.format(self.nsc + 1)])
        inp_d['inp{}'.format(self.nsc + 1)] = self.ld['relu_{}'.format(self.nsc + 1)](inp_d['inp{}'.format(self.nsc + 1)])
        
        for i in range(self.nsc + 2, 5 + 1):
#             print("before:", inp_d['inp{}'.format(i)].shape)
            inp_d['inp{}'.format(i)] = self.ld['bilinear_upsample_{}'.format(i)](inp_d['inp{}'.format(i)])
            inp_d['inp{}'.format(i)] = self.ld['conv_{}'.format(i)](inp_d['inp{}'.format(i)])
            inp_d['inp{}'.format(i)] = self.ld['bn_{}'.format(i)](inp_d['inp{}'.format(i)])
            inp_d['inp{}'.format(i)] = self.ld['relu_{}'.format(i)](inp_d['inp{}'.format(i)])
#             print("after:",inp_d['inp{}'.format(i)].shape)
            
        Z = self.ld['concat']([inp_d['inp1'], inp_d['inp2'], inp_d['inp3'], inp_d['inp4'], inp_d['inp5']])
        Z = self.ld['fusion_conv'](Z)
        Z = self.ld['fusion_bn'](Z)
        Z = self.ld['fusion_relu'](Z)
        
        # side output
        Zs = self.ld['sd_conv'](Z)
        
        if self.nsc != 0:
            Zs = self.ld['sd_biliear_upsample'](Zs)
            
        Zs = self.ld['sd_sigmoid'](Zs)
        
        return Z, Zs
    
    
###
def IoULoss_(calc_axis = (-3, -2, -1), smooth = 1e-8):
    def IoULoss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)

        intersection = tf.reduce_sum(tf.multiply(y_true, y_pred), axis = calc_axis)
        total = tf.reduce_sum(y_true, axis = calc_axis) + tf.reduce_sum(y_pred, axis = calc_axis)
        union = total - intersection
        
        IoU = (intersection + smooth) / (union + smooth)
        
        return 1 - IoU
    
    return IoULoss


###
class MS_SSIM_Loss(keras.losses.Loss):
    """
    
    """
    def __init__(self, calc_axis = (-3, -2, -1), max_val = 1.0, **kwargs):
        super().__init__(**kwargs)
        
        self.calc_axis = calc_axis
        self.max_val = max_val
        
        
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        ms_ssim = tf.image.ssim_multiscale(y_true, y_pred, max_val = self.max_val)
        
        return 1 - ms_ssim