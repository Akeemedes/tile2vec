import numpy as np
import sys
sys.path.append('../src')
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

class BasicBlock(tfk.Model):
    expansion = 1
    def __init__(self, in_planes, planes, strides =1, no_relu = False,\
        activation = "relu"):
        #print(planes, in_planes)
        super(BasicBlock, self).__init__()
        self.no_relu = no_relu
        self.activation = activation

        self.activation_fn = tf.nn.relu if self.activation == "relu" else tf.nn.leaky_relu\
             if self.activation =="leaky_relu" else None
        self.conv1 = tfkl.Conv2D(planes, kernel_size = 3, strides = strides, padding="same", use_bias = False)
        self.conv2 = tfkl.Conv2D(planes, kernel_size = 3, strides = 1, padding="same", use_bias = False)
        self.conv3 = tfkl.Conv2D(planes, kernel_size = 3, strides = 1, padding="same", use_bias = True)
        self.bn = tfkl.BatchNormalization()

        self.shortcut = tfk.Sequential()
        if strides != 1 or in_planes != self.expansion*planes:
            self.shortcut = tfk.Sequential([
                tfkl.Conv2D(self.expansion*planes, kernel_size=1, strides=strides, use_bias=False),
                tfkl.BatchNormalization()]
            )

    def call(self, x):
        out = self.activation_fn(self.bn(self.conv1(x)))
        #print(out.shape, x.shape)
        if self.no_relu:
            return self.bn(self.conv3(out))
        #print(self.shortcut(x).shape,self.conv2(out).shape)
        return self.activation_fn(self.bn(self.conv2(out)) + self.shortcut(x))



class Bottleneck(tfk.Model):
    expansion = 4

    def __init__(self, in_planes, planes, strides =1):
        super(Bottleneck, self).__init__()
        self.conv1 = tfkl.Conv2D(planes, kernel_size = 1, use_bias = False)
        self.conv2 = tfkl.Conv2D(planes, kernel_size = 3, strides = strides, padding = "same",use_bias = False)
        self.conv3 = tfkl.Conv1D(self.expansion*planes , kernel_size = 1, use_bias = False)
        self.bn = tfkl.BatchNormalization()
        self.shortcut = tfk.Sequential()
        if strides != 1 or in_planes != self.expansion*planes:
            self.shortcut = tfk.Sequential([tfkl.Conv2D(self.expansion*planes, kernel_size=1, strides=strides,\
                    use_bias=False), self.bn])

    def call(self, x):
        out = tfk.Sequential([self.conv1, self.bn, tfkl.ReLU, self.conv2, self.bn, tfkl.ReLU(),self.conv3, self.bn ])(x)
        return tf.nn.relu(out + self.shortcut(x))

class ResNet(tfk.Model):
    def __init__(self, block, num_blocks, n_classes =10, in_channels = 4, \
        z_dim = 512, supervised = False, no_relu = False, loss_type = 'triplet', \
            tile_size = 50, activation = 'relu'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.supervised = supervised
        self.no_relu = no_relu
        self.loss_type = loss_type
        self.tile_size = tile_size
        self.activation = activation

        self.activation_fn = tfkl.ReLU if self.activation == "relu" else tfkl.LeakyReLU\
             if self.activation =="leaky_relu" else None
        

        self.conv1 = tfkl.Conv2D(64, kernel_size=3, strides=1,
            padding="same", use_bias=False)
        self.bn = tfkl.BatchNormalization()
        self.layers_ = []
        self.layers_.append(tfk.Sequential([self.conv1, self.bn, self.activation_fn()]))
        self.layers_.append(self._make_layer(block, 64, num_blocks[0], strides=1))
        self.layers_.append(self._make_layer(block, 128, num_blocks[1], strides=2))
        self.layers_.append(self._make_layer(block, 256, num_blocks[2], strides=2))
        self.layers_.append(self._make_layer(block, 512, num_blocks[3], strides=2))
        self.layers_.append(self._make_layer(block, self.z_dim, num_blocks[4],
            strides=2, no_relu=self.no_relu))
        self.linear = tfkl.Dense(n_classes, input_shape = (self.z_dim*block.expansion,))
        

    def _make_layer(self, block, planes, num_blocks, strides, no_relu=False):
        strides = [strides] + [1]*(num_blocks-1)
        layers_ = []
        for strides in strides:
            layers_.append(block(self.in_planes, planes, strides=strides,
                no_relu=no_relu, activation=self.activation))
            self.in_planes = planes * block.expansion
        return tfk.Sequential(layers_)

    def encode(self, x, verbose = False):
        for layer in self.layers_:
            x = layer(x)
            if verbose:
                print(x.shape)
        size_pool = {50:4, 25:2, 75:5, 100:7}
        try:
            x = tfkl.AveragePooling2D(size_pool[self.tile_size])(x)
        except:
            print("tile_size error: try {}".format(size_pool.keys()))
        if verbose: print("pooling size: {}".format(x.shape))
        z = tfkl.Flatten()(x)
        if verbose: print("flattened: {}".format(z.shape))
        return z

    def call(self, x):
        z = self.encode(x)
        return self.linear(z) if self.supervised else z



#from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

print("here")


def ResNet18(n_classes=10, in_channels=4, z_dim=512, supervised=False,
    no_relu=False, loss_type='triplet', tile_size=50, activation='relu'):
    return ResNet(BasicBlock, [2,2,2,2,2], n_classes=n_classes,
        in_channels=in_channels, z_dim=z_dim, supervised=supervised,
        no_relu=no_relu, loss_type=loss_type, tile_size=tile_size, 
        activation=activation)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

def test(resnet_ = ResNet18):
    net = resnet_()
    y = net(tf.random.normal([1,50,50,4]))
    print(y.shape)
    
#test()



    

        

