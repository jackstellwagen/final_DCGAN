import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io
import scipy
import time


tf.reset_default_graph()

def noise_array():
    return np.random.normal(scale = 0.05, size =(500,36))




def gen_data():

    perfect_data = np.indices((500,36))[1]
    sin = lambda t: np.sin(1/np.random.choice((2,6,8,12)) *t)
    data = np.apply_along_axis(sin, 1, perfect_data)

    return data + noise_array()


#data = gen_data()
#data = np.load("/home/jack/caltech_research/tissue_data_jack_GAN/tissue_data_jack/tissue_data_jack_normalized.npy")


#n_samp, n_input = data.shape


def find_layers(n_input):
    
    layers = 0 
    current_base = n_input
    valid_nums = [5,6,7,8,9,11,13,17,19,21]
    for i in range(7):
       if current_base%2 ==0:
          if current_base/2 in valid_nums:
             return i , int(current_base/2)
       else:
          return None
       current_base /= 2
    return 6, int(current_base)

    """
    valid_nums = [5,6,7,8,9,11,13,17,19,21]
    for num in valid_nums:
        layers = np.log2(n_input/num)
        if layers == int(layers):
            return int(layers)-1, num
    """
     
#print(n_input, find_layers(n_input))
def generate_filters_gen(layer_count, gen_filters):
    filters = []
    for i in range(layer_count):
        filters += [int(gen_filters/2)]
        gen_filters /= 2
    return filters

def generate_filters_disc(layer_count, disc_filters):
    filters = []
    disc_filters_max = disc_filters
    for i in range(layer_count):
        filters += [int(disc_filters/2)]
        disc_filters /= 2
    return filters[::-1] + [disc_filters_max]


    """
    filters = []
    #disc_filters *= 2
    for i in range(layer_count):
        filters += [int(disc_filters*2)]
        disc_filters *= 2
    return filters
    """

class DCGAN(object):
    def __init__(self, data):
        tf.reset_default_graph()
        self.batch_size = 35
        #self.num_steps = num_steps
        self.vector_dim = 8
        self.data = data
        self.n_samp, self.n_input = self.data.shape
        print(self.data.shape, "shape")
        self.layer_count, self.starting_size = find_layers(self.n_input)
        print(self.layer_count, "layer count")
        #self.k_size= 50#max(5, min(n_input/50,30))
        self.k_size = max(self.starting_size, int(self.n_input/50))
        print(self.k_size, "ksize")

        self.gen_hidden_layers = [0]* self.layer_count
        self.disc_hidden_layers = [0]* self.layer_count


        self.gen_initial_filters = 128
        self.disc_max_filters = 256


        self.disc_filters = generate_filters_disc(self.layer_count, self.disc_max_filters)
        self.gen_filters = generate_filters_gen(self.layer_count, self.gen_initial_filters)
        #self.disc_filters = generate_filters_disc(self.layer_count, self.disc_initial_filters)
        #self.disc_filters = self.gen_filters[::-1] + [self.gen_initial_filters]
        print(self.disc_filters, self.gen_filters)

        #for i in range(len(self.gen_filters)): self.gen_filters[i] = self.gen_filters[i]*2
        #self.disc_initial_filters = self.gen_filters[-1]

    def generator(self,x, isTrain=True,reuse=False):
        with tf.variable_scope('Generator', reuse=reuse):
            gen_dense = tf.layers.dense(x, units= self.starting_size * self.gen_initial_filters ,kernel_initializer =tf.contrib.layers.xavier_initializer())#332
            gen_dense = tf.reshape(gen_dense, shape=[-1,self.starting_size, 1, self.gen_initial_filters]) #128
            #print(self.gen_initial_filters, "dense")
            gen_dense = tf.layers.batch_normalization(gen_dense,momentum=0.9, training =isTrain, epsilon=0.00001)
            gen_dense = tf.nn.relu(gen_dense)
            print(gen_dense.get_shape(), "gen_dense")
            for i in range(self.layer_count):
                if i == 0:

                    self.gen_hidden_layers[i] = tf.layers.conv2d_transpose(gen_dense, self.gen_filters[i], [self.k_size,1], strides=[2,1],kernel_initializer = tf.contrib.layers.xavier_initializer(), padding= "same")
                    self.gen_hidden_layers[i] = tf.nn.relu(tf.layers.batch_normalization(self.gen_hidden_layers[i], training =isTrain,momentum=0.9,epsilon=0.00001))
                    print(self.gen_hidden_layers[i].get_shape(), "gen",i)

                else:
                    self.gen_hidden_layers[i] = tf.layers.conv2d_transpose(self.gen_hidden_layers[i-1], self.gen_filters[i], [self.k_size,1], strides=[2,1],kernel_initializer = tf.contrib.layers.xavier_initializer(), padding= "same")
                    self.gen_hidden_layers[i] = tf.nn.relu(tf.layers.batch_normalization(self.gen_hidden_layers[i], training = isTrain,momentum=0.9,epsilon=0.00001))
                    print(self.gen_hidden_layers[i].get_shape(), "gen",i)


            gen_final_layer = tf.layers.conv2d_transpose(self.gen_hidden_layers[self.layer_count-1], 1, [self.k_size,1], strides=[2,1],kernel_initializer = tf.contrib.layers.xavier_initializer(), padding = "same")
            gen_final_layer = tf.nn.tanh(tf.squeeze(gen_final_layer, axis = 2))
            print(gen_final_layer.get_shape(), "gen_final")

            return gen_final_layer




    def discriminator(self,x,isTrain=True, reuse=False):
        with tf.variable_scope('Discriminator', reuse=reuse):
            #print(x.get_shape(), "X-shape, disc")
            disc_initial_layer = tf.layers.conv1d(x,self.disc_filters[0], self.k_size,strides=2, padding = "Same",kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
            disc_initial_layer = tf.nn.leaky_relu(tf.layers.batch_normalization(disc_initial_layer, momentum=0.9, training =isTrain, epsilon=0.00001))
            print(disc_initial_layer.get_shape(), "disc_initial")
            for i in range(self.layer_count):
                if i == 0:
                    #print(disc_filters, "Filters!!")
                    self.disc_hidden_layers[i] = tf.layers.conv1d(disc_initial_layer, self.disc_filters[i+1], self.k_size, strides=2,kernel_initializer = tf.contrib.layers.xavier_initializer(), padding= "same")
                    self.disc_hidden_layers[i] = tf.nn.leaky_relu(tf.layers.batch_normalization(self.disc_hidden_layers[i], training =isTrain,momentum=0.9,epsilon=0.00001))
                    print(self.disc_hidden_layers[i].get_shape(), "disc",i)
                else:
                    self.disc_hidden_layers[i] = tf.layers.conv1d(self.disc_hidden_layers[i-1], self.disc_filters[i+1], self.k_size, strides=2,kernel_initializer = tf.contrib.layers.xavier_initializer(), padding= "same")
                    self.disc_hidden_layers[i] = tf.nn.leaky_relu(tf.layers.batch_normalization(self.disc_hidden_layers[i], training = isTrain,momentum=0.9,epsilon=0.00001))
                    print(self.disc_hidden_layers[i].get_shape(), "disc",i)
            #print(disc_hidden_layers[layer_count-2].get_shape(), "Second to last layer, disc")
            #print(starting_size, "starting size")
            disc_final_layer = tf.layers.conv1d(self.disc_hidden_layers[self.layer_count-1], 1, self.starting_size,strides =1, padding = "VALID",kernel_initializer =tf.contrib.layers.xavier_initializer_conv2d())
            print(disc_final_layer.get_shape(), "final layer disc")
            out = tf.nn.sigmoid(disc_final_layer)
            return out, disc_final_layer

    def build_model(self):

        self.random_vector = tf.placeholder(tf.float32,shape=[None,self.vector_dim])
        self.real_image_input = tf.placeholder(tf.float32, shape=[None,self.n_input, 1])


        self.isTrain = tf.placeholder(dtype=tf.bool)

        self.gen_sample = self.generator(self.random_vector, self.isTrain)


        self.disc_real,self.disc_real_logits = self.discriminator(self.real_image_input,self.isTrain)
        self.disc_fake,self.disc_fake_logits = self.discriminator(self.gen_sample,self.isTrain, reuse=True)

        self.gan_model = self.discriminator(self.gen_sample,reuse=True)

        self.gen_target = tf.placeholder(tf.int32, shape=[None])
        self.disc_target = tf.placeholder(tf.int32, shape=[None])

        self.disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.disc_real_logits, labels=tf.ones_like(self.disc_real_logits)))
        self.disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.disc_fake_logits, labels=tf.zeros_like(self.disc_fake_logits)))
        self.disc_loss = self.disc_loss_real + self.disc_loss_fake
        self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.disc_fake_logits, labels=tf.ones_like(self.disc_fake_logits)))


        self.optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5)
        self.optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.002, beta1= 0.5)


        self.gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        self.disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

        self.update_ops_gen = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = "Generator")
        self.update_ops_disc = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Discriminator")

        with tf.control_dependencies(self.update_ops_gen):
             self.train_gen = self.optimizer_gen.minimize(self.gen_loss, var_list=self.gen_vars)

        with tf.control_dependencies(self.update_ops_disc):
            self.train_disc = self.optimizer_gen.minimize(self.disc_loss, var_list=self.disc_vars)

    def train(self,num_steps,continue_training = False):
        init = tf.global_variables_initializer()
        d_loss = np.zeros(shape=(1000))
        g_loss = np.zeros(shape=(1000))
        j = 0
        saver = tf.train.Saver()
        s = time.time()
        with tf.Session() as sess:
            sess.run(init)
            if continue_training:
                saver.restore(sess, "trained_network/trained.ckpt")
            for i in range(1, num_steps+1):
                if i%50==0:
                   #self.data = gen_data()
                   print("new_data")
                sample = np.random.randint(self.n_samp, size=self.batch_size)
                epoch_x = self.data[sample,:]
                epoch_x = np.reshape(epoch_x, newshape=[-1, self.n_input, 1])
                z = np.random.normal(0, 0.5, size=[self.batch_size, self.vector_dim])
               	dl,_,dlr,dlf = sess.run([self.disc_loss,self.train_disc,self.disc_loss_real, self.disc_loss_fake], feed_dict = {self.real_image_input:epoch_x, self.random_vector:z, self.isTrain:True})
                gl, _ = sess.run([self.gen_loss,self.train_gen], feed_dict = {self.random_vector:z,self.isTrain:True})
                

                if i % 100 == 0 or i == 1:
                    print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
                    print("DLR:",dlr,",", "DLF:", dlf)
                if int(i%(num_steps/1000)) == 0:
                    d_loss[j] = dl
                    g_loss[j] = gl
                    j+=1
            save_path = saver.save(sess, "trained_network/trained.ckpt")
            time_taken = time.time()-s
            sample = np.random.randint(self.n_samp, size=self.batch_size)
            epoch_x = self.data[sample,:]
            epoch_x = np.reshape(epoch_x, newshape=[-1, self.n_input,1])

            z = np.random.normal(0, 0.5, size=[100, self.vector_dim])
            samp  = sess.run(self.gen_sample, feed_dict={self.random_vector: z, self.isTrain:False})
            np.save("gen_output/generator_output.npy", samp)
            np.save("gen_output/g_loss.npy", g_loss)
            np.save("gen_output/d_loss.npy", d_loss)

    def sampler(self, size):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            tf.train.Saver().restore(sess, "trained_network/trained.ckpt")
            z = np.random.normal(0, 0.5, size=[size, self.vector_dim])
            samp  = sess.run(self.gen_sample, feed_dict={self.random_vector: z, self.isTrain:False})
            
            np.save("gen_output/sample.npy", samp)
            return samp






if __name__ == "__main__":
    x = DCGAN(data)
    x.build_model()
    #x.train(5000)
    print(x.sampler(3456))


# def unnormalize(data, orig):
#    data += 1
#    data = data/2
#    data = data*orig.max()
