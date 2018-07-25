from plot import plot
from GAN_master import DCGAN
from histogram import compare
import numpy as np
import scipy.io
from data_management import data_format, pca_func


def noise_array():
    return np.random.normal(scale = 0.05, size = (40000,144))




def gen_data():

    perfect_data = np.indices((40000,144))[1]
    sin = lambda t: np.sin((1/np.random.choice((25,60)) *t))
    data = np.apply_along_axis(sin, 1, perfect_data)

    return data + noise_array()


data = gen_data()

#data should be formatted so each row is one sample
#data = np.load("
#data = np.load("/home/jack/caltech_research/tissue_data_jack_GAN/tissue_data_jack/tissue_data_jack_normalized.npy")


data = scipy.io.mmread("/home/jack/caltech_research/tissue_data/droplet/Marrow-10X_P7_3/matrix.mtx")

data = data_format(data)
data, pca = pca_func(data, 28)

print(data.shape, "data shape")
training_steps = 100

def GAN_runner(data, retrain=False):
     GAN = DCGAN(data)
     GAN.build_model()
     GAN.train(training_steps, retrain)

def GAN_sampler(num_samples):
     net = DCGAN(data)
     net.build_model()
     return net.sampler(num_samples)

def histogram_runner(data):
    np.random.shuffle(data)
    if data.shape[0] % 2 !=0:
          #print(data.shape, "init")
          data = data[:-1]
          #print(data.shape, "jsjsjsj")
    data1, data2 = np.split(data,2)
    fake = GAN_sampler(data.shape[0]).squeeze(axis=2)
    np.random.shuffle(fake)    


    l = []
    for i in range(5):
        np.random.shuffle(fake)
        np.random.shuffle(data)
        box_list_fake = compare(data[:data1.shape[0]], fake[:data1.shape[0]])
        print(sum(box_list_fake)/ len(box_list_fake), "fake")
        l += [sum(box_list_fake)/ len(box_list_fake)]
    print(sum(l)/len(l), "Average Fake\n")

    l = []
    for i in range(5):
        np.random.shuffle(data)
        data1, data2 = np.split(data,2)
        box_list = compare(data2, data1)
        print(sum(box_list)/ len(box_list), "Real")
        l += [sum(box_list)/ len(box_list)]
    print(sum(l)/len(l), "Average Real\n")


def plot_runner(data):
     print(data.shape, "plot data")
     fake = GAN_sampler(data.shape[0]).squeeze(axis=2)
     print(fake.shape, "fake plot data")
     plot(fake,data)

#histogram_runner(data)
def main():
   GAN_runner(data)
   #histogram_runner(data)
   plot_runner(data)
   
if __name__ == "__main__":
    main()
