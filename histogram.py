
"""
Plots the input data column vs column and then divides the plot into 100 equal regions
and counts the occurances in each region, using a 2d histogram function. Computes
an accuracy by subtracting the histograms for real and fake data and dividing them by
the real data histogram, where all 0s are changed to ones.

Compares real data against itself for a control

The data should be formatted so each row is one sample

Program slows down at a rate of n^2 where n is the length of one sample. Large samples
need to be reduced

@Author Jack Stellwagen
"""
import numpy as np






#fake_well_trained = np.load("tissue_data_jack_gen_output/gen_output/generator_output.npy").squeeze(axis=2)

#print(fake_well_trained.shape,  "shape")


#fake_1epoch = np.load("tissue_data_jack_gen_output/gen_output/tissue_data_jack_1epoch.npy").squeeze(axis=2)
#fake_10epochs = np.load("tissue_data_jack_gen_output/gen_output/tissue_data_jack_10epoch.npy").squeeze(axis=2)
#fake_100epochs = np.load("tissue_data_jack_gen_output/gen_output/tissue_data_jack_100epoch.npy").squeeze(axis=2)
#fake_1000epochs = np.load("tissue_data_jack_gen_output/gen_output/tissue_data_jack_1000epoch.npy").squeeze(axis=2)





#real = np.load("tissue_data_jack_normalized.npy")

#real = real.transpose()

#data_len = real.shape[1]
#print(data_len, "len")
#print(real.shape, "shape")



def compare(real, fake):
    box_list = []
    data_len = real.shape[1]
    #print(real.shape, "real")
    #print(fake.shape, "fake")
    if data_len<35:
        for i in range(data_len):
            for j in range(data_len):
                real_boxes,_,_ = np.histogram2d(x = real[:,i], y = real[:,j],bins = 10)
                fake_boxes,_,_ = np.histogram2d(x = fake[:,i], y = fake[:,j], bins = 10)
                maxed_real = np.maximum(real_boxes,1)
                box_list += [np.sum(np.absolute((real_boxes)- (fake_boxes))/maxed_real)]
        return box_list
   
    else:
        for i in range(35):
            for j in range(35):
                x = np.random.randint(0, data_len)
                y = np.random.randint(0, data_len)
                real_boxes,_,_ = np.histogram2d(x = real[:,x], y = real[:,y],bins = 10)
                fake_boxes,_,_ = np.histogram2d(x = fake[:,x], y = fake[:,y], bins = 10)
                maxed_real = np.maximum(real_boxes,1)
                box_list += [np.sum(np.absolute((real_boxes)- (fake_boxes))/maxed_real)]
        return box_list

def compare_real(real):
    #if len(real)%2 !=0:
    #    real = real[0:-1]
    np.random.shuffle(real)
    #arr_ind = range(len(real[0]))
    #print(len(real[0]),"sksks")
    #arr_ind = np.random.permutation(arr_ind)
    #real = real[:,arr_ind]

    real, fake = np.split(real, 2)
    return compare(real,fake)

def normal_compare(real, fake):
    #arr_ind = range(len(real))
    np.random.shuffle(real)
    #if len(real)%2 !=0:
    #    real = real[0:-1]
    real, _ = np.split(real, 2)


    #print(fake.shape, "shshsh")
    np.random.shuffle(fake)
    #print(real.shape[0], "real.shape[0]")
    fake = fake[:real.shape[0]]

    return compare(real,fake)


"""

abs(real - fake) /  max(1, real)


"""

if __name__ == '__main__':
    l = []


    for i in range(5):
        box_list_well_trained = normal_compare(real, fake_well_trained)
        print(sum(box_list_well_trained)/ len(box_list_well_trained), "well_trained")
        l += [sum(box_list_well_trained)/ len(box_list_well_trained)]
    print(sum(l)/len(l), "Average Fake\n")


    l=[]
    for i in range(1):
        box_list_real = compare_real(real)
        print(sum(box_list_real)/ len(box_list_real), "real")
        l += [sum(box_list_real)/ len(box_list_real)]
    print(sum(l)/len(l), "Average Real")
    #for i in range(10):


    box_list_1epoch = normal_compare(real, fake_1epoch)
    print(sum(box_list_1epoch)/ len(box_list_1epoch), "trained 1 epochs")

    box_list_10epoch = normal_compare(real, fake_10epochs)
    print(sum(box_list_10epoch)/ len(box_list_10epoch), "trained 10 epochs")

    box_list_100epoch = normal_compare(real, fake_100epochs)
    print(sum(box_list_100epoch)/ len(box_list_100epoch), "trained 100 epochs")

    box_list_1000epoch = normal_compare(real, fake_1000epochs)
    print(sum(box_list_1000epoch)/ len(box_list_1000epoch), "trained 1000 epochs")



    l= []
    for i in range(2):
        random = np.random.uniform(-1,1,size=real.shape)
        box_list_random = normal_compare(real, random)
        l += [sum(box_list_random)/ len(box_list_random)]
        print(sum(box_list_random)/ len(box_list_random), "random")
    print(sum(l)/len(l), "Average Random")

    """
    real_dif = []
    for i in range(150):
        np.random.shuffle(real)
        box_list = compare_real(real)
        real_dif += [sum(box_list)/ len(box_list)]
        print(i)
    print(real_dif,sum(real_dif)/len(real_dif) )
    """
