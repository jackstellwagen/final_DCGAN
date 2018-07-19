import matplotlib.pyplot as plt
import numpy as np
"""


np.histogram2d(x = x[:,0].squeeze(), y = x[:,1].squeeze())

"""


#gen_data = np.load("tissue_data_jack_gen_output/gen_output/generator_output.npy").squeeze(axis=2)




#t_data = np.load("tissue_data_jack_normalized.npy")




def plot(gen_data, t_data):
    print(t_data.shape)
    final_shape = min(t_data.shape[0], gen_data.shape[0])
    t_data = t_data[:final_shape]
    gen_data = gen_data[:final_shape]



    size = 1
    c_real = "blue"
    c_gen = "orange"



    arr_idx = range((len(t_data)-1)+ (len(gen_data)-1))
    arr_idx = np.random.permutation(arr_idx)

    c = [c_real] * (len(t_data))
    c += [c_gen] * (len(gen_data) )

    c = np.array(c)


    proj = np.concatenate((t_data, gen_data), axis =0)


    proj = proj[arr_idx, :]
    c = c[arr_idx]

    fig, ax = plt.subplots(nrows=4, ncols=4)

    i = 0



    for row in ax:
        j = 0
        for col in row:
            #col.scatter(t_data[:,i], t_data[:,j], s =1)#, color = "black")
            #col.scatter(gen_data[:,i], gen_data[:,j], s =1, alpha = 0.6)
            col.scatter(proj[:,i], proj[:,j],s=size, c=c,alpha=1)
            j += 1
        i += 1
    plt.show()
if __name__ == "__main__":
    plot(gen_data,t_data)


