import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize as scop


def get_maxima_minima(xs, ys):

    ys_avg = moving_average(ys, n=30)

    maxima = []
    minima = []
    last = ys_avg[0]
    status = "up"
    for y_idx, y in enumerate(ys_avg):
        if status=="up":
            if y<last:
                maxima.append([xs[y_idx-1],last])
                status="down"
        elif status=="down":
            if y>last:
                minima.append([xs[y_idx-1],last])
                status="up"
        last=y
    maxima = np.array(maxima)
    minima = np.array(minima)
    #return(maxima, minima)
    left_min_possible = [m for m in minima if m[0]>349 and m[0]<400]
    if len(left_min_possible)!=0:
        left_min_min_idx = np.argmin([m[1] for m in left_min_possible])
        left_min = left_min_possible[left_min_min_idx]
    else:
        left_min = [xs[0], ys_avg[0]]

    max_1_possible = [m for m in maxima if m[0]>370 and m[0]<470]
    max_1_max_idx = np.argmax([m[1] for m in max_1_possible])
    max_1 = max_1_possible[max_1_max_idx]
    max_2_possible = [m for m in maxima if m[0]>500 and m[0]<620]
    max_2_max_idx = np.argmax([m[1] for m in max_2_possible])
    max_2 = max_2_possible[max_2_max_idx]
    min_1_possible = [m for m in minima if m[0]>420 and m[0]<530]
    min_1_min_idx = np.argmin([m[1] for m in min_1_possible])
    min_1 = min_1_possible[min_1_min_idx]

    right_min_possible = [m for m in minima if m[0]>600 and m[0]<651]
    if len(right_min_possible)!=0:
        right_min_min_idx = np.argmin([m[1] for m in right_min_possible])
        right_min = right_min_possible[right_min_min_idx]
    else:
        right_min = [xs[-1], ys_avg[-1]]

    return(left_min, max_1, min_1, max_2, right_min)


def fit_maxima(name, xs, ys, maxima_minima, num=2):
    max_1 = maxima_minima[1]
    max_2 = maxima_minima[3]
    if num == 2:

        a0, m0, sigma0 = max_1[1], max_1[0], 20
        a1, m1, sigma1 = max_2[1], max_2[0], 30
        x_fit = np.linspace(350, 650, 2000)
        y_fit_init = fitfunction_maxima2(x_fit, a0, m0, sigma0, a1, m1, sigma1)
        fit_init = [a0, m0, sigma0, a1, m1, sigma1]
        try:
            fit_opt, fit_cov = scop.curve_fit(fitfunction_maxima2, xs, ys, p0=[a0, m0, sigma0, a1, m1, sigma1])
        except:
            print("Fit of %i gaussians to %s did not converge"%(num, name))
            fit_opt = [0.0, m0, sigma0, 0.0, m1, sigma1]
            fit_cov = None
        y_fit = fitfunction_maxima2(x_fit, fit_opt[0], fit_opt[1], fit_opt[2], fit_opt[3], fit_opt[4], fit_opt[5])
        #return(x, y_init, y_init, [])

    if num == 3:

        a0, m0, sigma0 = max_1[1], max_1[0], 20
        a1, m1, sigma1 = 0.98*max_2[1], max_2[0]+5, 30
        a2, m2, sigma2 = 0.2*max_2[1], max_2[0]-40, 20
        x_fit = np.linspace(350, 650, 2000)
        y_fit_init = fitfunction_maxima3(x_fit, a0, m0, sigma0, a1, m1, sigma1, a2, m2, sigma2)
        fit_init = [a0, m0, sigma0, a1, m1, sigma1, a2, m2, sigma2]
        try:
            fit_opt, fit_cov = scop.curve_fit(fitfunction_maxima3, xs, ys, p0=[a0, m0, sigma0, a1, m1, sigma1, a2, m2, sigma2])
        except:
            print("Fit of %i gaussians to %s did not converge"%(num, name))
            fit_opt = [0.0, m0, sigma0, 0.0, m1, sigma1, 0.0, m2, sigma2]
            fit_cov = None
        y_fit = fitfunction_maxima3(x_fit, fit_opt[0], fit_opt[1], fit_opt[2], fit_opt[3], fit_opt[4], fit_opt[5], fit_opt[6], fit_opt[7], fit_opt[8])
        #return(x_fit, y_init, y_init, [])

    return(x_fit, y_fit, y_fit_init, fit_init, fit_opt)

def fitfunction_maxima4(x, a0, m0, sigma0, a1, m1, sigma1, a2, m2, sigma2, a3, m3, sigma3):
    y = a0*np.exp(-0.5*((x-m0)/(sigma0))**2.0) + a1*np.exp(-0.5*((x-m1)/(sigma1))**2.0) + a2*np.exp(-0.5*((x-m2)/(sigma2))**2.0) + a3*np.exp(-0.5*((x-m3)/(sigma3))**2.0)
    return(y)

def fitfunction_maxima3(x, a0, m0, sigma0, a1, m1, sigma1, a2, m2, sigma2):
    y = a0*np.exp(-0.5*((x-m0)/(sigma0))**2.0) + a1*np.exp(-0.5*((x-m1)/(sigma1))**2.0) + a2*np.exp(-0.5*((x-m2)/(sigma2))**2.0)
    return(y)

def fitfunction_maxima2(x, a0, m0, sigma0, a1, m1, sigma1):
    y = a0*np.exp(-0.5*((x-m0)/(sigma0))**2.0) + a1*np.exp(-0.5*((x-m1)/(sigma1))**2.0)
    return(y)

def fitfunction_maxima1(x, a0, m0, sigma0):
    y = a0*np.exp(-0.5*((x-m0)/(sigma0))**2.0)
    return(y)

def get_gaussians(xs, fit_opt):
    gaussians=[]
    for i in range(len(fit_opt)//3):
        a0, m0, sigma0 = fit_opt[3*i], fit_opt[3*i+1], fit_opt[3*i+2]
        gaussians.append(fitfunction_maxima1(xs, a0, m0, sigma0))
    return(gaussians)

def moving_average(a, n=3):
    m = len(a)
    avg = []
    for idx in range(m):
        if idx<n//2:
            avg.append(np.mean(a[0:idx+n//2]))
        elif idx > m-n//2:
            avg.append(np.mean(a[idx:]))
        else:
            avg.append(np.mean(a[idx-n//2:idx+n//2]))
    avg = np.array(avg)
    return avg



def reduce(data_x, data_y, num=10):
    n_new = len(data_y[0])//num
    data_x_new = np.zeros((len(data_y), n_new))
    data_y_new = np.zeros((len(data_y), n_new))
    for i1 in range(len(data_y)):
        for i2 in range(n_new):
            data_x_new[i1][i2]=np.mean(data_x[i1][i2*num:(i2+1)*num])
            data_y_new[i1][i2]=np.max(data_y[i1][i2*num:(i2+1)*num])
        if (i1+1)%1000==0 or (i1+1)==len(data_y):
            print("reduced %i of %i XRD samples"%(i1+1, len(data_y)))
    return(data_x_new, data_y_new)




def data_augmentation(features, labels, factor=2.0):
    num_target = int(round(factor*len(features)))
    num_todo = len(features)-num_target


def bg(l):
    a, b, c = np.random.random(3)-0.5
    xs = np.linspace(-1.0, 1.0, l)
    ys = a*xs**3.0 + b*xs**2.0 + c*xs
    ys-=np.min(ys)
    ys/=np.max(ys)
    fluct_noise_level_max = 100
    ys*=(np.random.random()*0.8+0.2)*fluct_noise_level_max
    base_noise_level_max = 100.0
    ys+=(np.random.random()*0.8+0.2)*base_noise_level_max
    return(ys)


def disturb_spectrum(to_add, l):

    # shift randomly in x
    s = np.random.randint(140)-70
    to_add_new = []
    if s>=0: # shift right -> fill beginning with 0s
        for j in range(s):
            to_add_new.append(to_add[0])
        for j in range(l-s):
            to_add_new.append(to_add[j])
    if s<0: # shift left
        for entry in to_add[-s:]:
            to_add_new.append(entry)
        for j in range(-s):
            to_add_new.append(to_add[-1])
    to_add = np.array(to_add_new)

    # scale in x
    factor_scale_x = np.random.random()*0.05+0.975
    to_add_new = []
    for j in range(l):
        j2 = int(round(factor_scale_x*j))
        if j2>=l:
            break
        to_add_new.append(to_add[j2])
    for j in range(l-len(to_add_new)):
        to_add_new.append(to_add[-1])
    to_add=np.array(to_add_new)



    # scale in y
    factor_scale_y = np.random.random()*0.8+0.3
    to_add *= factor_scale_y

    # make wider
    w = np.random.randint(10)+1
    to_add = np.array([np.max(to_add[max(0,i):min(l,i+w)]) for i in range(l)])

    return(to_add)

def generate_fake_data(data_x, data_y, num=100):
    print("generate %i fake spectra"%(num))
    data_x_fake = []
    data_y_fake = []
    labels_fake = []
    l = len(data_x[0])
    probability_pure = 0.5
    do_plot=True
    if do_plot:
        plt.figure(figsize=(30,20))
        num_to_plot = 200
        num_plotted = 0
        #fig, axes = plt.subplots(4,4)
        #i1 = 0
        #j1 = 0
    #num_reference_spectra = len(data_x)
    num_reference_spectra = 2
    for i in range(num):
        x = data_x[0]
        # generate some background
        y = bg(l)
        # add gaussian noise
        max_noise = 20
        strength = max_noise*(np.random.random()*0.9+0.1)
        y += np.random.normal(loc=0.0, scale = 1.0, size=l)*strength
        # select a random spectrum
        idx = np.random.randint(num_reference_spectra)
        to_add = data_y[idx]
        to_add-=np.min(to_add)
        # disturb it
        to_add = disturb_spectrum(to_add, l)
        # finally add it
        y += to_add
        # make a non pure one
        additional_peaks = np.zeros((l))
        if np.random.random()<probability_pure:
            label = "pure"
        else:
            label = "mixed"
            num_gaussians = np.random.randint(10)+2
            a0_overall = np.random.random()*0.9+0.1
            for j in range(num_gaussians):
                a0 = np.abs(np.random.normal())*330.0+43.0
                m0 = np.random.random()*40.0+10.0
                sigma0 = np.random.random()*0.35+0.15
                additional_peak = a0_overall*a0*np.exp(-0.5*((x-m0)/(sigma0))**2.0)
                #additional_peak = a0*np.exp(-0.5*((x-m0)/(sigma0))**2.0)
                additional_peaks += additional_peak
            y+=additional_peaks
            # sometimes add small fractions of other spectra
            if np.random.random()<0.2:
                if num_reference_spectra==1:
                    other_idx = 1
                else:
                    other_idx = idx
                    while other_idx==idx:
                        other_idx = np.random.randint(num_reference_spectra)
                to_add = data_y[other_idx]
                to_add-=np.min(to_add)
                to_add = disturb_spectrum(to_add, l)
                to_add *= (np.random.random()*0.2+0.05)
                y += to_add
        if do_plot:
            if num_plotted < num_to_plot:
                if label=="mixed":
                    pass
                    plt.plot(x, y, "r-")
                else:
                    plt.plot(x, y, "k-")
                num_plotted+=1
            #else:
            #    break
        #plt.figure()
        #plt.plot(x, y, "k-", label="final fake spectrum")
        #plt.plot(x, data_y[idx], "r-", label="underlying reference")
        #plt.plot(x, additional_peaks, "b-", label="fake peaks")
        #axes[i1][j1].plot(x, data_y[idx], "k-")
        #axes[i1][j1].plot(x, y, "r-")
        #axes[i1][j1].get_xaxis().set_ticks([])
        #axes[i1][j1].get_yaxis().set_ticks([])
        #i1+=1
        #if i1==4:
        #    i1=0
        #    j1+=1

        labels_fake.append(label)
        data_x_fake.append(x.tolist())
        data_y_fake.append(y.tolist())
        if (i+1)%100==0 or (i+1)==num:
            print("%i of %i fake spectra done"%(i+1, num))
        #if i==15:
        #    break

    if do_plot:
        plt.xlabel("Angle [degree]")
        plt.ylabel("XRD intensity [a.u.]")
        plt.xlim([10.0, 50.0])
        plt.ylim([0.0, 4500.0])
        #plt.legend(loc="upper right")
        plt.savefig("fake_spectra.png", dpi=300)
        plt.close()

    data_x_fake = np.array(data_x_fake)
    data_y_fake = np.array(data_y_fake)
    #axes[3][2].set_xlabel("Angle [degree]")
    #axes[2][0].set_ylabel("XRD intensity [a.u.]")
    #plt.subplots_adjust(hspace=0, wspace=0)
    #plt.savefig("fake_spectra.png", dpi=120)
    #plt.close()
    #exit()
    return(data_x_fake, data_y_fake, labels_fake)




def r2_metric(y_true, y_pred):
    """
    Compute r2 metric.

    Args:
        y_true (tf.tensor): True y-values.
        y_pred (tf.tensor): Predicted y-values.

    Returns:
        tf.tensor: r2 metric.

    """
    SS_res =  ks.backend.sum(ks.backend.square(y_true - y_pred)) 
    SS_tot = ks.backend.sum(ks.backend.square(y_true-ks.backend.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + ks.backend.epsilon()) )



def lr_lin_reduction(learning_rate_start = 1e-3, learning_rate_stop = 1e-5, epo = 10000, epomin= 1000):
    """
    Make learning rate schedule function for linear reduction.

    Args:
        learning_rate_start (float, optional): Learning rate to start with. The default is 1e-3.
        learning_rate_stop (float, optional): Final learning rate at the end of epo. The default is 1e-5.
        epo (int, optional): Total number of epochs to reduce learning rate towards. The default is 10000.
        epomin (int, optional): Minimum number of epochs at beginning to leave learning rate constant. The default is 1000.

    Returns:
        func: Function to use with LearningRateScheduler.
    
    Example:
        lr_schedule_lin = tf.keras.callbacks.LearningRateScheduler(lr_lin_reduction)
    """
    def lr_out_lin(epoch):
        if(epoch < epomin):
            out = learning_rate_start
        else:
            out = float(learning_rate_start - (learning_rate_start-learning_rate_stop)/(epo-epomin)*(epoch-epomin))
        return out
    return lr_out_lin



def lr_log_reduction(learning_rate_start = 1e-3, learning_rate_stop = 1e-5, epo = 10000, epomin= 1000):
    """
    Make learning rate schedule function for linear reduction.

    Args:
        learning_rate_start (float, optional): Learning rate to start with. The default is 1e-3.
        learning_rate_stop (float, optional): Final learning rate at the end of epo. The default is 1e-5.
        epo (int, optional): Total number of epochs to reduce learning rate towards. The default is 10000.
        epomin (int, optional): Minimum number of epochs at beginning to leave learning rate constant. The default is 1000.

    Returns:
        func: Function to use with LearningRateScheduler.
    
    Example:
        lr_schedule_lin = tf.keras.callbacks.LearningRateScheduler(lr_lin_reduction)
    """
    def lr_out_log(epoch):
        if(epoch < epomin):
            out = learning_rate_start
        else:
            out = np.exp(float(np.log(learning_rate_start) - (np.log(learning_rate_start)-np.log(learning_rate_stop))/(epo-epomin)*(epoch-epomin)))
        return out
    return lr_out_log


