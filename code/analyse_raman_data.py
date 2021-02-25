import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import utils as u
import time

np.random.seed(43)

if not os.path.exists("raman"):
    os.makedirs("raman")

# RAMAN DATA
time0=time.time()
labels = []
names_labels = []
for lineidx,line in enumerate(open("../data/Ramanratio.csv","r")):
    if lineidx>0:
        labels.append(float(line.replace("\n","").split(",")[1]))
        names_labels.append(line.replace("\n","").split(",")[0])


data_x = []
data_y = []
names_unique = []
names_all = []
for lineidx,line in enumerate(open("../data/Ramandata.csv","r")):
    if lineidx==0:
        for x in line.replace("\n","").split(","):
            x = x.replace("Sample ","").replace("Sample","").replace("sample ","").replace("sample","")
            if x!="":
                names_all.append(x)
                if "-" in x:
                    name_unique = x.split("-")[0]
                else:
                    name_unique=x
                if name_unique not in names_unique:
                    names_unique.append(x)
    elif lineidx>0:
        data_x.append([])
        data_y.append([])
        data_here=[]
        for x in line.replace("\n","").split(","):
            data_here.append(float(x))
        if len(data_here)!=2*len(names_all):
            print("WARNING: line %i does not contain %i entries for %i samples, only found %i entries."%(len(names_all), len(names_all), len(data_here)/2))
        for idx in range(len(names_all)):
            data_x[-1].append(data_here[2*idx])
            data_y[-1].append(data_here[2*idx+1])



data_x = np.array(data_x).T
data_y = np.array(data_y).T


data_x = np.flip(data_x, axis=1)
data_y = np.flip(data_y, axis=1)

# select the range from 350 to 650 nm
min_idx = np.argmin(np.abs(data_x[0]-350.0))
max_idx = np.argmin(np.abs(data_x[0]-650.0))

data_x = data_x[:,min_idx:max_idx]
data_y = data_y[:,min_idx:max_idx]

print(data_x.shape, data_y.shape)

data_x_avaraged=[]
data_y_avaraged=[]
indeces_to_average_all = []
for name in names_unique:
    indeces_to_average=[]
    for idx, fullname in enumerate(names_all):
        if "-" in fullname:
            fullname = fullname.split("-")[0]
        if fullname==name:
            # some more selection criteria
            select=True
            if np.min(data_y[idx])>100:
                select=False
                print("high background: %s"%(name))
            if np.max(data_y[idx])<100:
                select=False
                print("flat line: %s"%(name))
            if select:
                indeces_to_average.append(idx)
    indeces_to_average_all.append(indeces_to_average)
    indeces_to_average=np.array(indeces_to_average)
    x_averaged = np.mean(data_x[indeces_to_average], axis=0)
    y_averaged = np.mean(data_y[indeces_to_average], axis=0)

    data_x_avaraged.append(x_averaged.tolist())
    data_y_avaraged.append(y_averaged.tolist())

    y_std = np.std(data_y[indeces_to_average], axis=0)

data_x_avaraged = np.array(data_x_avaraged)
data_y_avaraged = np.array(data_y_avaraged)

num_samples = len(names_unique)

print("Number of Raman spectra: %i"%(num_samples))

xmin, xmax = 350.0, 650.0
ymin, ymax = -200.0, 4000.0
plt.figure()
for idx in range(min(10,num_samples)):
    plt.plot(data_x_avaraged[idx], data_y_avaraged[idx], "-", color="C%i"%(idx), label="sample %i"%(idx))
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.savefig("raman/raman_first_10.png", dpi=120)
plt.close()




sample_number = 18
plt.figure()
idx = [i for i in range(len(names_unique)) if int(names_unique[i])==sample_number][0]
plt.plot(data_x_avaraged[idx], data_y_avaraged[idx], "-", color="C0", label="sample %i"%(sample_number))
for counter,idx2 in enumerate(indeces_to_average_all[idx]):
    plt.plot(data_x[idx2], data_y[idx2], "-", color="C1", alpha=0.7, linewidth=1)
    #print("compute maxima and minima of sample %s, subsample %s"%(names_unique[idx], idx2))
    #maxima, minima = u.get_maxima_minima(data_x[idx2], data_y[idx2])
    ##for m in maxima:
    ##    plt.scatter([m[0]], [m[1]], marker="^", facecolor="r", edgecolor="k")
    #for m in minima:
    #    plt.scatter([m[0]], [m[1]], marker="v", facecolor="b", edgecolor="k")
     
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.savefig("raman/raman_sample_%i.png"%(sample_number), dpi=120)
plt.close()





# compute maxima and minima
maxima_minima = []
maxima_minima_all = []
for idx in range(num_samples):
    print("compute maxima and minima of sample %s"%(names_unique[idx]))
    left_min, max_1, min_1, max_2, right_min = u.get_maxima_minima(data_x_avaraged[idx], data_y_avaraged[idx])
    maxima_minima.append([left_min, max_1, min_1, max_2, right_min])
    maxima_minima_here = []
    for idx2 in indeces_to_average_all[idx]:
        print("compute maxima and minima of sample %s, subsample %s"%(names_unique[idx], idx2))
        left_min, max_1, min_1, max_2, right_min = u.get_maxima_minima(data_x[idx2], data_y[idx2])
        maxima_minima_here.append([left_min, max_1, min_1, max_2, right_min])
    maxima_minima_all.append(maxima_minima_here)


# do the fitting of gaussians

fits2 = []
fits2_all = []
fits3 = []
fits3_all = []
for idx in range(num_samples):
    print("compute gaussian fit of sample %s"%(names_unique[idx]))
    name = "sample %s"%(names_unique[idx])
    x_fit, y_fit, y_fit_init, fit_init, fit_opt = u.fit_maxima(name, data_x_avaraged[idx], data_y_avaraged[idx], maxima_minima[idx], num=2)
    fits2.append([x_fit, y_fit, y_fit_init, fit_init, fit_opt])
    x_fit, y_fit, y_fit_init, fit_init, fit_opt = u.fit_maxima(name, data_x_avaraged[idx], data_y_avaraged[idx], maxima_minima[idx], num=3)
    fits3.append([x_fit, y_fit, y_fit_init, fit_init, fit_opt])
    fits2_here = []
    fits3_here = []
    for counter,idx2 in enumerate(indeces_to_average_all[idx]):
        print("compute gaussian fit of sample %s, subsample %s"%(names_unique[idx], idx2))
        name = "sample %s, subsample %s"%(names_unique[idx], idx2)
        x_fit, y_fit, y_fit_init, fit_init, fit_opt = u.fit_maxima(name, data_x[idx2], data_y[idx2], maxima_minima_all[idx][counter], num=2)
        fits2_here.append([x_fit, y_fit, y_fit_init, fit_init, fit_opt])
        x_fit, y_fit, y_fit_init, fit_init, fit_opt = u.fit_maxima(name, data_x[idx2], data_y[idx2], maxima_minima_all[idx][counter], num=3)
        fits3_here.append([x_fit, y_fit, y_fit_init, fit_init, fit_opt])
    fits2_all.append(fits2_here)
    fits3_all.append(fits3_here)



# compute the areas and ratios etc.


ratios2 = []
ratios2_all = []
ratios3 = []
ratios3_all = []

ratios2_n = []
ratios2_n_all = []
ratios3_n = []
ratios3_n_all = []

ratios_nofit = []
ratios_nofit_all = []

for sample_number in names_unique:
    sample_number = int(sample_number)
    for num_gaussians in [2,3]:
        print("compute integrals and ratios for sample %i (%i gaussians)"%(sample_number, num_gaussians))
        idx = [i for i in range(len(names_unique)) if int(names_unique[i])==sample_number][0]
        # 0 is x, 1 is y, 2 is y_init, 3 is fit params init, 4 is fit params opt
        if num_gaussians==2:
            gaussians = u.get_gaussians(fits2[idx][0], fits2[idx][4])
            integrals = []
            integrals_n = []
            means=[]
            for gidx, g in enumerate(gaussians):
                a0, m0, sigma0 = fits2[idx][4][gidx*3], fits2[idx][4][gidx*3+1], fits2[idx][4][gidx*3+2]
                means.append(m0)
                integral1 = a0*(np.pi/(0.5/sigma0**2.0))**0.5
                dx = fits2[idx][0][1]-fits2[idx][0][0]
                integral2 = np.sum(g)*dx
                integrals.append(integral1)
                integrals_n.append(integral2)
                if abs(integral1-integral2)/integral1>0.05:
                    print("WARNING: integral difference detected: %.3f %.3f"%(integral1, integral2))
            order=np.argsort(means)
            ratio2 = integrals[order[1]]/integrals[order[0]]
            ratio2_n = integrals_n[order[1]]/integrals_n[order[0]]
            ratios2.append(ratio2)
            ratios2_n.append(ratio2_n)

        elif num_gaussians==3:
            gaussians = u.get_gaussians(fits3[idx][0], fits3[idx][4])
            integrals = []
            integrals_n = []
            means=[]
            for gidx, g in enumerate(gaussians):
                a0, m0, sigma0 = fits3[idx][4][gidx*3], fits3[idx][4][gidx*3+1], fits3[idx][4][gidx*3+2]
                means.append(m0)
                integral1 = a0*(np.pi/(0.5/sigma0**2.0))**0.5
                dx = fits3[idx][0][1]-fits3[idx][0][0]
                integral2 = np.sum(g)*dx
                integrals.append(integral1)
                integrals_n.append(integral2)
                if abs(integral1-integral2)/integral1>0.05:
                    print("WARNING: integral difference detected: %.3f %.3f"%(integral1, integral2))
            order=np.argsort(means)
            ratio3 = (integrals[order[1]]+integrals[order[2]])/integrals[order[0]]
            ratio3_n = (integrals_n[order[1]]+integrals_n[order[2]])/integrals_n[order[0]]
            ratios3.append(ratio3)
            ratios3_n.append(ratio3_n)

        ratios2_here = []
        ratios2_n_here = []
        ratios3_here = []
        ratios3_n_here = []
        for counter,idx2 in enumerate(indeces_to_average_all[idx]):
            print("compute integrals and ratios for sample %i, subsample %i (%i gaussians)"%(sample_number, counter, num_gaussians))
            if num_gaussians==2:
                gaussians = u.get_gaussians(fits2_all[idx][counter][0], fits2_all[idx][counter][4])
                integrals = []
                integrals_n = []
                means=[]
                for gidx, g in enumerate(gaussians):
                    a0, m0, sigma0 = fits2_all[idx][counter][4][gidx*3], fits2_all[idx][counter][4][gidx*3+1], fits2_all[idx][counter][4][gidx*3+2]
                    means.append(m0)
                    integral1 = a0*(np.pi/(0.5/sigma0**2.0))**0.5
                    dx = fits2_all[idx][counter][0][1]-fits2_all[idx][counter][0][0]
                    integral2 = np.sum(g)*dx
                    integrals.append(integral1)
                    integrals_n.append(integral2)
                    if abs(integral1-integral2)/integral1>0.05:
                        print("WARNING: integral difference detected: %.3f %.3f"%(integral1, integral2))
                order=np.argsort(means)
                ratio2 = integrals[order[1]]/integrals[order[0]]
                ratio2_n = integrals_n[order[1]]/integrals_n[order[0]]
                ratios2_here.append(ratio2)
                ratios2_n_here.append(ratio2_n)

            elif num_gaussians==3:
                gaussians = u.get_gaussians(fits3_all[idx][counter][0], fits3_all[idx][counter][4])
                integrals = []
                integrals_n = []
                means=[]
                for gidx, g in enumerate(gaussians):
                    a0, m0, sigma0 = fits3_all[idx][counter][4][gidx*3], fits3_all[idx][counter][4][gidx*3+1], fits3_all[idx][counter][4][gidx*3+2]
                    means.append(m0)
                    integral1 = a0*(np.pi/(0.5/sigma0**2.0))**0.5
                    dx = fits3_all[idx][counter][0][1]-fits3_all[idx][counter][0][0]
                    integral2 = np.sum(g)*dx
                    integrals.append(integral1)
                    integrals_n.append(integral2)
                    if abs(integral1-integral2)/integral1>0.05:
                        print("WARNING: integral difference detected: %.3f %.3f"%(integral1, integral2))
                order=np.argsort(means)
                ratio3 = (integrals[order[1]]+integrals[order[2]])/integrals[order[0]]
                ratio3_n = (integrals_n[order[1]]+integrals_n[order[2]])/integrals_n[order[0]]
                ratios3_here.append(ratio3)
                ratios3_n_here.append(ratio3_n)


        if num_gaussians==2:
            ratios2_all.append(ratios2_here)
            ratios2_n_all.append(ratios2_n_here)
        elif num_gaussians==3:
            ratios3_all.append(ratios3_here)
            ratios3_n_all.append(ratios3_n_here)

    # don't use fits, just use the raw data
    idx = [i for i in range(len(names_unique)) if int(names_unique[i])==sample_number][0]
    x = data_x_avaraged[idx]
    y = data_y_avaraged[idx]
    x1, x2, x3 = maxima_minima[idx][0][0], maxima_minima[idx][2][0], maxima_minima[idx][4][0]
    i1, i2, i3 = [np.argmin(np.abs(x-x_i)) for x_i in [x1, x2, x3]]
    print(y.shape, i1, i2, i3)
    ratio_nofit = np.sum(y[i2:i3])/np.sum(y[i1:i2])
    ratios_nofit.append(ratio_nofit)

    ratios_nofit_here = []
    for counter,idx2 in enumerate(indeces_to_average_all[idx]):
        x = data_x[idx2]
        y = data_y[idx2]
        x1, x2, x3 = maxima_minima_all[idx][counter][0][0], maxima_minima_all[idx][counter][2][0], maxima_minima_all[idx][counter][4][0]
        i1, i2, i3 = [np.argmin(np.abs(x-x_i)) for x_i in [x1, x2, x3]]
        print(y.shape, i1, i2, i3)
        ratio_nofit_here = np.sum(y[i2:i3])/np.sum(y[i1:i2])
        ratios_nofit_here.append(ratio_nofit_here)
    ratios_nofit_all.append(ratios_nofit_here)



ratios2_avg = []
ratios3_avg = []
ratios2_n_avg = []
ratios3_n_avg = []
ratios_nofit_avg = []
for idx in range(num_samples):
    ratios2_avg.append(np.mean(ratios2_all[idx]))
    ratios3_avg.append(np.mean(ratios3_all[idx]))
    ratios2_n_avg.append(np.mean(ratios2_n_all[idx]))
    ratios3_n_avg.append(np.mean(ratios3_n_all[idx]))
    ratios_nofit_avg.append(np.mean(ratios_nofit_all[idx]))
    print("Sample %s"%(names_unique[idx]))
    print("2 gaussians:           %.3f %.3f       3 gaussians: %.3f %.3f"%(ratios2[idx], ratios2_avg[idx], ratios3[idx], ratios3_avg[idx]))
    print("2 gaussians (num):     %.3f %.3f       3 gaussians: %.3f %.3f"%(ratios2_n[idx], ratios2_n_avg[idx], ratios3_n[idx], ratios3_n_avg[idx]))
    print("numerical calculation: %.3f %.3f"%(ratios_nofit[idx], ratios_nofit_avg[idx]))
    print("real label: %.3f"%(labels[idx]))

exit()
outdir="raman/all_ratios"
if not os.path.exists(outdir):
    os.makedirs(outdir)
todo = ["2gaussians", "2gaussians_avg", "3gaussians", "3gaussians_avg"]#, "2gaussians_num", "2gaussians_num_avg", "3gaussians_num", "3gaussians_num_avg"]
for x in todo:
    s=20
    r_min = 0.0
    r_max = 30.0
    plt.figure(figsize=(6,6))
    if x =="2gaussians":
        ratios = ratios2
    elif x =="2gaussians_avg":
        ratios = ratios2_avg
    elif x =="2gaussians_num":
        ratios = ratios2_n
    elif x =="2gaussians_num_avg":
        ratios = ratios2_n_avg
    elif x =="3gaussians":
        ratios = ratios3
    elif x =="3gaussians_avg":
        ratios = ratios3_avg
    elif x =="3gaussians_num":
        ratios = ratios3_n
    elif x =="3gaussians_num_avg":
        ratios = ratios3_n_avg

    plt.scatter(labels, ratios, marker="o", facecolor="C0", edgecolor="k", alpha=0.7, s=s)
    for idx,n in enumerate(names_unique):
        plt.text(labels[idx]+0.2, ratios[idx]-0.25, n, fontsize=8)
    plt.plot([r_min,r_max],[r_min,r_max],"k-")
    plt.xlabel("Ratio (manual fit)")
    if "2" in x:
        plt.ylabel("Ratio (2 Gaussians)")
    elif "3" in x:
        plt.ylabel("Ratio (3 Gaussians)")
    plt.xlim([r_min,r_max])
    plt.ylim([r_min,r_max])
    plt.savefig("%s/ratio_%s.png"%(outdir, x), dpi=300)
    plt.close()



time1=time.time()
time_for_all_ratios = time1-time0
print("Time needed to calculate all ratios: %.2f seconds"%(time_for_all_ratios))



do_all_plots=True

if do_all_plots:
    outdir="raman/all_fits"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    #sample_number = 17
    for sample_number in names_unique:
        sample_number = int(sample_number)
        print("plot sample %i"%(sample_number))
        for num_gaussians in [2,3]:

            plt.figure()
            idx = [i for i in range(len(names_unique)) if int(names_unique[i])==sample_number][0]
            plt.plot(data_x_avaraged[idx], data_y_avaraged[idx], "-", color="C0", label="sample %i"%(sample_number))

            # 0 is x, 1 is y, 2 is y_init, 3 is fit params init, 4 is fit params opt
            if num_gaussians==2:
                plt.plot(fits2[idx][0], fits2[idx][1], "g--", label="fit")
                gaussians = u.get_gaussians(fits2[idx][0], fits2[idx][4])
                for gidx, g in enumerate(gaussians):
                    plt.plot(fits2[idx][0], g, "k--")
            elif num_gaussians==3:
                plt.plot(fits3[idx][0], fits3[idx][1], "g--", label="fit")
                gaussians = u.get_gaussians(fits3[idx][0], fits3[idx][4])
                for g in gaussians:
                    plt.plot(fits3[idx][0], g, "k--")

            '''
            left_min, max_1, min_1, max_2, right_min = maxima_minima[idx]
            s=50
            plt.scatter([left_min[0]], [left_min[1]], marker="s", facecolor="C0", edgecolor="k", s=s)
            plt.scatter([max_1[0]], [max_1[1]], marker="^", facecolor="C0", edgecolor="k", s=s)
            plt.scatter([max_2[0]], [max_2[1]], marker="^", facecolor="C0", edgecolor="k", s=s)
            plt.scatter([min_1[0]], [min_1[1]], marker="v", facecolor="C0", edgecolor="k", s=s)
            plt.scatter([right_min[0]], [right_min[1]], marker="s", facecolor="C0", edgecolor="k", s=s)
            for counter,idx2 in enumerate(indeces_to_average_all[idx]):
                plt.plot(data_x[idx2], data_y[idx2], "-", color="C1", alpha=0.7, linewidth=1)
                
                left_min, max_1, min_1, max_2, right_min = maxima_minima_all[idx][counter]
                plt.scatter([left_min[0]], [left_min[1]], marker="s", facecolor="C1", edgecolor="k", s=s)
                plt.scatter([max_1[0]], [max_1[1]], marker="^", facecolor="C1", edgecolor="k", s=s)
                plt.scatter([max_2[0]], [max_2[1]], marker="^", facecolor="C1", edgecolor="k", s=s)
                plt.scatter([min_1[0]], [min_1[1]], marker="v", facecolor="C1", edgecolor="k", s=s)
                plt.scatter([right_min[0]], [right_min[1]], marker="s", facecolor="C1", edgecolor="k", s=s)
            '''

            plt.legend(loc="best")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xlim([xmin, xmax])
            plt.ylim([ymin, ymax])
            plt.savefig("%s/raman_sample_%i_%igaussians.png"%(outdir, sample_number, num_gaussians), dpi=120)
            plt.close()



outdir="raman/data"
if not os.path.exists(outdir):
    os.makedirs(outdir)
s=20
for mainidx in range(1):
    num=9
    fig, axes = plt.subplots(num,num, figsize=(20,20))
    i, j = 0, 0
    idx_start = num*num*mainidx
    idx_end1 = min(num*num*(mainidx+1), num_samples)
    idx_end2 = num*num*(mainidx+1)
    for idx in range(idx_start, idx_end1):
        axes[i][j].plot(data_x_avaraged[idx], data_y_avaraged[idx], "-", color="C0", label=names_unique[idx], linewidth=1)

        left_min, max_1, min_1, max_2, right_min = maxima_minima[idx]
        axes[i][j].scatter([left_min[0]], [left_min[1]], marker="s", facecolor="C0", edgecolor="k", s=s)
        axes[i][j].scatter([max_1[0]], [max_1[1]], marker="^", facecolor="C0", edgecolor="k", s=s)
        axes[i][j].scatter([max_2[0]], [max_2[1]], marker="^", facecolor="C0", edgecolor="k", s=s)
        axes[i][j].scatter([min_1[0]], [min_1[1]], marker="v", facecolor="C0", edgecolor="k", s=s)
        axes[i][j].scatter([right_min[0]], [right_min[1]], marker="s", facecolor="C0", edgecolor="k", s=s)

        axes[i][j].text(380,2500,names_unique[idx],fontsize=10) # 100, 2000
        for idx2 in indeces_to_average_all[idx]:
            axes[i][j].plot(data_x[idx2], data_y[idx2], "-", color="C1", alpha=0.7, linewidth=1)
        j+=1
        if j==num:
            j=0
            i+=1
    i, j = 0, 0
    for idx in range(idx_start, idx_end2):
        axes[i][j].get_xaxis().set_ticks([])
        axes[i][j].get_yaxis().set_ticks([])
        axes[i][j].set_xlim([xmin, xmax])
        axes[i][j].set_ylim([ymin, ymax])
        j+=1
        if j==num:
            j=0
            i+=1
    #plt.legend(loc="best")
    #plt.xlabel("x")
    #plt.ylabel("y")

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig("raman/raman_all_%i.png"%(mainidx), dpi=120)
    plt.close()



np.savetxt("%s/data_raman_wavelengths.txt"%(outdir), data_x_avaraged)
np.savetxt("%s/data_raman_intensities.txt"%(outdir), data_y_avaraged)
outfile=open("%s/data_raman_names.txt"%(outdir) ,"w")
for name in names_unique:
    outfile.write("%s\n"%(name))
outfile.close()



