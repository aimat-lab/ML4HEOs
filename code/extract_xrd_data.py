import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import utils as u
import time

np.random.seed(43)

num_to_reduce = 10

data_x = []
data_y = []
names = []
for lineidx,line in enumerate(open("../data/XRDdata_corrected2.csv","r")):
    if lineidx==0:
        for x in line.replace("\n","").split(","):
            if x!="":
                names.append(x)
    elif lineidx>0:
        data_x.append([])
        data_y.append([])
        data_here=[]
        for x in line.replace("\n","").split(","):
            if x=="":
                print("WARNING: found empty entry in line %i"%(lineidx))
            else:
                data_here.append(float(x))
        if len(data_here)!=2*len(names):
            print("WARNING: line %i does not contain %i entries for %i samples, only found %i entries."%(lineidx, len(names), len(names), len(data_here)/2))
        for idx in range(len(names)):
            data_x[-1].append(data_here[2*idx])
            data_y[-1].append(data_here[2*idx+1])

data_x = np.array(data_x).T
data_y = np.array(data_y).T
print(data_x.shape, data_y.shape)

############
# reduce it
############
data_x_reduced, data_y_reduced = u.reduce(data_x, data_y, num=num_to_reduce)
print(data_x_reduced.shape, data_y_reduced.shape)


num_samples = len(names)





if not os.path.exists("xrd"):
    os.makedirs("xrd")
outdir="xrd/data_ml"
if not os.path.exists(outdir):
    os.makedirs(outdir)
np.savetxt("%s/data_xrd_wavelengths.txt"%(outdir), data_x)
np.savetxt("%s/data_xrd_intensities.txt"%(outdir), data_y)
np.savetxt("%s/data_xrd_wavelengths_red.txt"%(outdir), data_x_reduced)
np.savetxt("%s/data_xrd_intensities_red.txt"%(outdir), data_y_reduced)
outfile=open("%s/data_xrd_names.txt"%(outdir) ,"w")
for name in names:
    outfile.write("%s\n"%(name.replace(" ","_")))
outfile.close()


###################
# referece spectra
###################
data_x_ref = []
data_y_ref = []
names_ref = []
for lineidx,line in enumerate(open("../data/Phases_XRD_data.csv","r")):
    if lineidx==0:
        for x in line.replace("\n","").split(","):
            if x!="":
                names_ref.append(x)
    elif lineidx>0:
        data_x_ref.append([])
        data_y_ref.append([])
        data_here=[]
        for x in line.replace("\n","").split(","):
            if x!="":
                data_here.append(float(x))
        if len(data_here)!=2*len(names_ref):
            print("WARNING: line %i does not contain %i entries for %i samples, only found %i entries."%(lineidx, len(names_ref), len(names_ref), len(data_here)/2))
        for idx in range(len(names_ref)):
            data_x_ref[-1].append(data_here[2*idx])
            data_y_ref[-1].append(data_here[2*idx+1])

data_x_ref = np.array(data_x_ref).T
data_y_ref = np.array(data_y_ref).T
print(data_x_ref.shape, data_y_ref.shape)


np.savetxt("%s/data_xrd_wavelengths_ref.txt"%(outdir), data_x_ref)
np.savetxt("%s/data_xrd_intensities_ref.txt"%(outdir), data_y_ref)
outfile=open("%s/data_xrd_names_ref.txt"%(outdir) ,"w")
for name in names_ref:
    outfile.write("%s\n"%(name))
outfile.close()


#####################
# generate fake data
#####################
generate_fake_data=True
if generate_fake_data:

    data_x_synthetic, data_y_synthetic, labels_synthetic = u.generate_fake_data(data_x_ref, data_y_ref, num=20000)


    data_x_synthetic_reduced, data_y_synthetic_reduced = u.reduce(data_x_synthetic, data_y_synthetic, num=num_to_reduce)
    #np.savetxt("%s/data_xrd_wavelengths_synthetic.txt"%(outdir), data_x_synthetic)
    np.savetxt("%s/data_xrd_intensities_synthetic.txt"%(outdir), data_y_synthetic)
    np.savetxt("%s/data_xrd_wavelengths_synthetic_red.txt"%(outdir), data_x_synthetic_reduced)
    np.savetxt("%s/data_xrd_intensities_synthetic_red.txt"%(outdir), data_y_synthetic_reduced)
    outfile=open("xrd/data_ml/phase_pure_or_not_synthetic.txt", "w")
    for x in labels_synthetic:
        outfile.write("%i\n"%(1 if x=="mixed" else 0))
    outfile.close()


####################
# read the SI table
####################

names=[]
compositions=[]
compositions_int=[]
phases_names=[]
phases_percentages=[]
phases_pure_or_not=[]
phase=[]

for lineidx,line in enumerate(open("../data/chemical_composition_table.csv","r")):
    spl = line.replace("\n","").split(",")
    if lineidx>2:
        name=spl[1]
        if name!="":
            names.append("sample_%s"%(name))
            composition=[]
            for x in spl[7:12]:
                if x!="-":
                    composition.append(int(x.split("±")[0]))
                else:
                    composition.append(0)
            if len(composition) != 5:
                print("EXIT, sample %s"%(name))
            compositions.append(composition)
            composition_int=[]
            for x in spl[2:7]:
                if x!="-":
                    composition_int.append(float(x.split("±")[0]))
                else:
                    composition_int.append(0.0)
            if len(composition_int) != 5:
                print("EXIT, sample %s"%(name))
            compositions_int.append(composition_int)

            if len(phase)!=0:
                phases_names.append([p[0] for p in phase])
                phases_percentages.append([p[1] for p in phase])
                if len(phase)==1:
                    phases_pure_or_not.append("pure")
                else:
                    phases_pure_or_not.append("mixed")
                phase=[]
        if spl[12]!="":
            phase_name=spl[12].replace(" ","").replace("\"","").replace("\'","").split("(")[0]
            phase_percentage=float(spl[12].split("(")[1].split("%")[0])
            phase.append([phase_name, phase_percentage])
phases_names.append([p[0] for p in phase])
phases_percentages.append([p[1] for p in phase])
if len(phase)==1:
    phases_pure_or_not.append("pure")
else:
    phases_pure_or_not.append("mixed")


print(num_samples, len(phases_pure_or_not))
mixed_IDs = {"Fm-3m":[7,8,10,11,23,71,88], "Ia-3":[16,22], "P63/m":[]}
# plot combined plots
for phase in ["Fm-3m", "Ia-3", "P63/m"]:
    plt.figure(figsize=[15,10])
    for idx in range(num_samples):
        name = names[idx]
        sample_ID = int(name.split("_")[1])
        if phases_pure_or_not[idx]=="pure" and phases_names[idx][0]==phase:
            plt.plot(data_x[idx], data_y[idx], "-", color="k", linewidth=1, zorder=0)
        else:
            if sample_ID in mixed_IDs[phase]:
                plt.plot(data_x[idx], data_y[idx], "-", color="r", linewidth=1, zorder=1)
    #plt.legend(loc="best")
    plt.xlabel("Angle [degree]")
    plt.ylabel("XRD intensity")
    plt.savefig("xrd/combined_spectra/%s.png"%(phase.replace("/","")), dpi=120)
    plt.close()



######################
# save all the output
######################


outfile=open("xrd/data_ml/experimental_phases.txt","w")
for idx, name in enumerate(names):
    print(name, phases_pure_or_not[idx], phases_names[idx], phases_percentages[idx])
    phase=""
    for idx2, x in enumerate(phases_names[idx]):
        phase+="%s(%.1f%%)"%(x, phases_percentages[idx][idx2])
    outfile.write("%s %s %s\n"%(name, phases_pure_or_not[idx], phase))


outfile=open("xrd/data_ml/compositions.txt","w")
for c in compositions:
    outfile.write("%f %f %f %f %f\n"%(c[0], c[1], c[2], c[3], c[4]))
outfile.close()
outfile=open("xrd/data_ml/compositions_int.txt","w")
for c in compositions_int:
    outfile.write("%f %f %f %f %f\n"%(c[0], c[1], c[2], c[3], c[4]))
outfile.close()


outfile=open("xrd/data_ml/phase_pure_or_not.txt", "w")
for x in phases_pure_or_not:
    outfile.write("%i\n"%(1 if x=="mixed" else 0))
outfile.close()

phases_no_referece = []
outfile=open("xrd/data_ml/phases.txt", "w")
for sample_idx, x in enumerate(phases_names):
    concs = [0.0, 0.0, 0.0, 0.0]
    phases_main = ["Fm-3m", "Ia-3", "P63/m"]
    for pidx, p in enumerate(phases_main):
        for idx,p2 in enumerate(x):
            if p2==p:
                concs[pidx] += phases_percentages[sample_idx][idx]
    
    for idx, p in enumerate(x):
        if p not in phases_main:
            concs[3]+=phases_percentages[sample_idx][idx]
            if p not in phases_no_referece:
                phases_no_referece.append(p)
    print(names[sample_idx], phases_pure_or_not[sample_idx], phases_names[sample_idx], phases_percentages[sample_idx], phases_main, concs)

    outfile.write("%f %f %f %f\n"%(concs[0], concs[1], concs[2], concs[3]))
outfile.close()
print("no reference for phases:")
print(phases_no_referece)

exit()
#######################
# plot all the spectra
#######################

outdir="xrd/all_spectra"
if not os.path.exists(outdir):
    os.makedirs(outdir)


for sample_idx in range(num_samples):
    print("plotting %s (%i of %i)"%(names[sample_idx], sample_idx+1, num_samples))
    phase = ""
    for p_idx, p in enumerate(phases_names[sample_idx]):
        phase+="%s (%.1f%%) "%(p, phases_percentages[sample_idx][p_idx])
    xmin=10.0
    xmax=50.0
    ymin = 0.0
    ymax = 4500.0
    fig, axes = plt.subplots(3,1)
    for i in range(3):
        axes[i].plot(data_x[sample_idx], data_y[sample_idx], "r-", label="%s: %s"%(names[sample_idx], phase))
        #axes[i].plot(data_x_reduced[sample_idx], data_y_reduced[sample_idx], "b.-", label="%s reduced"%(names[sample_idx]))
        axes[i].plot(data_x_ref[i], data_y_ref[i], "k-", label="%s"%(names_ref[i]))
        axes[i].set_xlim([xmin, xmax])
        axes[i].set_ylim([ymin, ymax])
        axes[i].legend(loc="upper right")
        if i in [0,1]:
            axes[i].get_xaxis().set_ticks([])
    axes[2].set_xlabel("Angle [degree]")
    axes[1].set_ylabel("XRD intensity [a.u.]")
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig("%s/%s.png"%(outdir, names[sample_idx]), dpi=300)
    plt.close()




outdir="xrd/all_spectra2"
if not os.path.exists(outdir):
    os.makedirs(outdir)

for sample_idx in range(num_samples):
    print("plotting %s (%i of %i)"%(names[sample_idx], sample_idx+1, num_samples))
    phase = ""
    for p_idx, p in enumerate(phases_names[sample_idx]):
        phase+="%s (%.1f%%) "%(p, phases_percentages[sample_idx][p_idx])
    xmin=10.0
    xmax=50.0
    ymin = 0.0
    ymax = [np.max(data_y[sample_idx]), np.max(data_y_ref[0]), np.max(data_y_ref[1]), np.max(data_y_ref[2])]
    fig, axes = plt.subplots(4,1, figsize=(12,8))
    axes[0].plot(data_x[sample_idx], data_y[sample_idx], "r-", label="%s: %s"%(names[sample_idx], phase), linewidth=0.5)
    for i in range(4):
        if i>0:
            axes[i].plot(data_x_ref[i-1], data_y_ref[i-1], "k-", label="%s"%(names_ref[i-1]), linewidth=0.5)
        axes[i].set_xlim([xmin, xmax])
        axes[i].set_ylim([ymin, 1.2*ymax[i]])
        axes[i].legend(loc="upper right")
        if i in [0,1,2]:
            axes[i].get_xaxis().set_ticks([])
    axes[3].set_xlabel("Angle [degree]")
    axes[2].set_ylabel("               XRD intensity [a.u.]")
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig("%s/%s.png"%(outdir, names[sample_idx]), dpi=600)
    plt.close()


