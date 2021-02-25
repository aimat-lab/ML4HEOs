import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import utils as u
import time

np.random.seed(43)

num_to_reduce = 10



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


if not os.path.exists("xrd"):
    os.makedirs("xrd")
outdir="xrd/data_ml"
if not os.path.exists(outdir):
    os.makedirs(outdir)
np.savetxt("%s/data_xrd_wavelengths_ref.txt"%(outdir), data_x_ref)
np.savetxt("%s/data_xrd_intensities_ref.txt"%(outdir), data_y_ref)
outfile=open("%s/data_xrd_names_ref.txt"%(outdir) ,"w")
for name in names_ref:
    outfile.write("%s\n"%(name))
outfile.close()


#####################
# generate fake data
#####################


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


