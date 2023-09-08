# %%
import matplotlib.pyplot as plt
# set_matplotlib_formats('pdf', 'png')
import matplotlib as mpl
import numpy as np

import os

mpl.rc( 'savefig', dpi=75)
mpl.rc( 'figure', autolayout=False,figsize=(18,14),titlesize=30)
mpl.rc( 'axes', labelsize=14,titlesize=20)
mpl.rc( 'lines', linewidth=2.0, markersize=8)
# mpl.rc( 'font', size=14,family='Times New Roman', serif='cm')
# mpl.rc( 'font', size=14,family='DejaVu Sans', serif='cm')
mpl.rc( 'legend', fontsize=14)
# mpl.rc( 'text', usetex=True)

mpl.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{helvet}',
    r'\usepackage{sansmath}',
    r'\sansmath',
    r'\renewcommand{\familydefault}{\sfdefault}',
    r'\usepackage[T1]{fontenc}',
    r'\usepackage{graphicx}',
    r'\usepackage{relsize}',
    r'\newcommand{\bigpi}{\scalebox{5}{\ensuremath{\pi}}}'
]

# %%
# grid/time/sed parameters
L=20000
dt=0.001
dx=200
M=np.int(L/dx)
print('Streamwise spatial resolution (number of resolved points): M='+str(np.int(L/dx)))
print('Temporal resolution (model time step): $\Delta{t}$=%g days' % (dt*365.25))

au=0.95
do_random = False
random_seed = 2
rqh=1

# total time is Ntoprint*Nprint*dt in years. e.g. 1500*4000*0.001=6000y
Ntoprint=1000 # how often to record /dt. e.g. 4000 = 4y 
Nprint= 10 # how many times to record. 
            # e.g.. 1500=1500 records of 4 years for total 6000y
N_subset_print_step = 1  # more grouping. e.g. plot every 10 = 10*4y 

# %%
# model parameters
qbfeed0=0.000834             # Bedload feed  [m^2/s]
Tc=40                        # Reference duration   [y]
high_year=10                 # Duration of high feed rate  [y]
rt=high_year/Tc              # Fraction of time at high feed rate  [-]
upl=0.005                    # Uplift rate  [m/y] 0.005
Lr=1                         # Length of reach  [m]
pcr=0.05                     # p_l in paper
pclr=1-pcr                   # p_h in paper
peta=(1-pcr)/(pclr-pcr)      # p in paper
alt=4                        # ?
nt=1.5                       # ?
tausc=0.0495                 # Critical Shields number  [-]
fis=1                        # ?
Rr=1.65                      # Submerged specific gravity of sediment  [-]
beta=0.00005                 # Bedrock abrasion coefficient  [L^-1]
g=9.81                       # Acceleration due to gravity  [m/s^2]
Cz=10                        # Chezy resistance coefficient  [-]
Qf=300                       # Flood discharge  [m^3/s]
B=100                        # Channel width  [m]
intermittency=0.05           # Flood intermittency  [-]
lambda_a=0.35                # Porosity of alluvium  [-]
D=0.02                       # Gravel grain size  [m]
eta_a_initial=0.5            # Initial alluvial thickness  [m]
slope_initial=0.004          # Initial slope (bedrock and bed)  [-]

# %%
# sedimentograph
if not do_random:
    h_Yr=Tc*rt
    print(h_Yr)
    Yr=np.arange(0,Ntoprint*Nprint*dt+dt,dt)
    print(Yr.size)
    h_Yr_step=np.int(h_Yr/dt)
    print(h_Yr_step)
    Yr_step=np.int(Tc/dt)
    print(Yr_step)

    qbfeed=np.zeros((Yr.size))

    for i in range(0,np.int(Ntoprint*Nprint*dt/Tc),1):
        qbfeed[0]=rqh*qbfeed0

        h_Yr_index=np.arange(i*Tc/dt+1,i*Tc/dt+h_Yr_step+1,1)
        h_Yr_index=h_Yr_index.astype(int)
        qbfeed[h_Yr_index]=rqh*qbfeed0*np.ones(h_Yr_index.size)

        l_Yr_index=np.arange(i*Tc/dt+h_Yr_step+1,i*Tc/dt+Yr_step+1,1)
        l_Yr_index=l_Yr_index.astype(int)
        qbfeed[l_Yr_index]=(1-rt*rqh)/(1-rt)*qbfeed0*np.ones(l_Yr_index.size)

if not do_random:
    fh = plt.figure(figsize=(14,10))
    ax = plt.gca()

    _ = plt.plot(Yr,qbfeed,'-',label='$r_{qh}$='+str(rqh),lw=1 );
    _ = plt.title('Sedimentograph', fontsize=24);
    _ = plt.xlabel('Time, t  [y]',fontsize=24);
    _ = plt.ylabel('Sediment feed, $q_{af}$  [m$^3$/s]', fontsize=24);
    ax.tick_params(labelsize=20);
    _ = plt.xlim(0,np.int(Ntoprint*Nprint*dt));
    # plt.xticks(np.arange(0,np.int(Ntoprint*Nprint*dt)+10,10*100));
    _ = plt.legend(loc='upper right');

    plt.show();
    #plt.savefig('sedimentograph.png');
    #np.savetxt('qbfeed.csv',qbfeed);

# I am skiping the random functionality since I don't use it

qbfeed[:]=rqh*qbfeed0  # uncomment for times shorter than a period feed
print(qbfeed)

# %%
# variables initialization
slope_vary=np.zeros(M+1)
etaa=np.zeros(M+1)
etab=np.zeros(M+1)

eta_a=np.zeros([M+1,Ntoprint+1])
eta_b=np.zeros([M+1,Ntoprint+1])
slope=np.zeros([M+1,Ntoprint+1])
slope_b=np.zeros([M+1,Ntoprint+1])

eta_a_1=np.zeros([M+1,Nprint])
eta_b_1=np.zeros([M+1,Nprint])
eta_ab_1=np.zeros([M+1,Nprint])
slope_b_1=np.zeros([M+1,Nprint])
slope_1=np.zeros([M+1,Nprint])
Qtt=np.zeros([M+1,Nprint])
PC=np.zeros([M+1,Nprint])
PA=np.zeros([M+1,Nprint])

xx=np.arange(0,L+dx,dx)
s_i=slope_initial*np.ones(M+1)
eta_a[:,0]=eta_a_initial*np.ones(M+1)
eta_b[:,0]=L*slope_initial-xx*slope_initial
slope[:,0]=s_i
slope_b[:,0]=s_i
#print(eta_a[:,0],eta_b[:,0],slope[:,0],slope_b[:,0])

qstarx0=np.zeros(M+1)
qstarx=np.zeros(M+1)
qtt=np.zeros(M+1)
pc=np.zeros(M+1)
qttst=np.zeros(M+1)
qqtf=np.zeros(Ntoprint+1)
qtstar=np.zeros(M+2)

# %%
# matrices for methods
A=np.zeros([M+1,M+1])
A[0,0]=1
A[M,M]=-1
A[0,1]=-1
A[M,M-1]=1
for i in range(1,M,1):
    A[i,i+1]=-0.5
    A[i,i-1]=0.5
print(A,A.size,np.size(A,axis=1))

au = 0.95
Bm=np.zeros([M+1,M+2])
Bm[M,M]=-1
Bm[M,M+1]=1
for i in range(0,M,1):
    Bm[i,i]=-au
    Bm[i,i+1]=2*au-1
    Bm[i,i+2]=1-au
print(Bm,Bm.size,np.size(Bm,axis=0))
# %%
# define integrate function
def integrate():
    seconds_per_year = (365.25)*24*60*60
    for w in range(0,Nprint,1):         # Nprint  =  1500
        print(w,end=' ',flush=True)
        for j in range(0,Ntoprint,1):   # Ntoprint = 4000
            slope_vary=np.dot(A,(eta_a[:,j]+eta_b[:,j])/dx)
            slope_vary[slope_vary<0]=0
            taux=(((Qf/B)**2)/(Cz**2)/g)**(1/3)*(slope_vary**(2/3))/Rr/D
            qstarx0=(fis*taux-tausc)
            qstarx0[qstarx0<0]=0
            qstarx=alt*(qstarx0)**nt

            qtt=((Rr*g*D)**0.5)*D*qstarx

            pc = pcr+(pclr-pcr)*eta_a[:,j]/Lr
            pc[eta_a[:,j]>=(Lr*peta)]=1
            pc[eta_a[:,j]<0]=0

            pa = (pc-pcr)/(1-pcr)
            qttst = pa*qtt
    #         qttst=(pc-pcr)/(1-pcr)*qtt

            qqtf[j]=qbfeed[j+w*Ntoprint]
            qtstar=np.append(qqtf[j],qttst)
            #==============================================================================
            # Calculate alluvial height
            #==============================================================================
            #        etaa=eta_a[:,j].copy()
            etaa=eta_a[:,j]
            etaa -= ( (dt*seconds_per_year*intermittency/(1-lambda_a)/dx)
                         *np.dot(Bm,qtstar)/pc )
            etaa[etaa<0]=0
            if etaa[M]>(Lr*peta):
                etaa[M]=Lr*peta

            #        eta_a[:,j+1]=etaa.copy()
            eta_a[:,j+1]=etaa
            #==============================================================================
            # Calculate bed height
            #==============================================================================
            #        etab=eta_b[:,j].copy()
            etab=eta_b[:,j]
            if upl!=0:
                etab += dt*(upl-beta*seconds_per_year*intermittency*(1-pa)*qttst)
            etab[etab<0]=0
            etab[M]=0
            #        eta_b[:,j+1]=etab.copy()
            eta_b[:,j+1]=etab

        eta_a[:,0]=eta_a[:,Ntoprint].copy()
        eta_b[:,0]=eta_b[:,Ntoprint].copy()

        eta_a_1[:,w]=eta_a[:,Ntoprint].copy()
        eta_b_1[:,w]=eta_b[:,Ntoprint].copy()
        eta_ab_1[:,w]=eta_a[:,Ntoprint]+eta_b[:,Ntoprint]
        slope_b_1[:,w]=np.dot(A,eta_b_1[:,w]/dx)
        slope_1[:,w]=np.dot(A,(eta_a_1[:,w]+eta_b_1[:,w])/dx)
        Qtt[:,w]=qtt
        PC[:,w]=pc
        PA[:,w]=pa

# %%
# run model
print('Integrating over %d steps:' % Nprint)
integrate()
print('\nDone')

# %%
# plot prepare
xx=np.arange(0,L+dx,dx)
ini_a=eta_a_initial*np.ones(M+1)
ini_b=L*slope_initial-xx*slope_initial
ini_ab=ini_a+ini_b
ini_sb=slope_initial*np.ones(M+1)
ini_qtt=np.zeros([M+1])
ini_PC=peta*np.ones(M+1)
ini_PA=(peta-pcr)/(1-pcr)*np.ones(M+1)
dist=np.append(0,np.arange(0,(M+1)*dx,dx))
time=np.append(0,np.arange(0,(Nprint+1)*Ntoprint,Ntoprint))
# My fix for round error
time = time * dt

list_data=[eta_a_1, eta_b_1,eta_ab_1,slope_b_1,slope_1,Qtt,PC,PA]
ini_data=[ini_a,ini_b,ini_ab,ini_sb,ini_sb,ini_qtt,ini_PC,ini_PA]
list_data1=[[]]*8
list_data2=[[]]*8

name=(['eta',
       'etab',
       'bedd',
       'slope_bed',
       'slope',
       'Qtt',
       'PC',
       'PA'])
name1=(['Alluvial bed thickness, $\eta_a$  [m]',
        'Bedrock elevation, $\eta_b$  [m]',
        'Bed elevation, $\eta_b+\eta_b$  [m]',
        'Bedrock bed slope, $S_b$  [-]',
        'Alluvial bed slope, $S$  [-]',
        'Capacity bedload transport/unit channel width $q_{ac}$  [m$^2$/s]',
        'Alluvial cover factor, $p$  [-]',
        'Adjusted alluvial cover factor, $p_a$  [-]'])

# %%
# plot store
Nwant0=Ntoprint*dt*1
Nwant00=np.arange(Nwant0,Nprint*Ntoprint*dt+Nwant0,Nwant0)
Nwant00=Nwant00[Nwant00<=Nprint*Ntoprint*dt]
position=np.zeros(Nwant00.size)

for i in range(0,int(Nwant00.size),1):
    position[i]= np.isclose(time, Nwant00[i], atol= dt/2).nonzero()[0]
position=position.astype(np.int32)

for i in range(0,8,1):
    list_data1[i]=np.zeros([M+2,Nprint+2])
    list_data1[i][0,:]=time
    list_data1[i][:,0]=dist
    list_data1[i][1:(M+2),1]=ini_data[i]
    list_data1[i][1:(M+2),2:(Nprint+2)]=list_data[i]
    #np.savetxt(name[i]+'.csv',list_data1[i],delimiter=',')
    
    list_data2[i]=np.zeros([M+2,Nwant00.size+2])
    list_data2[i][:,0]=list_data1[i][:,0]
    list_data2[i][:,1]=list_data1[i][:,1]
    list_data2[i][:,2:(position.size+2)]=list_data1[i][:,position]
    #np.savetxt(name[i]+'_'+str(int(Nwant0))+'.csv',list_data2[i],delimiter=',')

# %%
# plot figures
# every record is at Ntoprint*dt so Nwant is how often
# to pick a record to plot. e.g. assuming default
# Ntoprint*dt*100 = 4000*0.001*100 = 400y
Nwant=Ntoprint*dt*1
N_figures = 8

print(f'Plotting {N_figures} graphs over range 1:{Nprint} with steps {int(Ntoprint//N_subset_print_step)}, {int(Nwant/(Ntoprint*dt))}')

x_label = 'Streamwise coordinate, $x$  [km]'

for i in range(0,N_figures,1):
    # Choose data to plot
    list_data1[i]=np.zeros([M+2,Nprint+2])
    list_data1[i][0,:]=time
    list_data1[i][:,0]=dist
    list_data1[i][1:(M+2),1]=ini_data[i]
    list_data1[i][1:(M+2),2:(Nprint+2)]=list_data[i]

    # Create figure
    fh=plt.figure()
    ax=plt.gca()
    
    # Plot selected time slices
    for j in range (1,Nprint+2,int(Nwant/(Ntoprint*dt))):
        _ = plt.plot(xx/1000,list_data1[i][1:,j],
                     label=str(Ntoprint*dt*(j-1))+'yrs', lw=2)
        
    # Colorize lines in a meaningful, progressive fashion
    colormap = plt.cm.brg   
    # Forward
#     colors = [colormap(idx) for idx in np.linspace(0,1,len(ax.lines))]
    # Reversed
    colors = [colormap(idx) for idx in np.linspace(1,0,len(ax.lines))]
    for idx,line in enumerate(ax.lines):
        line.set_color(colors[idx])
    _ = plt.legend(loc='upper right',fontsize=16)
    
    # Label axes etc
    _ = plt.xlabel(x_label,fontsize=24)
    _ = plt.ylabel(name1[i],fontsize=24)
    if not do_random:
        _ = plt.title('MRSAA $r_{qh}$='+str(rqh)
                      +', high='+str(high_year)+' yrs',fontsize=24)
    else:
        _ = plt.title('MRSAA $r_{qh}$='+str(begin_random)+'-'+str(stop_random)
                 +', high='+str(high_year)+' yrs',fontsize=24)
    _ = ax.tick_params(labelsize=24)

    save_folder = "C:/Users/Paquito/Desktop/MRSAAc/plot2show/1200years/"
    #_ = plt.savefig(save_folder+name[i]+'_'+str(int(Nwant))+'_annot'+'.jpg', bbox_inches='tight')
    #_ = plt.savefig(save_folder+name[i]+'_'+str(int(Nwant))+'_annot'+'.pdf')
    _ = plt.show()

# %%
# Done
#save_folder = "C:/Users/Paquito/Desktop/MRSAAc/plot2show/1200years/"
#os.path.exists(save_folder)
# %%
