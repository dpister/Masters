import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import math
import matrixgen as mg

#Font
mg.set_font(legendsize=13,ticksize=12,labelsize=15)

#Variables
N=2                            #number spins         
s=1                            #spin
temp=[1,2,4,8]                 #temperature (for exp-values)
angle=2*math.pi/N
b=int(2*s+1)                         
colors=['red','blue','green','purple','orange','turquoise','magenta','brown','grey']
mag=0.6717                     #Bohr magneton/kB

#Text readin
lebedev=np.array([])
txt=open("lebedev50.txt")
line=1
N_grid=0
while line:
    line=txt.readline().split()              #theta/phi/omega
    lebedev=np.append(lebedev,line)
    N_grid+=1
N_grid-=1
lebedev=np.reshape(lebedev,(N_grid,3))

save_switch=True    
   
#Heisenberg interaction matrix
Jconst=1    
Jmatrix=np.zeros([N,N])
for ind in range(N-1):
    Jmatrix[ind+1,ind]=1
Jmatrix[N-1,0]=1
Jmatrix*=Jconst                

#magnetic field
tstart=-5
tfinish=5
tsteps=500
tstepsize=(tfinish-tstart)/tsteps
Bfield_var=r'$B$ in T'
Bfield_save='Bav'

#anisotropy tensor
Dmatrix=np.zeros([N,4])
Dconst=-8
tang=0*2*math.pi/360
Dvec=np.array([math.cos(tang),0,math.sin(tang)])
for ind in range(N):
    Dmatrix[ind,0]=Dconst
    rotZ=np.array([[math.cos(ind*angle),-math.sin(ind*angle),0],[math.sin(ind*angle),math.cos(ind*angle),0],[0,0,1]])
    vec=np.dot(rotZ,Dvec)
    for j in range(3):
        Dmatrix[ind,j+1]=vec[j]

#defining r-vectors for toroidal moment
Rmatrix=np.zeros([N,3])
Rvec=np.array([0,1,0])
angle=2*math.pi/N
for ind in range(N):
    #rotX=np.array([[1,0,0],[0,math.cos(ind*angle),-math.sin(ind*angle)],[0,math.sin(ind*angle),math.cos(ind*angle)]])
    #rotY=np.array([[math.cos(ind*angle),0,math.sin(ind*angle)],[0,1,0],[-math.sin(ind*angle),0,math.cos(ind*angle)]])
    rotZ=np.array([[math.cos(ind*angle),-math.sin(ind*angle),0],[math.sin(ind*angle),math.cos(ind*angle),0],[0,0,1]])
    vec=np.dot(rotZ,Rvec)
    for j in range(3):
        Rmatrix[ind,j]=vec[j]
        
        
m=mg.reverse_trafo(N,s)
hamJD=mg.ham_mat_JD(Jmatrix,1,Dmatrix,N,s,m)

                                        
#Plots(frame)
####################         
fig_mag_temp, ax_mag_temp=plt.subplots()
ax_mag_temp.set_xlabel(Bfield_var)
ax_mag_temp.set_ylabel(r'$<\mathcal{M}>(T)$ in J/K')
ax_mag_temp.set_title(r'$<\mathcal{M}>(T)$'+f' für N={N}, s={s}, J={Jconst} K, D={Dconst} K, gemittelt über Lebedev Gitter')


#Zeeman term
############
t_array=np.linspace(tstart,tfinish,tsteps)
t_array=np.append(t_array,[0.000001])
eMatrix=np.zeros([N,3])
for t in t_array:
    mag_av_arr=np.zeros(len(temp))
    for points in range(N_grid):
        theta=float(lebedev[points,0])
        phi=float(lebedev[points,1])
        omega=float(lebedev[points,2])
        evec_B=np.array([math.cos(theta)*math.sin(phi),math.sin(theta)*math.sin(phi),math.cos(phi)])
        for ind in range(N):
            eMatrix[ind,:]=evec_B
        def Bfield(t): 
            return t*eMatrix
    
        hamB=mg.ham_mat_B(Bfield,t,N,s,m)

        #Eigenvalues
        hamJDB=hamB+hamJD
        vals,vectors=la.eig(hamJDB)
        vals=np.real(vals)
        order=np.argsort(vals)     #place, in which the n-lowest number is 
        vals=np.sort(vals)

        Z=np.zeros(len(temp))
        j=0
        for ind in temp:
            for ind2 in range(b**N):
                Z[j]+=math.exp(-1/ind*vals[ind2])
            j+=1
    
        #magnetization
        j=0
        for ind1 in temp:
            mag_temp_res=0
            for ind2 in range(N):
                for ind3 in range(b**N):
                    spin_exp_res=np.real(mg.spin_exp(ind2,ind3,N,s,m,vectors,order))
                    mag_temp_res+=math.exp(-vals[ind3]/ind1)*(evec_B[0]*spin_exp_res[0]+evec_B[1]*spin_exp_res[1]+evec_B[2]*spin_exp_res[2])       
            mag_av_arr[j]-=2*omega*mag*mag_temp_res/Z[j]
            j+=1
            
    #Plots(data)
    ############
    j=0
    for ind1 in temp:
        ax_mag_temp.scatter(t,mag_av_arr[j], color=colors[j], marker='.', linewidths=0.000001)
        j+=1

#Labels
#######            
labels_temp=[]
for ind in temp:
    labels_temp.append('$T=$'+str(ind)+' K')
            
ax_mag_temp.legend(labels_temp,loc='upper right')

#Save
#####
if save_switch:        
    fig_mag_temp.savefig(f's{s}_J{Jconst}_D{Dconst}_'+Bfield_save+'_Mav_50.png',bbox_inches='tight')
    mg.ping()
else:                
    plt.show()
txt.close()