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
max_val=6
angle=2*math.pi/N
b=int(2*s+1)                         
colors=['red','blue','green','purple','orange','turquoise','magenta','brown','grey']
mag=0.6717   
temp=[1,2,4,8]                   #Bohr magneton/kB

save_switch=True
mag_switch=False   
   
#Heisenberg interaction matrix
Jx=1
Jz=1
fac=Jz/Jx   
Jmatrix=np.zeros([N,N])
for ind in range(N-1):
    Jmatrix[ind+1,ind]=1
Jmatrix[N-1,0]=1
Jmatrix*=Jx               

#anisotropy tensor
Dmatrix=np.zeros([N,4])
Dconst=-4
Dmatrix[:,0]=Dconst
m=mg.reverse_trafo(N,s)

#vectors
Rmatrix=np.zeros([N,3])
Rvec=np.array([0,1,0])
angle=2*math.pi/N
for ind in range(N):
    #rotX=np.array([[1,0,0],[0,math.cos(ind*angle),-math.sin(ind*angle)],[0,math.sin(ind*angle),math.cos(ind*angle)]])
    #rotY=np.array([[math.cos(ind*angle),0,math.sin(ind*angle)],[0,1,0],[-math.sin(ind*angle),0,math.cos(ind*angle)]])
    rotZ=np.array([[math.cos(ind*angle),-math.sin(ind*angle),0],[math.sin(ind*angle),math.cos(ind*angle),0],[0,0,1]])
    vec=np.dot(rotZ,Rvec)
    Rmatrix[ind,:]=vec

#magnetic field
Bvectors=np.zeros([N,3])
Bvec=np.array([0,0,1])
for i in range(N):
    Bvectors[i,:]=Bvec
def Bfield(t):
    return Bvectors
Bfield_name=r'$B^z=1$ T'

hamB=mg.ham_mat_B(Bfield,1,N,s,m)

#Plot
fig_eig2, ax_eig2=plt.subplots()
ax_eig2.set_xlabel(r'$\varphi$ in $^\circ$')  
ax_eig2.set_ylabel(r'$E_\nu$ in K')  
ax_eig2.set_title(r'Unterste Eigenwerte $E_\nu$'+f' für N={N}, s={s}, J={Jz} K, D={Dconst} K, {Bfield_name}')

fig_torZ, ax_torZ=plt.subplots()

ax_torZ.set_xlabel(r'$\varphi$ in $^\circ$')
ax_torZ.set_ylabel(r'$<\hat{\tau}^z>_\nu$ in a.u.')
ax_torZ.set_title(r'$<\hat{\tau}^z>_\nu$'+f' für N={N}, s={s}, J={Jz} K, D={Dconst} K, {Bfield_name}')

fig_exp_vec, ax_exp_vec=plt.subplots()
ax_exp_vec.set_xlabel(r'$<\hat{s}_1^x>_\nu$')
ax_exp_vec.set_ylabel(r'$<\hat{s}_1^y>_\nu$')
ax_exp_vec.set_title(r'($<\hat{s}_1^x>,<\hat{s}_1^y>)$'+f' für N={N}, s={s}, J={Jz} K, D={Dconst} K, {Bfield_name}')

if mag_switch:
    fig_mag_temp, ax_mag_temp=plt.subplots()
    plt.ylim(-0.1,0.1 )
    ax_mag_temp.set_xlabel(r'$\varphi$ in $^\circ$')
    ax_mag_temp.set_ylabel(r'$<\mathcal{M}>(T)$ in J/K')
    ax_mag_temp.set_title(r'$<\mathcal{M}>(T)$'+f' für N={N}, s={s}, J={Jz} K, D={Dconst} K, {Bfield_name}')

phi_array=np.linspace(0,90,360)
for phi in phi_array:
    
    #anisotropy tensor
    Dvec=np.array([math.cos(phi*math.pi/180)*math.sin(10*math.pi/180),math.sin(phi*math.pi/180)*math.sin(10*math.pi/180),math.cos(10*math.pi/180)])
    for ind in range(N):
        rotZ=np.array([[math.cos(ind*angle),-math.sin(ind*angle),0],[math.sin(ind*angle),math.cos(ind*angle),0],[0,0,1]])
        vec=np.dot(rotZ,Dvec)
        for j in range(3):
            Dmatrix[ind,j+1]=vec[j]

    hamJD=hamB+mg.ham_mat_JD(Jmatrix,fac,Dmatrix,N,s,m)

    #Eigenvalues
    vals,vectors=la.eig(hamJD)
    vals=np.real(vals)
    order=np.argsort(vals)     #place, in which the n-lowest number is 
    vals=np.sort(vals)
    
    tarr2=phi*np.ones(max_val)
    ax_eig2.scatter(tarr2,vals[:max_val] , color='blue', marker='.', linewidths=0.00001)
    
    for ind1 in range(max_val):
        tor_res=np.real(mg.tor_exp(ind1,Rmatrix,N,s,m,vectors,order))
        ax_torZ.scatter(phi,tor_res[2], color=colors[ind1], marker='.', linewidths=0.000001) 
        
    Z=np.zeros(len(temp))
    j=0
    for ind1 in temp:
        for ind2 in range(b**N):
            Z[j]+=math.exp(-1/ind1*vals[ind2])
        j+=1
    
    if mag_switch:
        j=0
        for ind1 in temp:
            mag_temp_array=np.zeros(len(temp))
            mag_temp_res=0
            for ind2 in range(N):
                for ind3 in range(b**N):
                    spin_exp_res=np.real(mg.spin_exp(ind2,ind3,N,s,m,vectors,order))
                    mag_temp_res+=math.exp(-vals[ind3]/ind1)*spin_exp_res[2]  
                    mag_temp_array[j]=mag_temp_res/Z[j]
            j+=1
    
    if mag_switch:
        j=0
        for ind1 in temp:
            ax_mag_temp.scatter(phi,mag_temp_array[j], color=colors[j], marker='.', linewidths=0.000001)
        j+=1
    for nu in range(max_val):
        exp_res=np.real(mg.spin_exp(0,nu,N,s,m,vectors,order))     
        ax_exp_vec.scatter(exp_res[0],exp_res[1], color=colors[nu] , marker='.', linewidths=0.000001)
        
labels_nu=[]
for ind in range(max_val):
    labels_nu.append(r'$\nu=$'+str(ind))
ax_torZ.legend(labels_nu,loc='upper right')
ax_exp_vec.legend(labels_nu,loc='upper right')

if mag_switch:
    labels_temp=[]
    for ind in temp:
        labels_temp.append('$T=$'+str(ind)+' K')
        ax_mag_temp.legend(labels_temp,loc='upper right')

     
#Save
#####
if save_switch:        
    fig_eig2.savefig(f's{s}_J{Jz}_D{Dconst}_Bz_evals_rot.png',bbox_inches='tight')
    if mag_switch:
        fig_mag_temp.savefig(f's{s}_J{Jz}_D{Dconst}_'+'Bz.png',bbox_inches='tight')
    fig_torZ.savefig(f's{s}_J{Jz}_D{Dconst}_Bz_'+'Tz_rot.png',bbox_inches='tight')
    fig_exp_vec.savefig(f's{s}_J{Jz}_D{Dconst}_Bz_'+'Sxy_rot.png',bbox_inches='tight')
    mg.ping()               
else:   
    plt.show()
    