__author__ = 'Eng-Hassan Masssry'
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
def gaussian(m,n,sigma):
    g=np.zeros((m,n))
    mm=m//2
    nn=n//2
    for x in range(-mm,mm+1):
        for  y in range(-nn,nn+1):
            p1=1/(sigma*(2*np.pi)**2)
            p2=np.exp(-(x**2+y**2)/(2*sigma**2))
            g[x+mm,y+nn]=p1*p2
    return g/g.sum()
#gaussian window
w=gaussian(5,5,2)
def show_pyr(pyr):
    plt.axes([-.1,.25,.7,.7])
    plt.imshow(pyr[0],cmap='gray')
    plt.axes([.35,.4,.45,.45])
    plt.imshow(pyr[1],cmap='gray')
    plt.axes([.65,.45,.26,.26])
    plt.imshow(pyr[2],cmap='gray')
    plt.axes([.85,.5,.15,.15])
    plt.imshow(pyr[3],cmap='gray')
    plt.figure()
def reduce(G0):
    r,c=G0.shape
    G1=np.zeros((r//2,c//2))
    for i in range(r//2):
        for j in range(c//2):
            sum=0
            #it depend on the size of gaussian window
            for m in range(-2,3):
                for n in range(-2,3):
                    ii=2*i+m
                    jj=2*j+m
                    if ii<r:
                        if ii>=0:
                            if jj<c:
                                if jj>=0:
                                    sum+=w[m+2,n+2]*G0[ii,jj]
            G1[i,j]=sum
    return G1
def expand(G):
    r,c=G.shape
    EPG=np.zeros((r*2,c*2))
    for i in range(r*2):
        for j in range(c*2):
            sum=0
            for p in range(-2,3):
                for q in range(-2,3):
                    ii=(i-p)/2
                    jj=(j-q)/2
                    if((ii<0)|(jj<0)):
                        continue
                    try:
                        sum+=w[p+2,q+2]*G[ii,jj]
                    except:
                        pass

            EPG[i,j]=sum
    return EPG
rgb_img1=skimage.io.imread('p1.jpg')
img1=skimage.color.rgb2gray(rgb_img1)
rgb_img2=skimage.io.imread('p2.jpg')
img2=skimage.color.rgb2gray(rgb_img2)
print('building pyr1')
#gaussian pyramid for first image
G=img1.copy()
gaus_pyr1=[G]
for i in range(3):
    G=reduce(G)
    gaus_pyr1.append(G)
show_pyr(gaus_pyr1)
print('building pyr2')
#gaussian pyramid for second image
G=img2.copy()
gaus_pyr2=[G]
for i in range(3):
    G=reduce(G)
    gaus_pyr2.append(G)
show_pyr(gaus_pyr2)
#*************************    ********************
print('building laplacian pyr1')
#laplacian pyramid for first image
#the output list of laplacian pyramid [L3,L2,L1,L0
lap_pyr1=[gaus_pyr1[-1]]
for i in range(2,-1,-1):
    L=gaus_pyr1[i]-expand(gaus_pyr1[i+1])
    lap_pyr1.append(L)
lap_pyr1.reverse()
show_pyr(lap_pyr1)
####################
print('building laplacian pyr2')
#laplacian pyramid for second image
lap_pyr2=[gaus_pyr2[-1]]
for i in range(2,-1,-1):
    L=gaus_pyr2[i]-expand(gaus_pyr2[i+1])
    lap_pyr2.append(L)
lap_pyr2.reverse()
show_pyr(lap_pyr2)
# plinding the to laplacian of the tow image
print('building plinding pyr')
plinding_lap_pyr=[]
for i in range(4):
    r,c=gaus_pyr1[i].shape
    L=np.hstack((lap_pyr1[i][:,:c//2],lap_pyr2[i][:,c//2:]))
    plinding_lap_pyr.append(L)
show_pyr(plinding_lap_pyr)
#########################################
#the original gaussian pyr
print('building original gaussian pyr')
org=plinding_lap_pyr[3]
org_gaus_pyr=[org]
for i in range(2,-1,-1):
    org=plinding_lap_pyr[i]+expand(org)
    org_gaus_pyr.append(org)
org_gaus_pyr.reverse()
show_pyr(org_gaus_pyr)
print('showing  the result')
plt.imshow(np.hstack((img1[:,:112],img2[:,112:])),cmap='gray')
plt.show()


