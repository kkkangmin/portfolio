import numpy as np
import thomas as tho
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

R=300 # xmax
Kx=100 # 기초 자산 1 의 행사가격
Ky=100 # 기초 자산2의 행사가격
Nx=61 # 기초 자산 1의 격자 갯수
Ny=61 # 기초 자산2의 격자 갯수
x=np.linspace(0,R,Nx) # 기초 자산1
y=np.linspace(0,R,Ny) # 기초 자산2
x_volatility=0.3 # 기초 자산1의 변동성
y_volatility=0.3 # 기초 자산2의 변동성
rho=0.3 # x,y의 상관 계수
r=0.03; # 이자율
h=x[1]-x[0] # 공간 격자 크기
dt=1/365 # 시간 격자 크기
T=1 # 만기
Nt=T/dt # 시간 간격 갯수
# 유한차분법으로 옵션 가격을 구하기 위한 초깃값
u0 = np.zeros((Nx, Ny))
for i in range (Nx):
    for j in range (Ny):
        u0[i, j] = np.maximum(np.maximum(x[i] - Kx, y[j] - Ky), 0)
# Payoff 그래프 그리기
X, Y = np.meshgrid(x, y)
fig1 = plt.figure()
ax = fig1.gca(projection = '3d')
ax.plot_surface(X, Y, u0[:,:], cmap = plt.cm.gray)
ax.view_init(elev = 30., azim = -132)
ax.set_xlabel('x', fontsize = 10)
ax.set_ylabel('y', fontsize = 10)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('Payoff', rotation = 90, fontsize = 10)
plt.show()
# x, y에 대한 0으로 된 벡터 생성
[ax, dx, cx, ay, dy, cy] = map(np.zeros, [Nx-2, Nx-2, Nx-2, Nx-2, Nx-2, Nx-2])
# 유한차분법을 사용하기 위한 x,y 에 대한 계수
#식 (3.28) 에서 확인 가능
ax[:] = -(x_volatility*x[1:Nx-1])**2/(2*h**2) + r*x[1:Nx-1]/(2*h)
dx[:] = 1/dt + (x_volatility*x[1:Nx-1]/h)**2 + r/2
cx[:] = -(x_volatility*x[1:Nx-1])**2/(2*h**2) - r*x[1:Nx-1]/(2*h)
# 식 (3.31)에서 확인 가능
ay[:] = -(y_volatility*y[1:Ny-1])**2/(2*h**2) + r*y[1:Ny-1]/(2*h)
dy[:] = 1/dt + (y_volatility*y[1:Ny-1]/h)**2 + r/2
cy[:] = -(y_volatility*y[1:Ny-1])**2/(2*h**2) - r*y[1:Ny-1]/(2*h)
u=u0; u2=u0;
[fx, fy] =map(np.zeros, [Nx-2, Ny-2])
# OSM과 토마스 알고리즘을 이용해 옵션 가격 계산
for n in range(int(Nt)):
    #hybrid 경계조건
    u[0, 0:Ny-1] = 2*u[1, 0:Ny-1] - u[2, 0:Ny-1]
    u[0:Nx-1, 0] = 2*u[0:Nx-1, 1] - u[0:Nx-1, 2]
    u[Nx-1, 2:Ny-2] = 2*u[Nx-2, 2:Ny-2] - u[Nx-3, 2:Ny-2]
    u[2:Nx-2, Ny-1] = 2*u[2:Nx-2, Ny-2] - u[2:Nx-2, Ny-3]
    u[1, Ny-1] = 2*u[2, Ny-2] - u[3, Ny-3]
    u[0, Ny-2] = 2*u[1, Ny-3] - u[2, Ny-4]
    u[0, Ny-1] = 2*u[1, Ny-2] - u[2, Ny-3]
    u[Nx-1, 1] = 2*u[Nx-2, 2] - u[Nx-3, 3]
    u[Nx-2, 0] = 2*u[Nx-3, 1] - u[Nx-4, 2]
    u[Nx-1, 0] = 2*u[Nx-2, 1] - u[Nx-3, 2]
    u[Nx-1, Ny-1] = 2*u[Nx-2, Ny-2] - u[Nx-3, Ny-3]
    u[Nx-2, Ny-1] = 2*u[Nx-3, Ny-2] - u[Nx-4, Ny-3]
    u[Nx-1, Ny-2] = 2*u[Nx-2, Ny-3] - u[Nx-3, Ny-4]
    # x축으로 풀기
    for j in range (1, Ny-1):
        fx[:] = u[1:Nx-1, j]/dt + 0.5*rho*x_volatility*y_volatility*x[1:Nx-1]*y[j]*(u[2:Nx, j+1] + u[0:Nx-2, j-1] - u[0:Nx-2, j+1] - u[2:Nx, j-1])/(4*h**2)
        # 디티클레 경계조건
        # 식 (3.3.1)에서 확인 가능
        fx[0] = fx[0] - ax[0]*u[0,j]
        fx[Nx-3] = fx[Nx-3] - cx[Nx-3]*u[-1, j]
        u2[1:Nx-1, j] = tho.thomas(ax, dx, cx, fx)
    #hybrid
    u2[0, 0:Ny-1] = 2*u2[1, 0:Ny-1] - u2[2, 0:Ny-1]
    u2[0:Nx-1, 0] = 2*u2[0:Nx-1, 1] - u2[0:Nx-1, 2]
    u2[Nx-1, 2:Ny-2] = 2*u2[Nx-2, 2:Ny-2] - u2[Nx-3, 2:Ny-2]
    u2[2:Nx-2, Ny-1] = 2*u2[2:Nx-2, Ny-2] - u2[2:Nx-2, Ny-3]
    u2[1, Ny-1] = 2*u2[2, Ny-2] - u2[3, Ny-3]
    u2[0, Ny-2] = 2*u2[1, Ny-3] - u2[2, Ny-4]
    u2[0, Ny-1] = 2*u2[1, Ny-2] - u2[2, Ny-3]
    u2[Nx-1, 1] = 2*u2[Nx-2, 2] - u2[Nx-3, 3]
    u2[Nx-2, 0] = 2*u2[Nx-3, 1] - u2[Nx-4, 2]
    u2[Nx-1, 0] = 2*u2[Nx-2, 1] - u2[Nx-3, 2]
    u2[Nx-1, Ny-1] = 2*u2[Nx-2, Ny-2] - u2[Nx-3, Ny-3]
    u2[Nx-2, Ny-1] = 2*u2[Nx-3, Ny-2] - u2[Nx-4, Ny-3]
    u2[Nx-1, Ny-2] = 2*u2[Nx-2, Ny-3] - u2[Nx-3, Ny-4]
    # y축으로 풀기
    for i in range(1, Nx-1):
        fy[:] = u2[i, 1:Ny-1]/dt + 0.5*rho*x_volatility*y_volatility*x[i]*y[1:Ny-1]*(u2[i+1, 2:Ny] + u2[i-1, 0:Ny-2] - u[i-1, 2:Ny] - u[i+1, 0:Ny-2])/(4*h**2)
        # 디리클레 경계조건
        #식 (3.32)에서 확인 가능
        fy[0] = fy[0] - ay[0]*u2[i,0]
        fy[Ny-3] = fy[Ny-3] - cy[Ny-3]*u2[i, -1]
        u[i, 1:Ny-1] = tho.thomas(ay, dy, cy, fy)
# 옵션가격 그래프 그리기
fig2 = plt.figure()
bx = fig2.gca(projection = '3d')
bx.plot_surface(X, Y, u[:,:], cmap = plt.cm.gray)
bx.view_init(elev = 30., azim = -132)
bx.set_xlabel('x', fontsize = 10)
bx.set_ylabel('y', fontsize = 10)
bx. zaxis.set_rotate_label(False)
bx.set_zlabel('Call option price', rotation = 90, fontsize = 10)
plt.show()
# x 가 100인 인덱스를 찾기 위해 np.where 함수 이용
ii = np.argwhere(x==100)
# y 가 100인 인덱스를 찾기 위해 np.where 함수 이용
jj = np.argwhere(y==100)
# x=100, y=100 일 때 콜옵션의 가격 출력
print('Price=%f' %(u[ii,jj]))
