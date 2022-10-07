import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import thomas as tho

facevalue = 10000 #액면금액
R = 300 # x,y 도메인의 최댓값
x_volatility = 0.249 # x의 변동성
y_volatility = 0.2182 # y의 변동성
rho = 0.0981 # x와 y의 상관계수
r = 0.0165 # 무위험 이자율
Nx = 61 # x격자 갯수
Ny = Nx #y격자 갯수
h = R/Nx # x, y 격자 간격
x0 = 100 # x의 최초 기준가격
y0 = 100 # y의 최초 기준가격
x = np.linspace(0, R, Nx) # 기초자산1
y = x # 기초자산2
T = 3 # 만기
Nt = 360*T # 시간격자갯수
dt = T/Nt # 시간격자간격
# 낙인 배리어 아래로 떨어지지 않는 경우와 그렇지 않는 경우
# ELS 가격 행렬 생성, 모두 0임
# 그리고 OS 방법을 사용하기 위해 u, ku 와 같은 크기의 행렬 생성
lst = [Nx, Ny]
[u, ku, old_u, old_ku] = map(np.zeros, [lst,lst,lst,lst])
#조기상환시 쿠폰 이자율
coupon_rate = np.array([0.15, 0.125, 0.10, 0.075, 0.05, 0.025])
#조기행사 시점 벡터
step = np.array([np.rint(Nt/6), np.rint(2*Nt/6), np.rint(3*Nt/6), np.rint(4*Nt/6), np.rint(5*Nt/6), Nt+1])
#조기 행사가
strike_price = np.array([0.75, 0.80, 0.85, 0.85, 0.90, 0.90])
dummy = 0.15 # 더미 이자율
kib = 0.50 # 낙인 배리어
# 유한차분법으로 ELS 가격을 구하기 위한 초깃값
for i in range(0, Nx):
    for j in range(0, Ny):
        if(x[i] < kib*x0 or y[j] < kib*y0):
            u[i, j] = np.minimum(x[i], y[j])/x0*facevalue
            ku [i, j] = np. minimum(x[i],y[j])/x0*facevalue
        elif (x[i] <= strike_price[0] *x0 or y[j] <= strike_price[0]*x0):
            u[i, j] = facevalue*(1+dummy)
            ku[i, j] = np.minimum(x[i], y[j])/x0*facevalue
        else:
            u[i, j] = facevalue*(1 + coupon_rate[0])
            ku[i, j] = facevalue*(1 + coupon_rate[0])
# 유안차분법을 사용하기 위한 계수
# 식 (3.28)에서 확인할 수 있다
[ax, dx, cx, ay, dy, cy] = map(np.zeros, [Nx-2, Nx-2, Nx-2, Ny-2, Ny-2, Ny-2])
ax[:] = -0.5*(x_volatility*x[1:Nx-1]/h)**2 + 0.5*r*x[1:Nx-1]/h
dx[:] = 1/dt + (x_volatility*x[1:Nx-1]/h)**2 + r*0.5
cx[:] = -0.5*(x_volatility*x[1:Nx-1]/h)**2 - 0.5*r*x[1:Nx-1]/h
# 선형 경계조건식 (3.12) 에서 확인할 수 있다
ax[Nx-3] = ax[Nx-3] - cx[Nx-3]
dx[Nx-3] = dx[Nx-3] + 2*cx[Nx-3]
#식 (3.31) 에서 확인할 수 있다
ay[:] = -0.5*(y_volatility*y[1:Ny-1]/h)**2 + 0.5*r*y[1:Ny-1]/h
dy[:] = 1/dt + (y_volatility*y[1:Ny-1]/h)**2 + r*0.5
cy[:] = -0.5*(y_volatility*y[1:Ny-1]/h)**2 - 0.5*r*y[1:Nx-1]/h
# 선형 경계조건식 (3.12) 에서 확인할 수 있다
ay[Ny-3] = ay [Ny-3] - cy[Ny-3]
dy[Ny-3] = dy [Ny-3] + 2*cy[Ny-3];
tag = 0 # 조기상환을 진행하기 위한 변수
bx = np.zeros(Nx-2); by = np.zeros(Nx-2);
for n in range(0, Nt):
    # 조기상환일의 페이오프
    if(n == step[tag]):
        #조기상환 조건을 만족하는 가장 작은 x,y 값 찾기
        gx=np.min(np.where(x >= x0*strike_price[tag+1]))
        gy=np.min(np.where(y >= y0*strike_price[tag+1]))
        u[gx:Nx-1, gy:Ny-1] = facevalue*(1 + coupon_rate[tag+1])
        ku[gx:Nx-1, gy:Ny-1] = facevalue*(1 + coupon_rate[tag+1])
        # 다음 조기상환일로 이동
        tag+=1
    # 낙인 배리어보다 아래 있는 x,y 값 찾기
    gx = np.min(np.where(x >= x0*kib))
    gy = np.min(np.where(y >= y0*kib))
    # 낙인 배리어 아래 구간의 u값에 ku값 반영
    u[:, 0:gy+1] = ku[:, 0:gy+1]
    u[0:gx+1, :] = ku[0:gx+1, :]
    old_u = u; old_ku = ku;
    # OSM과 토마스 알고리즘을 이용하여 값 계산
    #x축으로 풀기
    for j in range(1, Ny-1):
        bx[0:Nx-1] = old_u[1:Nx-1, j]/dt + 0.5*rho*x_volatility*y_volatility*x[1:Nx-1]*y[j]*(old_u[2:Nx, j+1] - old_u[2:Nx, j-1] - old_u[0:Nx-2, j+1] + old_u[0:Nx-2, j-1])/(4*h**2)
        u[1:Nx-1, j] = tho.thomas(ax,dx,cx,bx)
    # 보간법을 이용하여 마지막 값을 구해준다
    u[Nx-1, 1:Ny-1] = 2*u[Nx-2, 1:Ny-1] - u[Nx-3, 1:Ny-1]
    u[:, Ny-1] = 2*u[:, Ny-2] - u[:, Ny-3]
    old_u = u;
    # y축으로 풀기
    for i in range (1, Nx-1):
        by[0:Ny-1] = old_u[i, 1:Ny-1]/dt + 0.5*rho*x_volatility*y_volatility *x[i]*y [1: Ny-1]* (old_u[i+1,2: Ny] - old_u[i+1, 0:Ny-2] - old_u[i-1, 2:Ny] + old_u[i-1, 0:Ny-2])/(4*h**2)
        u[i, 1:Ny-1] = tho.thomas(ay, dy, cy, by)
    # 보간법을 이용하여 마지막 값을 구해준다
    u[1:Nx-1, Ny-1] = 2*u[1:Nx-1, Ny-2] - u[1:Nx-1, Ny-3]
    u[Nx-1, :] = 2*u[Nx-2, :] - u[Nx-3, :]
    # x축으로 풀기
    for j in range(1, Ny-1):
        bx[0:Nx-1] = old_ku[1:Nx-1, j]/dt + 0.5*rho*x_volatility*y_volatility*x[1:Nx-1]*y[j]*(old_ku[2:Nx, j+1] - old_ku[2:Nx, j-1] - old_ku[0:Nx-2, j+1] + old_ku[0:Nx-2, j-1])/(4*h**2)
        ku[1:Nx-1, j] = tho.thomas(ax, dx, cx, bx)
    # 보간법을 이용하여 마지막 값을 구해준다
    ku[Nx-1, 1:Ny-1] = 2*ku[Nx-2, 1:Ny-1] - ku[Nx-3, 1:Ny-1]
    ku[:, Ny-1] = 2*ku[:, Ny-2] - ku[:, Ny-3]
    old_ku = ku;
    # y 축으로 풀기
    for i in range (1, Nx-1):
        by[0: Ny-1] = old_ku[i, 1:Ny-1]/dt + 0.5*rho*x_volatility*y_volatility*x[i]*y[1:Ny-1]*(old_ku[i+1, 2:Ny] - old_ku[i+1, 0:Ny-2] - old_ku[i-1, 2:Ny] + old_ku[i-1, 0:Ny-2])/(4*h**2)
        ku[i, 1:Ny-1] = tho.thomas(ay, dy, cy, by)
    # 보간법을 이용하여 마지막 값을 구해준다
    ku[1:Nx-1, Ny-1] = 2*ku[1:Nx-1, Ny-2] - ku[1:Nx-1, Ny-3]
    ku[Nx-1, :] = 2*ku[Nx-2, :] - ku[Nx-3, :]
# 그래프 그리기
X, Y = np.meshgrid(x, y)
fig1 = plt.figure()
ax = fig1.gca(projection = '3d')
ax.plot_surface(X, Y, u, cmap = plt.cm.gray)
ax.view_init(elev = 31, azim = -134)
ax.set_xlabel('x', fontsize = 10)
ax.set_ylabel('y', fontsize = 10)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('ELS Price', rotation = 90, fontsize = 10)
fig2 = plt.figure()
bx = fig2.gca(projection = '3d')
bx.plot_surface(X, Y, ku, cmap = plt.cm.gray)
bx.view_init(elev = 31, azim = -134)
bx.set_xlabel('x', fontsize = 10)
bx.set_ylabel('y', fontsize = 10)
bx.zaxis.set_rotate_label(False)
bx.set_zlabel('ELS Price', rotation = 90, fontsize = 10)
plt.show()

# x=100, y=100 일의 ELS 가격을 찾기 위해 find 함수 이용
ii = np.where(x==100)
jj = np.where(y==100)
# ELS 가격 출력
print('Price=%f' %(u[ii, jj]))
