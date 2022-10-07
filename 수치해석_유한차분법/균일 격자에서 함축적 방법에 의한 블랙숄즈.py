import numpy as np
import matplotlib.pyplot as plt
import thomas as tho

K = 100 # 행사가격
R = 200 # 도메인의 최댓값
volatility = 0.3 # 변동성
r = 0.03 # 무위험 이자율
T = 1 # 만기
Nx = 31 # x 격차 갯수
Nt = 360 # 시간 격자 갯수
dt = T/Nt # 시간 격차 간격
x = np.linspace(0, R, Nx) # 기초자산
h = x[1] - x[0] # x격자 간격
#초깃값을 위해 만드는 0으로만 이루어진 행렬
u = np.zeros([Nx,Nt+1])
#초깃값
for i in range(0, Nx):
    u[i,0] = np.maximum(x[i] - K, 0)
# 페이오프 그리기
plt.plot(x, u[:, 0], 'k--', label = 'Payoff')
[a,d,c,b] = map(np.zeros, [Nx, Nx, Nx, Nx])
# 유한차분법을 사용하기 위한 계수. 식 (3.11)과 같음.
for i in range(0, Nx):
    a[i]=r*x[i]/(2*h)-(volatility*x[i])**2/(2*h**2)
    d[i]=(1/dt)+((volatility*x[i])**2/(h**2))+r
    c[i]=-r*x[i]/(2*h)-(volatility*x[i])**2/(2*h**2)
# 선형 경계조건. 식 (3.12) 에서 확인 가능
d[Nx-1] = d[Nx-1] + 2*c[Nx-1]
a[Nx-1] = a[Nx-1] - c[Nx-1]
# 유한차분법과 토마스 알고리즘을 이용해 옵션 가격 계산
for n in range(0, Nt):
    b=u[:, n]/dt
    u[:, n+1] = tho.thomas(a, d, c, b)
# np.where 함수를 이용하여 x=100인 x의 인덱스 찾기
ii = np.where(x == 100)
# 기초 자산 가격이 100일 경우 콜옵션 가격 그리기
plt.scatter(x[ii], u[ii,Nt],\
            color = 'k', label = 'Call price at underlying asset = 100')
# 옵션가격 그래프 그리기
plt.plot(x, u[:,Nt], 'k-', label = 'Call Price')
plt.xlabel("x", fontsize = 10)
plt.ylabel("Call Option Price", fontsize = 10)
plt.legend(loc = 'upper left')
plt.show()
# 기초자산가격이 100 일 경우 콜옵션 가격 출력
print('Price =%f' %(u[ii, Nt]))
