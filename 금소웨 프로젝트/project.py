# -*- coding: utf-8 -*-

import pandas as pd
df = pd.read_csv("D:/005930.KS.csv")
df2 = df[['Date', 'Adj Close']]
df3 = df2[['Adj Close']]
price = df3.values
p = []
i = 0
while i < len(price):
    p.append(price[i][0])
    i = i + 1

b = int(input("구간 길이를 입력해주세요. (2 ~ %d 사이): " % (len(p) - 1)))

value = []
t = 0
while t < len(p) - b + 1:
    pp = []
    u = 0
    while u < b:
        pp.append(p[t+u])
        u = u + 1
    value.append(pp)
    t = t + 1

recent = value[-1]
    
rvalue = []
i = 0
while i < len(value):
    temp = []
    j = 0
    while j < b:
        temp.append(value[i][j]/value[i][0])
        j = j + 1
    rvalue.append(temp)
    i = i + 1

recentr = rvalue[-1]

deviation = []
v = 0
while v < len(rvalue) - 1:
    dev = 0
    w = 0
    while w < b:
        sq = rvalue[v][w] - recentr[w]
        dev = dev + (sq ** 2)
        w = w + 1
    deviation.append((dev)**(1/2))
    v = v + 1

c = int(input("이후 구간 길이를 입력해주세요. (1 ~ %d 사이): " % (len(p) - b)))

k = 0
i = 0
while i < len(p) - b - c + 1:
    if deviation[k] > deviation[i + 1]:
        k = i + 1
    i = i + 1

ppp = []
ppp.append(value[k][-1])
i = 0
while i < c:
    ppp.append(p[k+b+i])
    i = i + 1

ppr = []
i = 0
while i < len(ppp) - 1:
    ppr.append(ppp[i+1]/ppp[i] - 1)
    i = i + 1

i = 0
while i < len(ppr):
    p.append(p[-1]*(1+ppr[i]))
    i = i + 1
pf = p[-c:]

print(pf)

import matplotlib.pyplot as plt

fig = plt.figure()
fig, ax = plt.subplots(3, 1)

ax[0].plot(p)

pppd = []
pppd.append(k)
i = 0
while i < b-1:
    pppd.append(pppd[-1]+1)
    i = i + 1
ax[0].plot(pppd, value[k], color = 'red')

ppl = []
ppl.append(len(p)-b-c)
i = 0
while i < b-1:
    ppl.append(ppl[-1]+1)
    i = i + 1
ax[0].plot(ppl, recent, color = 'red')

pppda = []
pppda.append(k+b)
i = 0
while i < c-1:
    pppda.append(pppda[-1]+1)
    i = i + 1
ax[0].plot(pppda, ppp[1:], color = 'black')

ppla = []
ppla.append(len(p)-c)
i = 0
while i < c-1:
    ppla.append(ppla[-1]+1)
    i = i + 1
ax[0].plot(ppla, p[-c:], color = 'black')
ax[0].set_title("Time Series")
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Price")

a1 = []
a2 = []
i = 0
while i < b:
    a1.append(value[k][i]/value[k][0])
    a2.append(recent[i]/recent[0])
    i = i + 1

ax[1].plot(a1)
ax[1].plot(a2)

ax[1].set_title("Compare")
ax[1].set_xlabel("Time space")
ax[1].set_ylabel("Ratio")

i = 0
while i < len(value):
    j = 0
    b1 = []
    while j < b:
        b1.append(value[i][j]/value[i][0])
        j = j + 1
    ax[2].plot(b1)
    i = i + 1
ax[2].set_title("Group")
ax[2].set_xlabel("Time space")
ax[2].set_ylabel("Ratio")

plt.subplots_adjust(hspace=1.30)
plt.show()