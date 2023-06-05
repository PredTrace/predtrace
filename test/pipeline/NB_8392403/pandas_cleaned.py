import pandas as pd
import pickle

import time
start = time.time()
df0=pickle.load(open('data_0.pickle','rb'))
df6 = df0[df0["date"].isnull() == False]
df7=pickle.load(open('data_7.pickle','rb'))
df13 = df7[df7["date"].isnull() == False]
df14=pickle.load(open('data_14.pickle','rb'))
df20 = df14[df14["date"].isnull() == False]
df21=pickle.load(open('data_21.pickle','rb'))
df27 = df21[df21["date"].isnull() == False]
df28 = df6.append(df13)
df29 = df28.append(df20)
df30 = df29.append(df27)
df31 = df30.sort_values(["date"])
df34 = df31.groupby(by=["date"])["date","SearchFrequency"].sum()

end = time.time()

print(end-start)
