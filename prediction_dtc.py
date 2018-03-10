import pandas as pd
import numpy as np
import pickle

sp_l = float(input("Enter the sepal Length:\t"))
sp_w = float(input('Enter the sepal Width:\t'))
pt_l = float(input('Enter the petal Length:\t'))
pt_w = float(input('Enter the petal Width:\t'))

loaded_decisionTreeClassifierModel = pickle.load(open('dtc_model.sav', 'rb'))
test = np.array([sp_l,sp_w,pt_l,pt_w])
test = test.reshape(1, -1)
result = loaded_decisionTreeClassifierModel.predict(test)
print(result)