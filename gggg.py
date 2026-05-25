from sklearn.preprocessing import OneHotEncoder
import numpy as np
d=np.array([["red"],["blue"],["green"],["red"]])
e=OneHotEncoder(sparse_output=True)
e=e.fit_transform(d)
print(e)