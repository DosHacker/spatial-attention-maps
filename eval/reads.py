import numpy as np
import matplotlib.pyplot as plt

data=np.load('20220615T094230481470-pushing_4-small_empty-base.npy',allow_pickle=True)
lastdate=[d[-1] for d in data]

print(data)
