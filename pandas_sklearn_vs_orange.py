import pandas as pd
import numpy as np

import Orange
import sklearn

orange_data = Orange.data.Table("zoo")
print(orange_data)

print(orange_data.X)
print(orange_data.Y)
print(orange_data.metas)
print(orange_data.domain)

df = pd.DataFrame(orange_data.X)
df['class'] = orange_data.Y
print(df)
