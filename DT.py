import pandas as pd

from ID3 import ID3



df = pd.read_csv('test.csv')

classifer = ID3(0)
classifer.train(df)




