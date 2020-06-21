import pandas as pd
import numpy as np
from ID3 import ID3
from DT_epsilon import ID3 as ID


def calc_accuracy(df, classifer, test=True):
    count = 0
    for i in range(len(df)):
        original_diagnosis = df.loc[i, 'diagnosis']
        example = df.loc[i, :]
        our_answer = round(classifer.classify(example))
        if our_answer != original_diagnosis:
            count += 1
    text = 'test' if test == True else 'train'
    print(f'{text} success rate is {(1 - count / len(df2)) * 100}')


df = pd.read_csv('train.csv')

# classifer = ID3(0)
# classifer.train(df, 27)


stds = [np.std(feature)*0.1 for feature in df.values.T]

classifer = ID(0, epsilon=stds)
classifer.train(df, 2)

df2 = pd.read_csv('test.csv')

calc_accuracy(df, classifer, test=False)
calc_accuracy(df2, classifer)
