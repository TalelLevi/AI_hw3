import pandas as pd
import numpy as np
from ML_algorithms import ID3
from ML_algorithms import KNN
from ML_algorithms import ID3 as ID3ep
from ML_algorithms import KnnEpsilon


def calc_accuracy(df, classifer, test=True):
    count = 0
    for i in range(len(df)):
        original_diagnosis = df.loc[i, 'diagnosis']
        example = df.loc[i, :]
        our_answer = round(classifer.predict(example))
        if our_answer != original_diagnosis:
            count += 1
    text = 'test' if test is True else 'train'
    print(f'{text} success rate is {(1 - count / len(df2)) * 100}')


algos_param = [(ID3, False, [3, 9, 27]), (ID3ep, True, [9]), (KNN, False, [1, 3, 9, 27]), (KnnEpsilon, True, [9])]
df = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')


for algo, epsilon, params in algos_param:
    for param in params:
        if algo is not KNN:
            print(f'now running {algo} with min examples in leaves {param}')
            classier = algo(epsilon=epsilon)
            classier.train(df, min_examples=param)
        elif algo is KNN:
            print(f'now running {algo} with {param} neighbors')
            classier = algo(param)
            classier.train(df)
        # calc_accuracy(df, classier, test=False)
        calc_accuracy(df2, classier)
