import pandas as pd

from ID3 import ID3



df = pd.read_csv('test.csv')

classifer = ID3(0)
classifer.train(df)


for i in range(len(df)):
    original_diagnosis = df.loc[i, 'diagnosis']
    example = df.drop('diagnosis', axis=1).loc[i, :]
    our_answer = classifer.classify(example)
    print(f'{our_answer} == {original_diagnosis} count is {i}')
    assert(original_diagnosis == our_answer)






