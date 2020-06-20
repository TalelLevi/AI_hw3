import pandas as pd

from ID3 import ID3



df = pd.read_csv('train.csv')

classifer = ID3(0)
classifer.train(df, 27)

df2 = pd.read_csv('test.csv')

for i in range(len(df)):
    original_diagnosis = df.loc[i, 'diagnosis']
    example = df.loc[i, :]
    our_answer = classifer.classify(example)
    # print(f'{our_answer} == {original_diagnosis} count is {i}')
    # assert(original_diagnosis == our_answer)

count = 0
for i in range(len(df2)):
    original_diagnosis = df2.loc[i, 'diagnosis']
    example = df2.loc[i, :]
    our_answer = round(classifer.classify(example))
    if our_answer != original_diagnosis:
        count += 1
        print(f'{our_answer} == {original_diagnosis} count is {i}')


print(f'success rate is {(1-count/len(df2))*100}')



