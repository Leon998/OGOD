import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import palettable  #python颜色库


def read_file(path):
    seq = []
    for line in open(path):
        line=line.strip('\n')
        seq.append(float(line))
    seq = np.array(seq).reshape(-1, 1)
    return seq

object_list = os.listdir('../evals')
object_list.sort()
file_list = []
for object in object_list:
    file_list.append(str('../evals/' + object + '/SP.txt'))

matrix = np.zeros(shape=(75,1))
for file in file_list:
    seq = read_file(file)
    matrix = np.column_stack((matrix,seq))
matrix = np.delete(matrix, 0, axis=1)


df = pd.DataFrame(matrix,
                  index=[str(i) for i in range(0, 75)],#DataFrame的行标签设置为大写字母
                  columns=object_list)#设置DataFrame的列标签
plt.figure(dpi=200, figsize=(10,8))
sns.heatmap(data=df,
            # cmap=sns.diverging_palette(10, 220, sep=80, n=10),#区分度显著色盘：sns.diverging_palette()使用
            annot=True, fmt=".2f", annot_kws={'size':5,'weight':'normal'},
           )
plt.title("Sequence Evaluation (score by division, step 1)")
plt.show()