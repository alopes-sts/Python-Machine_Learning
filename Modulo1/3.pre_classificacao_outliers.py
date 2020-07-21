import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 50)
arquivo = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/traffic-collision-data-from-2010-to-present.csv')

# pontos fora da curva precisam ser analisados e caso necessário removidos do dataset
arquivo.boxplot(column='Census Tracts')
plt.show()