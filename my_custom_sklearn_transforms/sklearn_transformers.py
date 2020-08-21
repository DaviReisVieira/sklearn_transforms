from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
    
class MediaNotas(BaseEstimator, TransformerMixin):
    '''Rebece uma string para ser o nome da coluna de medias'''
    def __init__(self, nome_da_coluna):
        self.nome = nome_da_coluna

    def fit(self, X, y=None):
        return self

    def media_notas(self, x):
        notas = []
        notas.append(x['NOTA_DE']) if x['NOTA_DE'] != np.nan else notas.append(0)
        notas.append(x['NOTA_EM']) if x['NOTA_EM'] != np.nan else notas.append(0)
        notas.append(x['NOTA_MF']) if x['NOTA_MF'] != np.nan else notas.append(0)
        notas.append(x['NOTA_GO']) if x['NOTA_GO'] != np.nan else notas.append(0)
        media = np.sum(notas)/4
        return pd.Series(data=[media], index=[self.nome])

    def transform(self, X):
        data = X.copy()

        coluna_media = data.apply(self.media_notas, axis=1)
        data_c_media = data.join(coluna_media)
        return data_c_media
    
class ColumnsTransformerGOINGLES(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        cl = ColumnTransformer([
        ('go_median', SimpleImputer(missing_values=np.nan,strategy='median'), ['NOTA_GO']),
        ('ingles_contant', SimpleImputer(missing_values=np.nan,strategy='most_frequent'), ['INGLES'])
                          
        ],remainder='drop')
        data['NOTA_GO'] = pd.DataFrame(cl.fit_transform(data))[0]
        data['INGLES'] = pd.DataFrame(cl.fit_transform(data))[1]
        return data
