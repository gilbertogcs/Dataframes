#!/usr/bin/env python
# coding: utf-8

# # Dataset Titanic

# pagina do curso
# https://hotmart.com/pt-br/club/datascience/products/1228268

# In[63]:


import pandas as pd
import numpy as np
import requests
import warnings
import os
import seaborn as sns
import matplotlib.pyplot as plt
import psycopg2
from psycopg2 import Error
from sqlalchemy import create_engine


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


pd.set_option('display.max_columns', None)


# # Carregando dados

# In[4]:


df_Titanic = pd.read_csv('train.csv')


# In[5]:


df_Titanic.shape


# In[6]:


display(df_Titanic)


# # Verificando dados nulos

# In[70]:


df_Titanic.isnull().sum()


# ## Verificando os tipos de dados para segmentar

# In[8]:


df_Titanic.dtypes


# ## Segmentando o Dataframe

# In[9]:


quali = []
quanti = []
for i in df_Titanic.dtypes.index:
    if df_Titanic.dtypes[i] == 'object':
        quali.append(i)
    else:
        quanti.append(i)   


# In[10]:


print('Lista quali', quali)
print('Lista quanti', quanti)


# # Criando o Dataframe Quantitativo

# Variáveis qualitativas são aquelas que expressam uma qualidade ou característica categórica ou nominal, como por exemplo cor dos olhos, estado civil ou tipo de produto

# In[11]:


df_Titanic_quanti = df_Titanic[quanti]


# In[12]:


df_Titanic_quanti.head()


# In[13]:


df_Titanic_quanti.describe()


# In[14]:


df_Titanic_quanti.median()


# # Criando o Dataframe Qualitativo

# Variáveis quantitativas são aquelas que expressam uma quantidade numérica ou mensurável, como por exemplo idade, peso, altura ou temperatura. Elas podem ser contínuas, quando assumem valores em uma escala numérica contínua, ou discretas, quando assumem valores apenas em um conjunto finito ou enumerável de possibilidades.

# In[15]:


df_Titanic_quali = df_Titanic[quali]


# In[16]:


#criando a tabela de frequencia 
df_Titanic_quali.groupby('Sex').Name.count()


# In[17]:


df_Titanic_quali.head()


# In[18]:


#fazendo a tabela de frequencia de todas as variaveis de uma só vez

for i in df_Titanic_quali.columns:
    if i == "Name":
        pass
    else:
        print(40 * '-')
        print('Variavel: ', i)
        print(df_Titanic_quali.groupby(i).Name.count())
        print(30 * '-')


# # Trabalhando com dados nulos

# In[19]:


#criando um dataframe vazio
#nesse dataframe ele vai conter todas as colunas do dataframe df_Titanic
nulos = pd.DataFrame()
nulos['Variavel'] = df_Titanic.columns


# In[20]:


#o objetivo é criar esse dataframe para saber quantos dados nulos existem para cada coluna do df_Titanic
nulos


# In[21]:


#adicionando outra coluna para mensurar a qtde de dados nulos no df_Titanic

nulos['Quantidade']= pd.Series()
nulos['Porcentagem']= pd.Series()

for i in nulos.index:
    nulos.Quantidade[i] = df_Titanic[nulos['Variavel'][i]].isna().sum()
    nulos.Porcentagem[i] = round(((df_Titanic[nulos['Variavel'][i]]).isna().sum()/df_Titanic.PassengerId.count()*100),3)


# In[22]:


nulos


# # Estratégias de analise

# * Existe muitas informações com dados nulos no Df
# * A Coluna Cabin contem 77% dos dados nulos não e viavel excluir os dados, precisa adotar um plano para o tratamento dos dados
# * A coluna Embarked existe 0,02% dos dados nulos, pode ser avaliado para excluir
# * A coluna Age contem 19,8% dos dados, também deve ser tratada
# 
# 
# -> A Coluna age é uma variavel quantitativa pode ser preenchida através da média de idade dos passageiros
# 
# -> A Coluna Cabin é uma variavel qualitativa pode ser preenchida pelo metodo modal

# In[23]:


#apagando dados nulos da coluna Embarked
df_Titanic_2 = df_Titanic.dropna(subset=['Embarked'])


# In[24]:


df_Titanic_2


# In[25]:


#preenchendo a variavel Age
df_Titanic_3 = df_Titanic_2.copy()
df_Titanic_3.Age = df_Titanic_2.Age.fillna(df_Titanic_2.Age.mean())


# In[26]:


#Não existe mais dados vazios na coluna Age
df_Titanic_3.Age.isna().sum()


# # Preenchendo os dados com o modal

# In[27]:


#Copiando o Df3 para um novo df4
#utilizando o mode para saber quais valores se repetem
df_Titanic_4 = df_Titanic_3.copy()
print(df_Titanic_4.Cabin.mode())


# In[28]:


moda= []
for i in df_Titanic_4.Cabin.mode().values:
    moda.append(i)


# In[29]:


moda


# In[30]:


#preenchendo de forma aleatória as cabines
import random
df_Titanic_4.Cabin = df_Titanic_4.Cabin.fillna(random.choice(moda))


# In[31]:


df_Titanic_4.Cabin.isna().sum()


# In[32]:


df_Titanic_4


# # Introdução ao tratamento de Outliers

# In[33]:


#plotando varios graficos na mesma figura
fig, axs = plt.subplots(2,3,figsize=(20,10))

axs[0,0].set_title('Survived')
axs[0,0].boxplot(df_Titanic_4.Survived)

axs[0,1].set_title('Pclass')
axs[0,1].boxplot(df_Titanic_4.Pclass)

axs[0,2].set_title('Age')
axs[0,2].boxplot(df_Titanic_4.Age)

axs[1,0].set_title('SibSp')
axs[1,0].boxplot(df_Titanic_4.SibSp)

axs[1,1].set_title('Parch')
axs[1,1].boxplot(df_Titanic_4.Parch)

axs[1,2].set_title('Fare')
axs[1,2].boxplot(df_Titanic_4.Fare)


# In[34]:


fig


# In[35]:


# Calculo dos Outliers

# Todos os pontos que estão fora do limite superior e inferior da amostra

# Limite superior = Q3 + 1,5 * DistanciaInterquartil
# Limite inferior = Q1 - 1,5 * DistanciaInterquartil

# Distância Interquartil = Valor do 3º Quartil - Valor do 1º Quartil (Q3 - Q1)


# Fazendo este cálculo para estas o Dataframe todo (apenas o quantitativo)

df_Titanic_4_quanti = df_Titanic[quanti]
colunas = df_Titanic_4_quanti.columns
outliers = []

for i in df_Titanic_4_quanti.columns:
    
    q3 = np.quantile(df_Titanic_4_quanti[i], 0.75)
    q1 = np.quantile(df_Titanic_4_quanti[i], 0.25)
    dist = q3 - q1 
    lim_inf = q1 - 1.5*dist
    lim_sup = q3 + 1.5*dist
    
    print(50 * '=')
    print('Variavel/Coluna: ',i)
    print('')
    print('Distancia entre Quartis: ',dist)
    print('Limite Inferior: ',lim_inf)
    print('Limite Superior: ',lim_sup)
    print(40 * '=')
    print('')

    outlier = 0

    for j in df_Titanic_4_quanti.index:
        if df_Titanic_4_quanti[i][j] < lim_inf:
            outlier = outlier + 1
        elif df_Titanic_4_quanti[i][j] > lim_sup:
            outlier= outlier + 1
        else: 
            pass
        
    outliers.append(outlier)
    
df_outlier = pd.DataFrame()
df_outlier['Variável'] = colunas
df_outlier['Outliers'] = outliers
df_outlier['Porcentagem'] = (outliers/df_Titanic_4_quanti.PassengerId.count()) * 100


# In[36]:


df_outlier


# Sobre os outliers
# * Cabe analisar e verificar se é preciso saber se os outliers do DF será incluido na analise
# 
# * Caso não seja a opção é segmentar o DF para analise do Df

# In[62]:


df_Titanic_4.isnull().sum()


# # Enviando o DataFrame para o Banco Postgres

# In[66]:


#conexao com o BD postgres
engine = create_engine('postgresql://postgres:admin362_python2023@localhost:5432/treinamento_py')


# In[67]:


try:
    envia_df = df_Titanic_4.to_sql('base_titanic', engine, if_exists="append", index = False)
except Exception as err:
    print(err)


# In[71]:


try:
    query = """ select * from base_titanic """
    df_Titanic_sql= pd.read_sql_query(query, engine)    
except Exception as err:
    print(err)


# In[72]:


df_Titanic_sql


# In[ ]:




