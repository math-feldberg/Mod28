import streamlit as st
from streamlit_option_menu import option_menu
#from streamlit_pandas_profiling import st_profile_report

import pandas as pd
from ydata_profiling import ProfileReport

import numpy as np

import matplotlib
matplotlib.use('agg')
from tkinter import *
from mttkinter import *

import seaborn as sns

from matplotlib import pyplot
import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn import datasets
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from scipy.stats import ks_2samp
import statsmodels.formula.api as smf
import statsmodels.api as sm

from datetime import datetime

import plotly.figure_factory as ff

from PIL import Image
from io import BytesIO

st.set_page_config(page_title='Projeto 2 - Cientista de Dados EBAC', page_icon='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRQlWkTPwbN4sIWlFBLTlDUnHXFNHRGqslRP6cTgzcOUdfFWxY_kS2rok5rUwCRbe3Hg-0&usqp=CAU', layout="wide", initial_sidebar_state="auto", menu_items=None)

with st.sidebar:
    
    image = Image.open('Por-Que-e-Como-Data-Science-e-Mais-do-Que-Apenas-Machine-Learning.jpg')
    st.sidebar.image(image)
    st.markdown("---")
    st.title('Projeto 2 - Cientista de Dados - EBAC')
    st.subheader('Aluno: Matheus Feldberg')
    st.markdown('[LinkedIn](https://www.linkedin.com/in/matheus-feldberg-521a93259)')
    st.markdown('[GitHub](https://github.com/math-feldberg/Mod16Ex01)')
    st.markdown("---")
         
    renda = pd.read_csv('./previsao_de_renda.csv')
    renda.drop(['Unnamed: 0', 'id_cliente'], axis=1)
    renda['posse_de_veiculo'] = renda['posse_de_veiculo'].map({True: 1,False: 0})
    renda['posse_de_imovel'] = renda['posse_de_imovel'].map({True: 1,False: 0})
    renda['data_ref'] = pd.to_datetime(renda['data_ref'])
    renda['data_ref'] = renda['data_ref'].dt.strftime('%m-%Y')   
    renda.dropna(inplace=True) 
        
    @st.cache_data

    def convert_df(renda):
                                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return renda.to_csv().encode('utf-8') 
    
    csv = convert_df(renda)

    st.download_button(
                                label="📥 Download do Dataframe em CSV",
                                data=csv,
                                file_name='dataframe.csv',
                                mime='text/csv')
                                    
    @st.cache_data

    def to_excel(renda):
                                output = BytesIO()
                                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                                renda.to_excel(writer, index=False, sheet_name='Sheet1')
                                writer.save()
                                processed_data = output.getvalue()
                                return processed_data

    renda_xlsx = to_excel(renda)
                                
    st.download_button(
                                label='📥 Download do Dataframe em EXCEL',
                                data=renda_xlsx,
                                file_name= 'dataframe.xlsx')   
        
   

with st.sidebar:
    
    selected = option_menu(
                        menu_title = 'Sumário',
                        options =  ['1. Etapa 1 CRISP - DM: Entendimento do negócio',
                        '2. Etapa 2 Crisp-DM: Entendimento dos dados',
                        '3. Etapa 3 Crisp-DM: Preparação dos dados',
                        '4. Etapa 4 Crisp-DM: Modelagem',
                        '5. Etapa 5 Crisp-DM: Avaliação dos resultados',
                        '6. Etapa 6 Crisp-DM: Implantação'],
                        default_index=0)
    
if selected == '1. Etapa 1 CRISP - DM: Entendimento do negócio':
        
    st.subheader('Etapa 1 CRISP - DM: Entendimento do negócio')
                
    st.markdown('''Como primeira etapa do CRISP-DM, vamos entender do que se trata o negócio, e quais os objetivos. 
Essa é uma base de proponentes de cartão de crédito, nosso objetivo é construir um modelo preditivo para 
90 em um horizonte de 12 meses) através de variáveis que podem ser observadas na data da avaliação do crédito 
(tipicamente quando o cliente solicita o cartão).
                            
Atividades do CRISP-DM:

- Objetivos do negócio:
Note que o objetivo aqui é que o modelo sirva o mutuário (o cliente) para que avalie suas próprias decisões, 
e não a instituição de crédito.
- Objetivos da modelagem:
O objetivo está bem definido: desenvolver o melhor modelo preditivo de modo a auxiliar o mutuário a tomar suas 
próprias decisões referentes a crédito.

Nessa etapa também se avalia a situação da empresa/segmento/assunto de modo a se entender o tamanho do público, 
relevância, problemas presentes e todos os detalhes do processo gerador do fenômeno em questão, e portanto dos dados.
Também é nessa etapa que se constrói um planejamento do projeto.''')
                        
elif selected == '2. Etapa 2 Crisp-DM: Entendimento dos dados':
    
    st.header('Etapa 2 Crisp-DM: Entendimento dos dados')

    st.markdown('A segunda etapa é o entendimento dos dados. Foram fornecidas 13 variáveis mais a variável resposta (em negrito na tabela). O significado de cada uma dessas variáveis se encontra na tabela.')
    st.subheader('Dicionário de dados')
    st.markdown('''Os dados estão dispostos em uma tabela com uma linha para cada cliente, e uma coluna para cada variável armazenando as características desses clientes. Colocamos uma cópia o dicionário de dados (explicação dessas variáveis) abaixo neste notebook:


| Variável                | Descrição                                           | Tipo         |
| ----------------------- |:---------------------------------------------------:| ------------:|
| data_ref                |  Data de referência                                 | data         |
| id_cliente              |  Identificação do cliente                           | inteiro      |
| sexo                    |  M = 'Masculino'; F = 'Feminino'                    | M/F          |
| posse_de_veiculo        |  True = 'possui'; False = 'não possui'              | True/False   |
| posse_de_imovel         |  True = 'possui'; False = 'não possui'              | True/False   |
| qtd_filhos              |  Quantidade de filhos                               | inteiro      |
| tipo_renda              |  Tipo de renda (ex: assaliariado, autônomo etc)     | texto        |
| educacao                |  Nível de educação (ex: secundário, superior etc)   | texto        |
| estado_civil            |  Estado civil (ex: solteiro, casado etc)            | texto        |
| tipo_residencia         |  Tipo de residência (ex: casa/apto, com os pais etc)| texto        |
| idade                   |  Idade em anos                                      | inteiro      |
| tempo_emprego           |  Tempo de emprego em anos                           | inteiro      |
| qt_pessoas_residencia   |  Quantidade de pessoas na residência                | inteiro      |
| **renda**               |  Renda do cliente em Reais                          | inteiro      |''')

    st.subheader('Carregando os pacotes')   

    st.markdown('É considerada uma boa prática carregar os pacotes que serão utilizados como a primeira coisa do programa.')

    st.markdown('Usaremos o seguinte código:')
                                            
    code = '''import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report

import pandas as pd
from pandas_profiling import ProfileReport

import numpy as np

import matplotlib
matplotlib.use('agg')
from tkinter import *
from mttkinter import *

import seaborn as sns

from matplotlib import pyplot
import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn import datasets
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from scipy.stats import ks_2samp
import statsmodels.formula.api as smf
import statsmodels.api as sm

from datetime import datetime

import plotly.figure_factory as ff

from PIL import Image
from io import BytesIO'''

    st.code(code, language='python')

    st.subheader('Carregando os dados') 
    st.markdown('O comando pd.read_csv é um comando da biblioteca pandas (pd.) e carrega os dados do arquivo csv indicado para um objeto *dataframe* do pandas.')

    renda = pd.read_csv('./previsao_de_renda.csv')
    renda.head(1)

    st.dataframe(renda)

    st.markdown('Ao lado você pode baixar o dataframe em formato CSV e EXCEL.')
                            
    st.subheader('Entendimento dos dados - Univariada')
    st.markdown('Nesta etapa tipicamente avaliamos a distribuição de todas as variáveis.')

    st.markdown('Vamos utilziar Pandas Profiling para exploração dos dados e para garantir a qualidade dos dados.')
    
    from streamlit_pandas_profiling import st_profile_report
      
    prof = renda.profile_report()
    st_profile_report(prof)
                                     
    st.download_button(
               label='📥 Download do ProfileReport', 
               data=prof.to_html(), 
               file_name='analise_renda.html')
    st.markdown("---")
           
    st.markdown('Distribuição das variáveis qualitativas no tempo (sexo, posse_de_veiculo, posse_de_imovel, tipo_renda, educacao, estado_civil e tipo_residencial):')
    
    renda.drop(['Unnamed: 0', 'id_cliente'], axis=1)
    renda['posse_de_veiculo'] = renda['posse_de_veiculo'].map({True: 1,False: 0})
    renda['posse_de_imovel'] = renda['posse_de_imovel'].map({True: 1,False: 0})
    renda['data_ref'] = pd.to_datetime(renda['data_ref'])
    renda['data_ref'] = renda['data_ref'].dt.strftime('%m-%Y')
    renda.head()

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.countplot(x= renda['data_ref'],  hue = renda['sexo'], data=renda)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.xticks(rotation=45);
    st.pyplot()

    sns.countplot(x= renda['data_ref'],  hue = renda['posse_de_veiculo'], data=renda)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(rotation=45);
    st.pyplot()

    sns.countplot(x= renda['data_ref'],  hue = renda['posse_de_imovel'], data=renda)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(rotation=45);
    st.pyplot()
                                            
    sns.countplot(x= renda['data_ref'],  hue = renda['tipo_renda'], data=renda)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks( rotation=45);
    st.pyplot()

    sns.countplot(x= renda['data_ref'],  hue = renda['educacao'], data=renda)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(rotation=45);
    st.pyplot()
                                            
    sns.countplot(x= renda['data_ref'],  hue = renda['estado_civil'], data=renda)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(rotation=45);
    st.pyplot()

    sns.countplot(x= renda['data_ref'],  hue = renda['tipo_residencia'], data=renda)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(rotation=45);
    st.pyplot()
                                                    
    st.header('Entendimento dos dados - Bivariadas')

    st.markdown('Entender a relação da renda representada pela variável resposta (```renda```) e as demais variáveis explicativas (demais). Para isto, vamos avaliar a capacidade econômica para diferentes grupos definidos pelas variáveis explicativas para obter um modelo preditor de renda.')

    st.markdown('Plotando uma matriz de correção:')
    st.dataframe(renda)
    data_corr = renda.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_corr, annot=True, fmt='.1g', vmin=-1, vmax=1, cmap='mako', center=0)
    plt.xticks(rotation = 40)
    st.pyplot()   

elif selected == '3. Etapa 3 Crisp-DM: Preparação dos dados':
                            
    st.header('Etapa 3 Crisp-DM: Preparação dos dados')

    st.subheader('Nessa etapa realizamos tipicamente as seguintes operações com os dados:')

    st.markdown(''' - **seleção**: Já temos os dados selecionados adequadamente?
    - **limpeza**: Precisaremos identificar e tratar dados faltantes
    - **construção**: construção de novas variáveis
    - **integração**: Temos apenas uma fonte de dados, não é necessário integração
    - **formatação**: Os dados já se encontram em formatos úteis?
                                    ''')

    st.markdown('''Em primeirio lugar, vamos realizar a seleção e limpeza dos dados:
    - a variável 'Unnamed: 0' apresenta valores únicos, logo, é irrelevante para o projeto.
    - a variável 'id_cliente' também não tem valor para a análise pretendida porque não interfere em nenhuma outra variável.
    - a variável 'tempo_emprego' tem 17.2% de valores missing.
    - a avariável 'data_ref' será adequada para mês e ano.''')

    st.markdown('As colunas Unnamed: 0 e id_cliente foram excluídas no início quando plotamos os gráficos de distribuição das variáveis qualitativas pelo tempo.')

    st.markdown('Identificando e tratando os dados missing:')
    
    renda = pd.read_csv('./previsao_de_renda.csv')
    renda.head(1)

    renda.drop(['Unnamed: 0', 'id_cliente'], axis=1)
    renda['posse_de_veiculo'] = renda['posse_de_veiculo'].map({True: 1,False: 0})
    renda['posse_de_imovel'] = renda['posse_de_imovel'].map({True: 1,False: 0})
    renda['data_ref'] = pd.to_datetime(renda['data_ref'])
    renda['data_ref'] = renda['data_ref'].dt.strftime('%m-%Y')
    renda.head()
        
    st.dataframe(renda.isna().sum())

    renda.dropna(inplace=True)
    renda.isna().sum()
    st.markdown('A coluna data_ref também foi corrigida no iníico para exibir apenas mês/ano::')
    st.dataframe(renda)
                        
elif selected == '4. Etapa 4 Crisp-DM: Modelagem':
                        
                st.header('Etapa 4 Crisp-DM: Modelagem')

                st.markdown('''Nessa etapa que realizaremos a construção do modelo. Os passos típicos são:
                                - Selecionar a técnica de modelagem
                                - Desenho do teste
                                - Avaliação do modelo''')
                
                renda = pd.read_csv('./previsao_de_renda.csv')
                renda.head(1)

                renda.drop(['Unnamed: 0', 'id_cliente'], axis=1)
                renda['posse_de_veiculo'] = renda['posse_de_veiculo'].map({True: 1,False: 0})
                renda['posse_de_imovel'] = renda['posse_de_imovel'].map({True: 1,False: 0})
                renda['data_ref'] = pd.to_datetime(renda['data_ref'])
                renda['data_ref'] = renda['data_ref'].dt.strftime('%m-%Y')
                renda.head()
        
                renda.dropna(inplace=True)
                X = pd.get_dummies(renda.drop(['Unnamed: 0', 'id_cliente', 'data_ref'], axis=1))
                y = renda['renda']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = 100)
                print(X_train.shape)
                print(X_test.shape)
                print(y_train.shape)
                print(y_test.shape)

                regr_1 = DecisionTreeRegressor(max_depth=4)
                regr_2 = DecisionTreeRegressor(max_depth=3)
                regr_3 = DecisionTreeRegressor(max_depth=2)

                regr_1.fit(X_train, y_train)
                regr_2.fit(X_train, y_train)
                regr_3.fit(X_train, y_train)

                plt.rc('figure', figsize=(40, 10))
                tp = tree.plot_tree(regr_1, 
                feature_names=X.columns,  
                filled=True)
                st.pyplot()
                plt.rc('figure', figsize=(10, 5))
                tp = tree.plot_tree(regr_2, 
                feature_names=X.columns,  
                filled=True) 
                st.pyplot()
                plt.rc('figure', figsize=(10, 5))
                tp = tree.plot_tree(regr_3, 
                feature_names=X.columns,  
                filled=True) 
                st.pyplot()
                mse1 = regr_1.score(X_train, y_train)
                mse2 = regr_2.score(X_train, y_train)
                mse2 = regr_3.score(X_train, y_train)


                template = "O MSE da árvore de treino com profundidade={0} é: {1:.2f}"

                st.write(template.format(regr_1.get_depth(),mse1).replace(".",","))
                st.write(template.format(regr_2.get_depth(),mse2).replace(".",","))
                st.write(template.format(regr_3.get_depth(),mse2).replace(".",","))

elif selected == '5. Etapa 5 Crisp-DM: Avaliação dos resultados':

    st.header('Etapa 5 Crisp-DM: Avaliação dos resultados')

    st.markdown('O modelo de árvore de regressão com profundidade 4 apresentou um MSE de 0.99 tanto para os dados de treino quanto para os dados de teste e, portanto, é o melhor para previsão dos dados de renda.')
                
    renda = pd.read_csv('./previsao_de_renda.csv')
    renda.head(1)

    renda.drop(['Unnamed: 0', 'id_cliente'], axis=1)
    renda['posse_de_veiculo'] = renda['posse_de_veiculo'].map({True: 1,False: 0})
    renda['posse_de_imovel'] = renda['posse_de_imovel'].map({True: 1,False: 0})
    renda['data_ref'] = pd.to_datetime(renda['data_ref'])
    renda['data_ref'] = renda['data_ref'].dt.strftime('%m-%Y')
    renda.head()
            
    renda.dropna(inplace=True)
    X = pd.get_dummies(renda.drop(['Unnamed: 0', 'id_cliente', 'data_ref'], axis=1))
    y = renda['renda']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = 100)                           
                
    regr_1 = DecisionTreeRegressor(max_depth=4)
    regr_2 = DecisionTreeRegressor(max_depth=3)
    regr_3 = DecisionTreeRegressor(max_depth=2)

    regr_1.fit(X_test, y_test)
    regr_2.fit(X_test, y_test)
    regr_3.fit(X_test, y_test)

    plt.rc('figure', figsize=(40, 10))
    tp = tree.plot_tree(regr_1, 
    feature_names=X.columns,  
    filled=True)
    st.pyplot()

    plt.rc('figure', figsize=(10, 5))
    tp = tree.plot_tree(regr_3, 
    feature_names=X.columns,  
    filled=True) 
    st.pyplot()

    plt.rc('figure', figsize=(10, 5))
    tp = tree.plot_tree(regr_3, 
    feature_names=X.columns,  
    filled=True) 
    st.pyplot()

    mse1 = regr_1.score(X_test, y_test)
    mse2 = regr_2.score(X_test, y_test)
    mse2 = regr_3.score(X_test, y_test)
    
    template = "O MSE da árvore de teste com profundidade={0} é: {1:.2f}"

    st.write(template.format(regr_1.get_depth(),mse1).replace(".",","))
    st.write(template.format(regr_2.get_depth(),mse2).replace(".",","))
    st.write(template.format(regr_3.get_depth(),mse2).replace(".",","))
                    
elif selected == '6. Etapa 6 Crisp-DM: Implantação':
                        
    st.header('Etapa 6 Crisp-DM: Implantação')

    st.markdown('Nessa etapa colocamos em uso o modelo desenvolvido, normalmente implementando o modelo desenvolvido em um motor que toma as decisões com algum nível de automação.')
