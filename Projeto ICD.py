#!/usr/bin/env python
# coding: utf-8

# # Projeto ICD - Óbitos por meningite pneumocócica no Brasil (1996 - 2019)

# **Aluno:** Lucas Giandoni Perez (2017092023)

# ## I. Introdução

# As meningites caracterizam-se pela inflamação, em geral de origem infecciosa, das meninges. Pia-máter, aracnoide e dura-máter são as três meninges que envolvem o Sistema Nervoso Central, fornecendo a este proteção mecânica e de agentes biológicos potencialmente patogênicos. Por meio do trajeto percorrido pelo líquido cefalorraquidiano (LCR), o processo infeccioso consegue se disseminar rapidamente, desenvolvendo uma reação inflamatória intensa. O acometimento de estruturas como medula espinal, córtex cerebral, cisternas da base do crânio, nervos cranianos e sistema ventricular propicia o surgimento dos sinais e sintomas no paciente.
# O quadro clínico da meningite costuma ser formado por três síndromes (conjunto de sinais e sintomas):
# 1) Síndrome da hipertensão intracraniana
# 2) Síndrome Toxêmica
# 3) Síndrome de irritação meníngea, com os sinais patognomônicos como: rigidez de nuca, sinal de Kernig, sinal de Brudzinski
# 
# O surgimento de novas ferramentas diagnósticas e terapêuticas nas últimas décadas alterou drasticamente a história natural da doença. Os pacientes com meningites bacterianas tratados de forma adequada costumam evoluir bem, sem complicações futuras. No entanto, a qualidade da assistência à saúde prestada em diferentes regiões do Brasil não é a mesma. Os condicionantes sociais de saúde são "fatores de vida e trabalho dos indivíduos e de grupos da população" que estão relacionadas com sua situação de saúde. Produção agrícola e de alimentos, educação, ambiente de trabalho, desemprego, acesso à água e ao esgoto, serviços sociais de saúde e habitação são alguns exemplos dessas condições. Assim, diferentes áreas do Brasil apresentam índices de saúde também diferentes.
# Além dos condicionantes sociais de saúde, um dos fatores mais analisados pelos grupos de pesquisa de saúde pública é a idade de óbito. Óbitos infantis (menores do que 1 ano), por exemplo, podem revelar informações muito precisas sobre a qualidade da assistência à saúde, o que também é válido para óbitos de crianças com idade menor ou igual a 14 anos. A informação qualitativa sobre a saúde local decorre de análises que estabelecem correlações entre as idades médias de óbitos e as causas específicas das mortes.
# O presente estudo visa avaliar a diferença entre a idade dos óbitos de crianças com idade menor ou igual a 14 anos causadas por meningites bacterianas causadas pelos agentes *Haemophilus* sp. e *Pneumococos* sp. (classificadas de acordo com o CID-10) nos anos 1996, 2000, 2010 e 2019, com dados provenientes do Sistema de Informações sobre Mortalidade (SIM). 

# ## II. Dados utilizados
# A base de dados analisada neste trabalho foi extraída do Sistema de Informações sobre Mortalidade (SIM), que contém dados de todos os óbitos ocorridos de 1996 até 2019

# In[4]:


get_ipython().system('pip install basedosdados')


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import basedosdados as bd
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats as ss
from IPython.display import HTML
from matplotlib import animation


# In[6]:


def despine(ax=None):
    if ax is None:
        ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


# In[7]:


#importando dados sobre o Sistema de Informações sobre Mortalidade (SIM)
df_sim = bd.read_table(dataset_id='br_ms_sim', 
            table_id='municipio_causa_idade_sexo_raca', billing_project_id='projeto-icd-324204')


# ## III. Exploração e visualização
# ### Visualizando o data frame
# 
# Variáveis:
# ano: categórica sigla_uf: categórica nominal id_municipio: categórica nominal causa_basica: categórica nominal idade: numérica discreta sexo: categórica nominal raca_cor: categórica nominal numero_obitos: numérica discreta

# In[ ]:


df_sim.head()


# In[ ]:


df_sim.count()


# In[ ]:


df_sim.shape


# In[ ]:


dict_obitos_ano = {'1996': 908883,
                  '1997': 903516,
                  '1998': 931895,
                  '1999': 938658,
                  '2000': 946686,
                  '2001': 961492,
                  '2002': 982807,
                  '2003': 1002340,
                  '2004': 1024073,
                  '2005': 1006827,
                  '2006': 1028172,
                  '2007': 1044114,
                  '2008': 1072822,
                  '2009': 1098388,
                  '2010': 1132888,
                  '2011': 1166351,
                  '2012': 1177145,
                  '2013': 1206476,
                  '2014': 1223152,
                  '2015': 1260774,
                  '2016': 1306311,
                  '2017': 1309729,
                  '2018': 1313887,
                  '2019': 1347263}


# ### Óbitos no Brasil de 1996 a 2019

# Para normalizar os dados brutos de óbitos, iremos estimar os óbitos a cada 100 mil habitantes. Para isso, foram importados dados extraídos dos censos IBGE de 2000 e 2010. Também foram importados os dados da população estimada nos anos de 1996 e 2019, os extremos da base de dados do SIM.

# In[ ]:


#População segundo o IBGE
pop_96 = 157070163
pop_00 = 169799170
pop_10 = 190755799
pop_19 = 211800000


# **Gráfico de colunas para visualizar as mortes por ano no Brasil (1996 - 2019)**
# 
# O número bruto de óbitos vem crescendo desde 1996

# In[ ]:


plt.figure(figsize=(10, 6))
plt.bar(dict_obitos_ano.keys(), dict_obitos_ano.values())
plt.xlabel("Ano")
plt.xticks(rotation=90)
plt.ylabel("Número de óbitos")
plt.title("Número de óbitos por ano registrados no Brasil")
plt.show()


# **Gráfico de colunas para visualizar as mortes por ano no Brasil (1996 - 2019)**
# 
# O número de óbitos a cada 100 mil habitantes também vem crescendo desde 1996

# In[ ]:


#Gráfico
plt.figure(figsize=(10, 6))
plt.bar(height = (dict_obitos_ano['1996']/pop_96*100000,
                  dict_obitos_ano['2000']/pop_00*100000,
                  dict_obitos_ano['2010']/pop_10*100000,
                  dict_obitos_ano['2019']/pop_19*100000),
        x = ('1996', '2000', '2010', '2019'))
plt.xlabel("Anos")
plt.xticks(rotation=45)
plt.ylabel("Óbitos a cada 100 mil habitantes")
plt.title("Óbitos a cada 100 mil habitantes de 1996 - 2019 no Brasil")
plt.show()


# ### Óbitos por meningites
# As meningites (não especificadas pelo CID-10) foram responsáveis por aproximadamente 0.13% das mortes totais no Brasil de 1996-2019. Embora seja um valor aparentemente irrisório, as meningites bacterianas constituem a principal causa de morbimortalidade por doença infecciosa do sistema nervoso central. De 1996 até 2019, houve uma redução do número de óbitos totais, o que pode ser justificado por diversos fatores, como: implementação do SUS e de uma assistência de saúde pública e universal de qualidade, introdução da vacina no Calendário Nacional de Vacinação que cobre diversos tipos de meningite, disponibilidade de terapias antimicrobianas mais eficazes, melhores do suporte intensivo pediátrico etc.

# In[ ]:


dict_cid = {'Meningite bacteriana não classificada em outra parte': 'G00',
            'Meningite por Haemophilus':'G00.0',
            'Meningite pneumocócica': 'G00.1',
            'Meningite estreptocócica': 'G00.2',
            'Meningite estafilocócica': 'G00.3',
            'Outras meningites bacterianas': 'G00.8',
            'Meningite bacteriana não especificada': 'G00.9',
            'Meningite em doenças bacterianas classificadas em outra parte':'G01',
            'Meningite em outras doenças infecciosas e parasitárias classificadas em outra parte': 'G02',
            'Meningite em doenças virais classificadas em outra parte': 'G02.0',
            'Meningite em micoses': 'G02.1',
            'Meningite em outras doenças infecciosas e parasitárias classificadas em outra parte': 'G02.8',
            'Meningite não-piogênica': 'G03.0',
            'Meningite crônica': 'G03.1',
            'Meningite recorrente benigna (Mollaret)': 'G03.2',
            'Meningite devido a outras causas especificadas': 'G03.8',
            'Meningite não especificada': 'G03.9'}
#Meningite bacteriana não classificada em outra parte: G00
#Meningite em doenças bacterianas classificadas em outra parte: G01
#Meningite em doenças infecciosas e parasitárias classificadas em outra parte: G02
#Meningite devida a outras causas e a causas não especificadas: G03


# In[ ]:


df_haemo = df_sim[df_sim['causa_basica'] == 'G000']
morte_haemo = df_haemo['numero_obitos'].sum()
df_pneumo = df_sim[df_sim['causa_basica'] == 'G001']
morte_pneumo = df_pneumo['numero_obitos'].sum()
df_estrepto = df_sim[df_sim['causa_basica'] == 'G002']
morte_estrepto = df_estrepto['numero_obitos'].sum()
df_outras = df_sim[df_sim['causa_basica'] == 'G008']
morte_outras = df_outras['numero_obitos'].sum()
df_bact_n_especific = df_sim[df_sim['causa_basica'] == 'G009']
morte_bact_n = df_bact_n_especific['numero_obitos'].sum()
df_bact_outra = df_sim[df_sim['causa_basica'] == 'G01']
morte_bact_outra = df_bact_outra['numero_obitos'].sum()
df_micose = df_sim[df_sim['causa_basica'] == 'G021']
morte_micose = df_micose['numero_obitos'].sum()
df_dip = df_sim[df_sim['causa_basica'] == 'G028']
morte_dip = df_dip['numero_obitos'].sum()
df_nao_piog = df_sim[df_sim['causa_basica'] == 'G03']
morte_nao_piog = df_nao_piog['numero_obitos'].sum()
df_cronica = df_sim[df_sim['causa_basica'] == 'G031']
morte_cronica = df_cronica['numero_obitos'].sum()
df_mollaret = df_sim[df_sim['causa_basica'] == 'G032']
morte_mollaret = df_mollaret['numero_obitos'].sum()
df_outros = df_sim[df_sim['causa_basica'] == 'G038']
morte_outros = df_outros['numero_obitos'].sum()
df_n_especific = df_sim[df_sim['causa_basica'] == 'G039']
morte_n_especific = df_n_especific['numero_obitos'].sum()


# In[ ]:


df_meningite = pd.concat([df_haemo, df_pneumo, df_outras,
                        df_bact_n_especific, df_bact_outra, df_micose,
                        df_dip, df_nao_piog, df_cronica, df_mollaret, df_outros,
                        df_n_especific])


# In[ ]:


df_meningite['raca_cor'].isnull().sum()/df_meningite['raca_cor'].count()*100

df_meningite.isnull().sum()

df_meningite['idade'] = df_meningite['idade'].fillna(df_meningite['idade'].mean())


# In[ ]:


#Modificando o sexo do dataframe df_meningite
#0 = feminino
#1 = masculino
df_meningite['sexo'] = np.where((df_meningite.sexo == '2'), 0, df_meningite.sexo)
df_meningite['sexo'] = np.where((df_meningite.sexo == '1'), 1, df_meningite.sexo)
mulheres_meningite = df_meningite[df_meningite['sexo'] == 'F']['sexo'].count()
homens_meningite = df_meningite[df_meningite['sexo'] == 'M']['sexo'].count()


# In[ ]:


#Mortes por meningite no tempo
meningite_96 = df_meningite.query('ano == 1996')
meningite_97 = df_meningite.query('ano == 1997')
meningite_98 = df_meningite.query('ano == 1998')
meningite_99 = df_meningite.query('ano == 1999')
meningite_00 = df_meningite.query('ano == 2000')
meningite_01 = df_meningite.query('ano == 2001')
meningite_02 = df_meningite.query('ano == 2002')
meningite_03 = df_meningite.query('ano == 2003')
meningite_04 = df_meningite.query('ano == 2004')
meningite_05 = df_meningite.query('ano == 2005')
meningite_06 = df_meningite.query('ano == 2006')
meningite_07 = df_meningite.query('ano == 2007')
meningite_08 = df_meningite.query('ano == 2008')
meningite_09 = df_meningite.query('ano == 2009')
meningite_10 = df_meningite.query('ano == 2010')
meningite_11 = df_meningite.query('ano == 2011')
meningite_12 = df_meningite.query('ano == 2012')
meningite_13 = df_meningite.query('ano == 2013')
meningite_14 = df_meningite.query('ano == 2014')
meningite_15 = df_meningite.query('ano == 2015')
meningite_16 = df_meningite.query('ano == 2016')
meningite_17 = df_meningite.query('ano == 2017')
meningite_18 = df_meningite.query('ano == 2018')
meningite_19 = df_meningite.query('ano == 2019')


# In[ ]:


dict_meningites_ano = {'1996': meningite_96['ano'].count(),
                      '1997': meningite_97['ano'].count(),
                      '1998': meningite_98['ano'].count(),
                      '1999': meningite_99['ano'].count(),
                      '2000': meningite_00['ano'].count(),
                      '2001': meningite_01['ano'].count(),
                      '2002': meningite_02['ano'].count(),
                      '2003': meningite_03['ano'].count(),
                      '2004': meningite_04['ano'].count(),
                      '2005': meningite_05['ano'].count(),
                      '2006': meningite_06['ano'].count(),
                      '2007': meningite_07['ano'].count(),
                      '2008': meningite_08['ano'].count(),
                      '2009': meningite_09['ano'].count(),
                      '2010': meningite_10['ano'].count(),
                      '2011': meningite_11['ano'].count(),
                      '2012': meningite_12['ano'].count(),
                      '2013': meningite_13['ano'].count(),
                      '2014': meningite_14['ano'].count(),
                      '2015': meningite_15['ano'].count(),
                      '2016': meningite_16['ano'].count(),
                      '2017': meningite_17['ano'].count(),
                      '2018': meningite_18['ano'].count(),
                      '2019': meningite_19['ano'].count()}


# #### Número de óbitos por meningites

# In[1]:


#Gráfico
plt.figure(figsize=(10, 6))
plt.bar(x=(dict_meningites_ano.keys()),
        height = (dict_meningites_ano.values()))
plt.xlabel("Anos")
plt.xticks(rotation=45)
plt.ylabel("Número total de óbitos causados por meningite")
plt.title("Número de óbitos causados por meningite (não especificada) de 1996 - 2019 no Brasil")
plt.show()


# In[ ]:


#Meningites/mortes totais
meningite_total_bruto = df_meningite['numero_obitos'].sum()/df_sim['numero_obitos'].count()*100
meningite_total_bruto


# #### Percentual dos óbitos por meningites dos óbitos totais

# In[ ]:


#Meningites/mortes totais no tempo
dict_meningites_ano_total = {'1996': meningite_96['ano'].count()/dict_obitos_ano['1996'],
                            '1997': meningite_97['ano'].count()/dict_obitos_ano['1997'],
                            '1998': meningite_98['ano'].count()/dict_obitos_ano['1998'],
                            '1999': meningite_99['ano'].count()/dict_obitos_ano['1999'],
                            '2000': meningite_00['ano'].count()/dict_obitos_ano['2000'],
                            '2001': meningite_01['ano'].count()/dict_obitos_ano['2001'],
                            '2002': meningite_02['ano'].count()/dict_obitos_ano['2002'],
                            '2003': meningite_03['ano'].count()/dict_obitos_ano['2003'],
                            '2004': meningite_04['ano'].count()/dict_obitos_ano['2004'],
                            '2005': meningite_05['ano'].count()/dict_obitos_ano['2005'],
                            '2006': meningite_06['ano'].count()/dict_obitos_ano['2006'],
                            '2007': meningite_07['ano'].count()/dict_obitos_ano['2007'],
                            '2008': meningite_08['ano'].count()/dict_obitos_ano['2008'],
                            '2009': meningite_09['ano'].count()/dict_obitos_ano['2009'],
                            '2010': meningite_10['ano'].count()/dict_obitos_ano['2010'],
                            '2011': meningite_11['ano'].count()/dict_obitos_ano['2011'],
                            '2012': meningite_12['ano'].count()/dict_obitos_ano['2012'],
                            '2013': meningite_13['ano'].count()/dict_obitos_ano['2013'],
                            '2014': meningite_14['ano'].count()/dict_obitos_ano['2014'],
                            '2015': meningite_15['ano'].count()/dict_obitos_ano['2015'],
                            '2016': meningite_16['ano'].count()/dict_obitos_ano['2016'],
                            '2017': meningite_17['ano'].count()/dict_obitos_ano['2017'],
                            '2018': meningite_18['ano'].count()/dict_obitos_ano['2018'],
                            '2019': meningite_19['ano'].count()/dict_obitos_ano['2019']}

#Gráfico
plt.figure(figsize=(10, 6))
plt.bar(x = (dict_meningites_ano_total.keys()) ,
        height = (dict_meningites_ano_total.values()))
plt.xlabel("Anos")
plt.xticks(rotation=45)
plt.ylabel("Percentual dos óbitos totais")
plt.title("Percentual de óbitos causados por meningite (não especificada) dos óbitos totais de 1996 - 2019 no Brasil")
plt.show()


# Selecionando os anos 1996, 2000, 2010 e 2019 para análise (censos IBGE e estimativa de 2019):
# Óbitos a cada 100 mil habitantes

# In[ ]:


#Gráfico
plt.figure(figsize=(10, 6))
plt.bar(height = (meningite_96['ano'].count()/pop_96*100000,
             meningite_00['ano'].count()/pop_00*100000,
             meningite_10['ano'].count()/pop_10*100000,
             meningite_19['ano'].count()/pop_19*100000),
        x = ('1996', '2000', '2010', '2019'))
plt.xlabel("Anos")
plt.xticks(rotation=45)
plt.ylabel("Óbitos a cada 100 mil habitantes")
plt.title("Óbitos a cada 100 mil habitantes por meningite (não especificada) de 1996 - 2019 no Brasil")
plt.show()


# ## IV. Tipos de meningite (CID-10) e número de óbitos

# In[ ]:


#Tabela
group_cid_obitos = df_meningite.groupby('causa_basica')['numero_obitos'].sum()
cid_obitos = pd.DataFrame(group_cid_obitos)
cid_obitos.reset_index(inplace=True)

group_cid_idade = df_meningite.groupby('causa_basica')['idade'].mean()
cid_idade = pd.DataFrame(group_cid_idade)
cid_idade.insert(value = df_meningite.groupby('causa_basica')['idade'].median(),
                 column = "Mediana", loc=1)
cid_idade.insert(value = df_meningite.groupby('causa_basica')['idade'].var(),
                 column = "Var", loc=2)
cid_idade.insert(value = df_meningite.groupby('causa_basica')['idade'].std(),
                 column = "DP", loc=3)
cid_idade.insert(value = group_cid_obitos, column = "Nº de óbitos", loc = 4)
cid_idade.insert(value = cid_idade['Nº de óbitos']/cid_idade['Nº de óbitos'].sum()*100,
                 column = "Percentual de óbitos", loc = 4) 
cid_idade.columns = ['Idade média', 'Idade mediana', 'Var', 'DP', 'Nº de óbitos', 'Percentual de óbitos']
cid_idade.reset_index(inplace=True)
cid_idade


# In[ ]:


meningite_idade_mean = df_meningite.groupby('causa_basica')['idade'].mean()

#Gráfico
plt.figure(figsize=(10, 6))
meningite_idade_mean.plot(kind = 'barh')
plt.xlabel("Tipo de meningite (CID-10)")
plt.ylabel("Média de idade dos óbitos")
plt.title("Média de idade dos óbitos causados por tipos específicos de meningite (CID-10) de 1996 - 2019 no Brasil")
plt.show()


# ### Distribuição dos tipos de meningite no Brasil
# O trabalho focará nos subgrupos de meningites especificados pelos códigos G00.0 e G00.1. Isso se dá pela prevalência da doença em seres humanos e no impacto causado na saúde pública.
# 
# G00.0 	  Meningite por Haemophilus
# 
# G00.1   	Meningite pneumocócica
# 
# G00.2   	Meningite estreptocócica
# 
# G00.3   	Meningite estafilocócica
# 
# G00.8   	Outras meningites bacterianas
# 
# G00.9   	Meningite bacteriana não especificada
# 
# *Obs:
# Meningite bacteriana não classificada em outra parte: G00;
# Meningite em doenças bacterianas classificadas em outra parte: G01;
# Meningite em doenças infecciosas e parasitárias classificadas em outra parte: G02;
# Meningite devida a outras causas e a causas não especificadas: G03

# In[ ]:


df_meningite = df_meningite[['ano', 'sigla_uf', 'causa_basica', 'idade', 'sexo']]


# In[ ]:


g0 = df_meningite[df_meningite['causa_basica'] == 'G000']
g1 = df_meningite[df_meningite['causa_basica'] == 'G001']
g2 = df_meningite[df_meningite['causa_basica'] == 'G002']
g3 = df_meningite[df_meningite['causa_basica'] == 'G003']
g8 = df_meningite[df_meningite['causa_basica'] == 'G008']
g9 = df_meningite[df_meningite['causa_basica'] == 'G009']
g00 = pd.concat([g0, g1, g2, g3, g8, g9])


# ### Meningite por *Haemophilus* sp.

# In[ ]:


#1996
h96 = g0[g0['ano'] == 1996]
#2000
h00 = g0[g0['ano'] == 2000]
#2010
h10 = g0[g0['ano'] == 2010]
#2019
h19 = g0[g0['ano'] == 2019]


# In[ ]:


#Óbitos de crianças em geral (<= 14 anos)
h96_ped = h96.query('idade <= 14')
h00_ped = h00.query('idade <= 14')
h10_ped = h10.query('idade <= 14')
h19_ped = h19.query('idade <= 14')
h_ped = g0.query('idade <= 14')

#Óbitos CLM (> 14 anos)
h96_clm = h96.query('idade > 14')
h00_clm = h00.query('idade > 14')
h10_clm = h10.query('idade > 14')
h19_clm = h19.query('idade > 14')
h_clm = g0.query('idade > 14')


# In[ ]:


#Intervalo de confiança para g0
li_g0 = np.percentile(g0['idade'], 2.5)
ls_g0 = np.percentile(g0['idade'], 97.5)

#Plotando a distribuição de idade
plt.hist(g0['idade'], bins=30, edgecolor='k')
plt.title('1996-2019; IC 95%')
plt.xlabel('Idade')
plt.ylabel('Número de óbitos')
plt.suptitle('Idade dos óbitos devido à meningite causada por Haemophilus sp. no Brasil')
plt.fill_between([li_g0, ls_g0], 4, 2000, alpha=0.8, color='grey')
plt.ylim(top=400)
despine()

#hist plot por grupo de anos
fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
axs[1,0].set_xlabel('Idade')
axs[1,1].set_xlabel('Idade')
axs[0,0].set_ylabel('Número de óbitos')
axs[1,0].set_ylabel('Número de óbitos')
axs[0,0].hist(h96['idade'])
axs[0,0].set_title('1996')
axs[0,1].hist(h10['idade'])
axs[0,1].set_title('2010')
axs[1,0].hist(h00['idade'])
axs[1,0].set_title('2000')
axs[1,1].hist(h19['idade'])
axs[1,1].set_title('2019')


# De 1996 para 2000, de acordo com o gráfico, houve uma queda no número de óbitos causados por meningite haemofílica. Se a queda realmente for evidenciada pela estatística, provavelmente se deve à introdução da vacina para o agente etiológico Haemophilus sp. no PNI

# In[ ]:


h96["idade_fixa"] = h96['idade']
h00["idade_fixa"] = h00['idade']
h10["idade_fixa"] = h10['idade']
h19["idade_fixa"] = h19['idade']


# **Teste de hipóteses**
# 
# H0: não há diferença estatística entre os óbitos ocorridos entre os grupos criança (<= 14 anos) e adulto (> 14 anos) em 1996 por meningite haemofílica

# Dados de 1996:

# In[ ]:


def bootstrap_mean(df1, df2, column, n=10000):
    size1 = len(df1)
    size2 = len(df2)
    values1 = np.zeros(n)
    values2 = np.zeros(n)
    values_diff = np.zeros(n)
    for i in range(n):
        sample1 = df1[column].sample(size1, replace=True, random_state=i)
        sample2 = df2[column].sample(size2, replace=True, random_state=i*3)
        values1[i] = sample1.mean()
        values2[i] = sample2.mean()
        values_diff[i] = sample1.mean() - sample2.mean()
    return values1, values2, values_diff

h96_ped = h96.query('idade_fixa <= 14')
h96_clm = h96.query('idade_fixa > 14')
col = 'idade'
h96_ped_perm, h96_clm_perm, h96_pedclm_perm = bootstrap_mean(h96_ped, h96_clm, col)


# In[ ]:


#Estatística observada:
h96_pedclm_obs = h96_clm['idade_fixa'].mean() - h96_ped['idade_fixa'].mean()
print('Estatística observada: ', h96_pedclm_obs)


# In[ ]:


#Histogramas dos resultados
plt.hist(h96_clm_perm, bins=100, edgecolor='k', label='Maior de 14 anos')
plt.hist(h96_ped_perm, bins=20, edgecolor='m', label='Menor ou igual a 14 anos')
plt.suptitle('Distribuição das idades médias dos óbitos de crianças e de adultos ocorridos por meningite pneumocócica em 1996')
plt.title('IC 95%')
plt.xlabel('Idade média')
plt.legend()
plt.fill_between([np.percentile(h96_clm_perm, 2.5),
                  np.percentile(h96_clm_perm, 97.5)],
                 4, 2000, alpha=0.8, color='grey')
plt.ylim(top=1000)
despine()
plt.show()


# In[ ]:


#Boxplot de cada grupo de idade para Haemophilus em 1996:
bp_data = [h96_ped_perm, h96_clm_perm]

plt.figure(figsize=(6, 4))
box = plt.boxplot(bp_data,
                  positions=[1, 2], whis=[2.5, 97.5],
                  labels=['Crianças', 'Adultos'])
plt.title('Boxplot - Idades médias de óbitos a cada 100 mil habitantes de crianças e adultos em 1996 por meningite causada por Haemophilus sp.')

plt.figure(figsize=(6, 4))
box = plt.boxplot(h96_pedclm_perm, whis=[2.5, 97.5],
                  labels=['Valor'])
plt.title('Boxplot - Diferença entre as idades médias de óbitos a cada 100 mil habitantes de crianças e adultos em 1996 por meningite causada por Haemophilus sp.')


# In[ ]:


#Cálculo do p-valor:
valor_p = (h96_pedclm_perm > h96_pedclm_obs).mean()
print('p-valor = ', valor_p) #1996


# Os boxplots não se cruzam no primeiro gráfico e o boxplot do último não cruza o zero. Podemos afirmar que existem evidências de que as médias de idade de óbitos dos grupos crianças (menores ou iguais a 14 anos) e adultos (maiores do que 14 anos) em 1996 de fato são diferentes. O resultado do p-valor (zero) também indica que há diferença estatística

# Dados de 2019:

# In[ ]:


#Bootstrap para a diferença de idade em todos os anos
h19_ped = h19.query('idade_fixa <= 14')
h19_clm = h19.query('idade_fixa > 14')
col = 'idade'
h19_ped_perm, h19_clm_perm, h19_pedclm_perm = bootstrap_mean(h19_ped, h19_clm, col)

#Estatística observada:
h19_pedclm_obs = h19_clm['idade_fixa'].mean() - h19_ped['idade_fixa'].mean()
h19_pedclm_obs

#Boxplot de cada grupo de idade para Haemophilus em 1996:
bp_data = [h19_ped_perm, h19_clm_perm]

plt.figure(figsize=(6, 4))
box = plt.boxplot(bp_data,
                  positions=[1, 2], whis=[2.5, 97.5],
                  labels=['Crianças', 'Adultos'])
plt.title('Boxplot - Idades médias de óbitos a cada 100 mil habitantes de crianças e adultos em 1996 por meningite causada por Haemophilus sp.no Brasil')

plt.figure(figsize=(6, 4))
box = plt.boxplot(h19_pedclm_perm, whis=[2.5, 97.5],
                  labels=['Valor'])
plt.title('Boxplot - Diferença entre as idades médias de óbitos a cada 100 mil habitantes de crianças e adultos em 1996 por meningite causada por Haemophilus sp.')


# In[ ]:


#Cálculo do p-valor:
valor_p = (h19_pedclm_perm > h96_pedclm_obs).mean()
print('p-valor = ', valor_p) #2019


# Ocorreram somente 10 óbitos por meningite haemofílica em 2019, o que reduz o nosso poder de inferência/estatístico
# 

# In[ ]:


h19.count()


# Também observamos que há uma diferença de idade de óbitos  entre os anos 1996 e 2000. Esta diferença é estatisticamente significativa?
# 
# **Teste de hipóteses**
# 
# H0: não há diferença estatística entre a idade média dos óbitos ocorridos em 1996 e 2000 por causa de meningite haemofílica

# In[ ]:


#Bootstrap para 1996 e 2000 - idades em geral:
h96_00 = pd.concat([h96, h00])

h96_idade = h96.query('ano == 1996')
h00_idade = h00.query('ano == 2000')
col = 'idade'

h96_perm, h00_perm, h0096_perm = bootstrap_mean(h96_idade, h00_idade, 'idade')


# In[ ]:


#Estatística observada:
h96_00_obs = h00_idade['idade_fixa'].mean() - h96_idade['idade_fixa'].mean()
h96_00_obs


# In[ ]:


#Histogramas dos resultados
plt.hist(h96_perm, bins=50, edgecolor='k', label='1996')
plt.hist(h00_perm, bins=100, edgecolor='m', label='2000')
plt.suptitle('Distribuição das idades médias dos óbitos ocorridos por meningite por Pneumococo em 1996 e 2000')
plt.title('IC 95%')
plt.xlabel('Idade média')
plt.fill_between([np.percentile(h96_perm, 2.5),
                  np.percentile(h96_perm, 97.5)],
                 4, 2000, alpha=0.8, color='blue')
plt.fill_between([np.percentile(h00_perm, 2.5),
                  np.percentile(h00_perm, 97.5)],
                 4, 2000, alpha=0.8, color='orange')
plt.ylim(top=1000)
plt.ylim(top=1000)
plt.legend()
despine()
plt.show()


# In[ ]:


#Boxplot 1996 x 2000
bp_data = [h96_perm, h00_perm]

plt.figure(figsize=(6, 4))
box = plt.boxplot(bp_data,
                  positions=[1, 2], whis=[2.5, 97.5],
                  labels=['1996', '2000'])
plt.title('Boxplot - Idades médias de óbitos a cada 100 mil habitantes nos anos de 1996 e 2000 por meningite causada por Haemophilus sp.')

plt.figure(figsize=(6, 4))
box = plt.boxplot(h0096_perm, whis=[2.5, 97.5],
                  labels=['Valor'])
plt.title('Boxplot - Diferença entre as idades médias de óbitos a cada 100 mil habitantes nos anos de 1996 e 2000 por meningite causada por Haemophilus sp.')


# In[ ]:


#Cálculo do p-valor:
valor_p = (h0096_perm > h96_00_obs).mean()
valor_p


# Embora os boxplots apresentem pontos em comum no primeiro gráfico e o boxplot do último gráfico cruza o zero, podemos afirmar que existem evidências de que as médias de idade de óbitos 1996 e 2000 são diferentes. O resultado do p-valor, 0.0159, indica que há diferença estatística.

# ### Meningite por *Pneumococo* sp.

# In[ ]:


g1["idade_fixa"] = g1['idade']

#1999
p96 = g1[g1['ano'] == 1996]
#2000
p00 = g1[g1['ano'] == 2000]
#2010
p10 = g1[g1['ano'] == 2010]
#2019
p19 = g1[g1['ano'] == 2019]


# In[ ]:


#Óbitos de crianças em geral (<= 14 anos)
p96_ped = p96.query('idade <= 14')
p00_ped = p00.query('idade <= 14')
p10_ped = p10.query('idade <= 14')
p19_ped = p19.query('idade <= 14')
p_ped = g1.query('idade <= 14')

#Óbitos CLM (> 18 anos)
p96_clm = p96.query('idade > 14')
p00_clm = p00.query('idade > 14')
p10_clm = p10.query('idade > 14')
p19_clm = p19.query('idade > 14')
p_clm = g1.query('idade > 14')


# In[ ]:


#Normalização dos dados
def norm(data):
  norm_data = (data - data.mean()) / data.std()
  return norm_data

#Percentil para um intervalo de 95% de confiança
li_g1 = np.percentile(g1['idade'], 2.5)
ls_g1 = np.percentile(g1['idade'], 97.5)

#Plotando a distribuição de idade
plt.hist(g1['idade'], bins=25, edgecolor='k')
plt.title('1996-2019; IC 95%')
plt.xlabel('Idade normalizada')
plt.ylabel('Número de óbitos')
plt.suptitle('Idade dos óbitos devido à meningite causada por Pneumococo no Brasil')
plt.fill_between([li_g1, ls_g1], 4, 2000, alpha=0.8, color='grey')
plt.ylim(top=1100)
despine()

#hist plot por grupo de anos
fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharey=True, sharex = True)
axs[0,0].hist(p96['idade'])
axs[0,0].set_title('1996')
axs[1,0].hist(p00['idade'])
axs[1,0].set_title('2000')
axs[0,1].hist(p10['idade'])
axs[0,1].set_title('2010')
axs[1,1].hist(p19['idade'])
axs[1,1].set_title('2019')


# Pela análise dos histogramas acima, podemos ter uma ideia de que a distribuição das idades médias de óbitos causados pelo pneumococo é bem diferente daquela devido ao *Haemophilus* sp.
# 
# **Teste de hipóteses**
# 
# H0: não há diferença estatística entre o grupo criança (<= 14 anos) e o grupo adulto (> 14 anos)

# In[ ]:


#Bootstrap para a diferença de idade em todos os anos
g1_ped = g1.query('idade_fixa <= 14')
g1_clm = g1.query('idade_fixa > 14')
col = 'idade'

g1_clm_perm, g1_ped_perm, g1_pedclm = bootstrap_mean(g1_clm, g1_ped, 'idade')


# In[ ]:


#Estatística observada:
htotal_pedclm_obs = g1_ped['idade_fixa'].mean() - g1_clm['idade_fixa'].mean()
htotal_pedclm_obs


# In[ ]:


#Histogramas dos resultados
plt.hist(g1_clm_perm, bins=100, edgecolor='k', label='Maior de 14 anos')
plt.hist(g1_ped_perm, bins=100, edgecolor='m', label='Menor ou igual a 14 anos')
plt.suptitle('Distribuição das idades médias dos óbitos de crianças e de adultos ocorridos por meningite pneumocócica no Brasil (1996 - 2019')
plt.xlabel('Idade média')
plt.legend()
despine()
plt.show()

#Boxplot de cada grupo de idade para Haemophilus em 1996:
bp_data = [g1_ped_perm, g1_clm_perm]

fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex = True)
fig.suptitle('Idade média dos óbitos de crianças e adultos por meningite pneumocócica no Brasil (1996 - 2019')
axs[0].boxplot(bp_data, positions=[1, 2], whis=[2.5, 97.5])
axs[0].set_title('Boxplot - Idades médias')
axs[1].boxplot(g1_pedclm, whis=[2.5, 97.5], positions=[1.5])
axs[1].set_title('Boxplot - Diferença entre as idades médias')


# Os boxplots, junto com a distribuição de probabilidade da idade média, indicam que provavelmente há diferença estatística entre os dois grupos etários durante o período de 1996 a 2019. A partir de agora, analisaremos somente o grupo de crianças, cujos dados disponíveis para análise da mortalidade estão muito mais completos. Antes, algumas comparações entre os dados de meningite por Haemophilus sp. e Pneumococo sp.

# ### Comparação entre as meningites pneumocócica e haemofílica

# Óbito de crianças em geral: idade menor ou igual a 14 anos
# 
# *OBS: óbito infantil corresponde aos óbitos de crianças com idade menor ou igual a 1 ano
# 
# **Percentual dos óbitos totais**

# In[ ]:


#1996
p96_t = p96['idade'].count()/df_meningite['idade'].count()
h96_t = h96['idade'].count()/df_meningite['idade'].count()
#2000
p00_t = p00['idade'].count()/df_meningite['idade'].count()
h00_t = h00['idade'].count()/df_meningite['idade'].count()
#2010
p10_t = p10['idade'].count()/df_meningite['idade'].count()
h10_t = h10['idade'].count()/df_meningite['idade'].count()
#2019
p19_t = p19['idade'].count()/df_meningite['idade'].count()
h19_t = h19['idade'].count()/df_meningite['idade'].count()
#Tabela
dict_percentual = {'1996':[p96_t, h96_t], '2000':[p00_t, h00_t], '2010':[p10_t, h10_t], '2019':[p19_t, h19_t]}
percentual = pd.DataFrame(dict_percentual)
pd.Index.names(percentual, ['Pneumococos sp.', 'Haemophilus sp.'])


# In[ ]:


#Gráfico
plt.figure(figsize=(10, 6))
percentual.plot(kind = 'barh')
plt.xlabel("% do total de óbitos")
plt.ylabel('Tipo de meningite')
plt.title("Percentual dos óbitos causados por tipos específicos de meningite (CID-10) das mortes totais de 1996 - 2019 no Brasil")
plt.show()


# #### Comparação no ano de 1996

# In[ ]:


ped96_boxplot = [p96_ped['idade'], h96_ped['idade']]

plt.figure(figsize=(8, 4))
box = plt.boxplot(x = (ped96_boxplot),
                  positions=[1, 2],
                  labels=['Pneumococo', 'Haemophilus'])
plt.title('Boxplot - Idade de acordo com a etiologia da meningite no Brasil')

fig, axs = plt.subplots(ncols = 2, figsize = (12, 5), sharey=True)
fig.suptitle('Histograma - Idade de acordo com a etiologia da meningite em 1996')
axs[0].hist(h_ped['idade'], bins = 20)
axs[0].set_title('Haemophilus')
axs[1].hist(p_ped['idade'], bins = 20)
axs[1].set_title('Pneumococo')


# In[ ]:


ecdf_p_ped = ECDF(p96_ped['idade'])
xl_p_ped = ecdf_p_ped.x
yl_p_ped = ecdf_p_ped.y

ecdf_h_ped = ECDF(h96_ped['idade'])
xl_h_ped = ecdf_h_ped.x
yl_h_ped = ecdf_h_ped.y

plt.plot(xl_p_ped, yl_p_ped, label='Pneumococo')
plt.plot(xl_h_ped, yl_h_ped, label='Haemophilus')
plt.xlabel('Idade do óbito')
plt.ylabel('% do total de óbitos')
plt.suptitle('Empirical CDF - Idade dos óbitos de crianças')
plt.legend()
plt.show()


# #### Comparação no ano de 2019

# In[ ]:


ped19_boxplot = [p19_ped['idade'], h19_ped['idade']]

plt.figure(figsize=(8, 4))
box = plt.boxplot(x = (ped19_boxplot),
                  positions=[1, 2],
                  labels=['Pneumococo', 'Haemophilus'])
plt.title('Boxplot - Idade de acordo com a etiologia da meningite no Brasil')


# In[ ]:


ecdf_p19_ped = ECDF(p19_ped['idade'])
xl_p_ped = ecdf_p19_ped.x
yl_p_ped = ecdf_p19_ped.y

ecdf_h19_ped = ECDF(h19_ped['idade'])
xl_h_ped = ecdf_h19_ped.x
yl_h_ped = ecdf_h19_ped.y

plt.plot(xl_p_ped, yl_p_ped, label='Pneumococo')
plt.plot(xl_h_ped, yl_h_ped, label='Haemophilus')
plt.xlabel('Idade do óbito')
plt.ylabel('% do total de óbitos')
plt.suptitle('Empirical CDF - Idade dos óbitos de crianças')
plt.legend()
plt.show()


# Existem poucos dados para analisarmos as mortes causadas pelo agente *Haemophilus* sp. e sua distribuição no Brasil, principalmente em anos mais recentes. Vamos verificar os dados das mortes por meningites causadas por pneumococos

# ## V. Meningite pneumocócica (CID-10 G001)

# A partir desta seção, o enfoque do trabalho será nos óbitos decorrentes de meningite causada pelo *Pneumococo* sp.. Os dados serão divididos por região da Federação e ano. Será feita uma breve análise da distribuição no Brasil, para então aplicarmos métodos de estatística inferencial.

# In[ ]:


#População por região administrativa (estimado - IBGE)
pop_se = 89632912
pop_ne = 57667842
pop_s = 30402587
pop_n = 18906962
pop_co = 6707336
pop_br = pop_se + pop_ne + pop_s + pop_n + pop_co


# In[ ]:


def norte(df):
    am = df[df['sigla_uf'] == 'AM']
    rr = df[df['sigla_uf'] == 'RR']
    ap = df[df['sigla_uf'] == 'AP']
    pa = df[df['sigla_uf'] == 'PA']
    to = df[df['sigla_uf'] == 'TO']
    ro = df[df['sigla_uf'] == 'RO']
    ac = df[df['sigla_uf'] == 'AC']
    N = pd.concat([am, rr, ap, pa, to, ro, ac])
    return N
def centro(df):
    co = df[df['sigla_uf'] == 'MT']
    mt = df[df['sigla_uf'] == 'MS']
    go = df[df['sigla_uf'] == 'GO']
    c = pd.concat([co, mt, go])
    return c
def nordeste(df):
    al = df[df['sigla_uf'] == 'AL']
    ba = df[df['sigla_uf'] == 'BA']
    ce = df[df['sigla_uf'] == 'CE']
    ma = df[df['sigla_uf'] == 'MA']
    pb = df[df['sigla_uf'] == 'PB']
    pe = df[df['sigla_uf'] == 'PE']
    pi = df[df['sigla_uf'] == 'PI']
    rn = df[df['sigla_uf'] == 'RN']
    se = df[df['sigla_uf'] == 'SE']
    n = pd.concat([al, ba, ce, ma,
                      pb, pe, pi, rn, se])
    return n
def sudeste(df):
    sp = df[df['sigla_uf'] == 'SP']
    rj = df[df['sigla_uf'] == 'RJ']
    mg = df[df['sigla_uf'] == 'MG']
    es = df[df['sigla_uf'] == 'ES']
    s = pd.concat([sp, rj, mg, es])
    return s
def sul(df):
    rs = df[df['sigla_uf'] == 'RS']
    sc = df[df['sigla_uf'] == 'SC']
    pr = df[df['sigla_uf'] == 'PR']
    S = pd.concat([rs, sc, pr])
    return S


# In[ ]:


def regiao_df(df):
    am = df[df['sigla_uf'] == 'AM']
    rr = df[df['sigla_uf'] == 'RR']
    ap = df[df['sigla_uf'] == 'AP']
    pa = df[df['sigla_uf'] == 'PA']
    to = df[df['sigla_uf'] == 'TO']
    ro = df[df['sigla_uf'] == 'RO']
    ac = df[df['sigla_uf'] == 'AC']
    N = pd.concat([am, rr, ap, pa, to, ro, ac])
    #Pneumo - CO: MT, MS, GO
    co = df[df['sigla_uf'] == 'MT']
    mt = df[df['sigla_uf'] == 'MS']
    go = df[df['sigla_uf'] == 'GO']
    c = pd.concat([co, mt, go])
    #Pneumo - NE
    al = df[df['sigla_uf'] == 'AL']
    ba = df[df['sigla_uf'] == 'BA']
    ce = df[df['sigla_uf'] == 'CE']
    ma = df[df['sigla_uf'] == 'MA']
    pb = df[df['sigla_uf'] == 'PB']
    pe = df[df['sigla_uf'] == 'PE']
    pi = df[df['sigla_uf'] == 'PI']
    rn = df[df['sigla_uf'] == 'RN']
    se = df[df['sigla_uf'] == 'SE']
    n = pd.concat([al, ba, ce, ma,
                      pb, pe, pi, rn, se])
    #Pneumo - SE
    sp = df[df['sigla_uf'] == 'SP']
    rj = df[df['sigla_uf'] == 'RJ']
    mg = df[df['sigla_uf'] == 'MG']
    es = df[df['sigla_uf'] == 'ES']
    s = pd.concat([sp, rj, mg, es])
    #Pneumo - S
    rs = df[df['sigla_uf'] == 'RS']
    sc = df[df['sigla_uf'] == 'SC']
    pr = df[df['sigla_uf'] == 'PR']
    S = pd.concat([rs, sc, pr])

    norte = pd.Series([len(N), len(N)*100000/pop_n, N['idade'].mean(), N['idade'].median(),
                   N['idade'].var(), N['idade'].std()],
                  index=['Nº óbitos', 'Óbitos por 100.000 habitantes', 'Idade média',  'Idade mediana',
                         'Variância da idade', 'D.P. idade'])
    nordeste = pd.Series([len(n), len(n)*100000/pop_ne, n['idade'].mean(), n['idade'].median(),
                   n['idade'].var(), n['idade'].std()],
                  index=['Nº óbitos', 'Óbitos por 100.000 habitantes', 'Idade média',  'Idade mediana',
                         'Variância da idade', 'D.P. idade'])
    centro = pd.Series([len(c), len(c)*100000/pop_co, c['idade'].mean(), c['idade'].median(),
                   c['idade'].var(), c['idade'].std()],
                  index=['Nº óbitos', 'Óbitos por 100.000 habitantes', 'Idade média',  'Idade mediana',
                         'Variância da idade', 'D.P. idade'])
    sudeste = pd.Series([len(s), len(s)*100000/pop_se, s['idade'].mean(), s['idade'].median(),
                   s['idade'].var(), s['idade'].std()],
                  index=['Nº óbitos', 'Óbitos por 100.000 habitantes', 'Idade média',  'Idade mediana',
                         'Variância da idade', 'D.P. idade'])
    sul = pd.Series([len(S), len(S)*100000/pop_s, S['idade'].mean(), S['idade'].median(),
                   S['idade'].var(), S['idade'].std()],
                  index=['Nº óbitos', 'Óbitos por 100.000 habitantes', 'Idade média',  'Idade mediana',
                         'Variância da idade', 'D.P. idade'])
    df_novo = pd.DataFrame({'Norte':norte, 'Nordeste':nordeste, 'Centro-Oeste':centro, 
             'Sudeste':sudeste, 'Sul':sul})
    return df_novo


# ### Uma visão geral sobre os óbitos por meningite pneumocócica no Brasil
# 
# **Dados de 1996**

# In[ ]:


regiao_df(p96_ped)


# Óbitos de crianças em 1996 por meningite penumocócica por idade e região

# In[ ]:


#Boxplot - comparação
p96_boxplot = [norte(p96_ped)['idade'], nordeste(p96_ped)['idade'], centro(p96_ped)['idade'], sul(p96_ped)['idade'],
               sudeste(p96_ped)['idade']]

plt.figure(figsize=(8, 4))
box = plt.boxplot(p96_boxplot,
                  positions=[1, 1.75, 2.5, 3.25, 4],
                  bootstrap=10000,
                  labels=['N', 'NE', 'CO', 'SE', 'S'])
plt.title('Boxplot)) - Idade das mortes por meningite pneumocócica em 1996')


# Traçando a ECDF para as diferentes regiões no ano de 1996:

# In[ ]:


#CDF para cada região
#CDF: probabilidade de que a variável idade (X) assuma um valor inferior ou igual a x

ecdf_n96 = ECDF(norte(p96_ped)['idade'])
xl_n96 = ecdf_n96.x
yl_n96 = ecdf_n96.y

ecdf_ne96 = ECDF(nordeste(p96_ped)['idade'])
xl_ne96 = ecdf_ne96.x
yl_ne96 = ecdf_ne96.y

ecdf_c96 = ECDF(centro(p96_ped)['idade'])
xl_c96 = ecdf_c96.x
yl_c96 = ecdf_c96.y

ecdf_s96 = ECDF(sul(p96_ped)['idade'])
xl_s96 = ecdf_s96.x
yl_s96 = ecdf_s96.y

ecdf_se96 = ECDF(sudeste(p96_ped)['idade'])
xl_se96 = ecdf_se96.x
yl_se96 = ecdf_se96.y

plt.plot(xl_n96, yl_n96)
plt.xlabel('N')
plt.ylabel('$P(X \leq x)$')
despine()

plt.plot(xl_ne96, yl_ne96)
plt.xlabel('NE')
plt.ylabel('$P(X \leq x)$')
despine()

plt.plot(xl_c96, yl_c96)
plt.xlabel('CO')
plt.ylabel('$P(X \leq x)$')
despine()

plt.plot(xl_s96, yl_s96)
plt.xlabel('S')
plt.ylabel('$P(X \leq x)$')
despine()

plt.plot(xl_se96, yl_se96)
plt.xlabel('SE')
plt.ylabel('$P(X \leq x)$')
despine()

plt.suptitle('Empirical CDF - Idade dos óbitos de crianças')
plt.legend(['N', 'NE', 'CO', 'SE', 'S'])
plt.show()


# A seguir, será realizado um teste de hipóteses comparando as regiões N e SE no ano de 1996. Essas duas regiões foram as escolhidas para análise por possuirem o menor e o maior IDH respectivamente
# 
# **Teste de hipóteses**
# **H0:** as regiões N e SE apresentam idades médias de óbitos de crianças por meningite pneumocócica sem diferença estatisticamente relevantes no ano de 1996

# In[ ]:


#Estatística observada:
p19_nse_obs = sudeste(p19_ped)['idade_fixa'].mean() - norte(p19_ped)['idade_fixa'].mean()

#Bootstrap
col = 'idade'
p19_n_perm, p19_se_perm, p19_nse_perm = bootstrap_mean(norte(p19_ped), sudeste(p19_ped), col)

#Boxplot de cada grupo de idade para Haemophilus em 1996:
bp_data = [p19_n_perm, p19_se_perm]

plt.figure(figsize=(6, 4))
box = plt.boxplot(bp_data,
                  positions=[1, 2], whis=[2.5, 97.5],
                  labels=['N', 'SE'])
plt.title('Boxplot - Idades médias de óbitos de crianças em 1996 por meningite causada por Haemophilus sp. nas regiões N e SE')

plt.figure(figsize=(6, 4))
box = plt.boxplot(p19_nse_perm, whis=[2.5, 97.5],
                  labels=['Valor'])
plt.title('Boxplot - Diferença entre as idades médias de óbitos de crianças em 1996 por meningite causada por Haemophilus sp. nas regiões N e SE')


# In[ ]:


#Cálculo do p-valor:
valor_p = (h19_pedclm_perm > h96_pedclm_obs).mean()
valor_p #2019


# **Dados de 2019**

# Para verificarmos se houve alguma diferença ao longo do tempo na idade média de óbitos de crianças causados por meningite pneumocócica ao longo dos últimos anos, iremos analisar somente os dados mais atuais (2019) e mais antigos (1996)

# In[ ]:


regiao_df(p96_ped)


# Óbitos de crianças em 1996 por meningite penumocócica por idade e região

# In[3]:


#Boxplot - comparação
p00_boxplot = [norte(p19_ped)['idade'], nordeste(p19_ped)['idade'], centro(p19_ped)['idade'], sul(p19_ped)['idade'],
               sudeste(p19_ped)['idade']]

plt.figure(figsize=(8, 4))
box = plt.boxplot(p00_boxplot,
                  positions=[1, 1.75, 2.5, 3.25, 4],
                  bootstrap=10000,
                  labels=['N', 'NE', 'CO', 'SE', 'S'])
plt.title('Boxplot)) - Idade das mortes por meningite pneumocócica em 2019')


# Traçando a ECDF para as diferentes regiões no ano de 2019:

# In[ ]:


#CDF para cada região
#CDF: probabilidade de que a variável idade (X) assuma um valor inferior ou igual a x

ecdf_n00 = ECDF(norte(p19_ped)['idade'])
xl_n00 = ecdf_n00.x
yl_n00 = ecdf_n00.y

ecdf_ne00 = ECDF(nordeste(p19_ped)['idade'])
xl_ne00 = ecdf_ne00.x
yl_ne00 = ecdf_ne00.y

ecdf_c00 = ECDF(centro(p19_ped)['idade'])
xl_c00 = ecdf_c00.x
yl_c00 = ecdf_c00.y

ecdf_s00 = ECDF(sul(p19_ped)['idade'])
xl_s00 = ecdf_s00.x
yl_s00 = ecdf_s00.y

ecdf_se00 = ECDF(sudeste(p19_ped)['idade'])
xl_se00 = ecdf_se00.x
yl_se00 = ecdf_se00.y

plt.plot(xl_n00, yl_n00)
plt.xlabel('N')
plt.ylabel('$P(X \leq x)$')
despine()

plt.plot(xl_ne00, yl_ne00)
plt.xlabel('NE')
plt.ylabel('$P(X \leq x)$')
despine()

plt.plot(xl_c00, yl_c00)
plt.xlabel('CO')
plt.ylabel('$P(X \leq x)$')
despine()

plt.plot(xl_s00, yl_s00)
plt.xlabel('S')
plt.ylabel('$P(X \leq x)$')
despine()

plt.plot(xl_se00, yl_se00)
plt.xlabel('SE')
plt.ylabel('$P(X \leq x)$')
despine()

plt.suptitle('Empirical CDF - Idade dos óbitos de crianças')
plt.legend(['N', 'NE', 'CO', 'SE', 'S'])
plt.show()


# A seguir, será realizado um outro teste de hipóteses comparando as regiões N e SE, mas para o ano de 2019.
# 
# **Teste de hipóteses**
# **H0:** as regiões N e SE apresentam idades médias de óbitos de crianças por meningite pneumocócica sem diferença estatisticamente relevantes no ano de 2019

# In[ ]:


#Estatística observada:
p19_nse_obs = sudeste(p19_ped)['idade_fixa'].mean() - norte(p19_ped)['idade_fixa'].mean()

#Bootstrap
col = 'idade'
p19_n_perm, p19_se_perm, p19_nse_perm = bootstrap_mean(norte(p19_ped), sudeste(p19_ped), col)

#Boxplot de cada grupo de idade para Haemophilus em 1996:
bp_data = [p19_n_perm, p19_se_perm]

plt.figure(figsize=(6, 4))
box = plt.boxplot(bp_data,
                  positions=[1, 2], whis=[2.5, 97.5],
                  labels=['N', 'SE'])
plt.title('Boxplot - Idades médias de óbitos de crianças em 2019 por meningite causada por Haemophilus sp. nas regiões N e SE')

plt.figure(figsize=(6, 4))
box = plt.boxplot(p19_nse_perm, whis=[2.5, 97.5],
                  labels=['Valor'])
plt.title('Boxplot - Diferença entre as idades médias de óbitos de crianças em 2019 por meningite causada por Haemophilus sp. nas regiões N e SE')


# In[ ]:


#Cálculo do p-valor:
valor_p = (h19_pedclm_perm > h96_pedclm_obs).mean()
valor_p #2019


# ### Inferência estatística

# Compilando em um Data Frame os dados dos óbitos por meningite pneumocócica. Cada coluna representa uma variável categórica. Para 
# realizar a regressão múltipla, devemos transformar os dados em números:

# **1) Correlação entre as idades médias de óbitos por meningite pneumocócica de 1996 e de 2019 no Brasil**

# In[ ]:


dict_9619_total = {'1996':p96_ped['idade'], '2019':p19_ped['idade']}
total_9619 = pd.DataFrame(dict_9619_total)
total_9619


# In[ ]:


plt.scatter(total_9619['1996'], total_9619['2019'], edgecolor='k', alpha=0.6, s=80)
plt.xlabel('1996')
plt.ylabel('2019')
plt.title('Idade dos óbitos: 1996 x 2019')
despine()


# In[ ]:


def covariance(x, y):
    n = len(x)
    x_m = x - np.mean(x)
    y_m = y - np.mean(y)
    return (x_m * y_m).sum() / (n - 1)
cov96_19 = covariance(total_9619['1996'], total_9619['2019'])

def corr(x, y):
    n = len(x)
    x_m = x - np.mean(x)
    x_m = x_m / np.std(x, ddof=1)
    y_m = y - np.mean(y)
    y_m = y_m / np.std(y, ddof=1)
    return (x_m * y_m).sum() / (n - 1)
corr96_19 = corr(total_9619['1996'], total_9619['2019'])

print('Covariância: ', cov96_19)
print('Correlação: ', corr96_19)


# **2) Existe correção entre as idades médias de óbitos por meningite pneumocócica do N e de SE em 1996?**

# In[ ]:


dict_1996_total = {'N':norte(p96_ped)['idade'], 'SE':sudeste(p96_ped)['idade']}
total_1996 = pd.DataFrame(dict_1996_total)
total_1996


# In[ ]:


N96 = total_1996['N']
SE96 = total_1996['SE']

plt.scatter(N96, SE96, edgecolor='k', alpha=0.6, s=80)
plt.xlabel('N')
plt.ylabel('SE')
plt.title('Idade dos óbitos em 1996: N x SE')
despine()


# In[ ]:


print('Covariância: ', covariance(N96, SE96))
print('Correlação: ', corr(N96, SE96))


# **3) Existe correlação entre as idades médias de óbitos por meningite pneumocócica do N e de SE em 2019?**

# In[ ]:


dict_2019_total = {'N':norte(p19_ped)['idade'], 'SE':sudeste(p19_ped)['idade']}
total_2019 = pd.DataFrame(dict_2019_total)
total_2019


# In[ ]:


N19 = total_2019['N']
SE19 = total_2019['SE']

plt.scatter(N19, SE19, edgecolor='k', alpha=0.6, s=80)
plt.xlabel('N')
plt.ylabel('SE')
plt.title('Idade dos óbitos em 1996: N x SE')
despine()


# In[ ]:


print('Covariancia: ', covariance(N19, SE19))
print('Correlação: ', corr(N19, SE19))


# ## VI. Conclusão
# 
# Após análise dos dados comparando as idades dos óbitos de crianças dos anos 1996 e 2019 e das regiões N e SE, podemos concluir:
# 
# 1) Evidências indicam que não há correlação entre as idades dos óbitos de crianças por meningite pneumocócica registradas em 1996 e 2019;
# 
# 2) Evidências indicam que não há correlação entre as idades dos óbitos de crianças por meningite pneumocócica registradas nas regiões N e SE em 1996;
# 
# 3) Evidências indicam que não há correlação entre as idades dos óbitos de crianças por meningite pneumocócica registradas nas regiões N e SE em 2019.
