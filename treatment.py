# -*- coding: utf-8 -*-
"""
Script Python para Processamento de Dados e Detecção de Outliers

Este script carrega um arquivo CSV, realiza transformações básicas (renomeação e formatação de colunas)
e aplica o algoritmo Isolation Forest para detectar e remover outliers.
O resultado é salvo em um novo arquivo CSV sem os outliers.
"""

# --- 0. Importação de Bibliotecas Necessárias ---
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import os # Para verificar a existência de arquivos

# --- 1. Configuração de Caminhos de Arquivo ---
# Defina o caminho para o arquivo CSV de entrada.
# Certifique-se de que este arquivo esteja acessível no ambiente onde o script será executado.
input_csv_path = '/content/parquetreader (2).csv' # Caminho do arquivo original

# Defina o caminho para o arquivo CSV de saída, onde o DataFrame limpo será salvo.
output_cleaned_csv_path = 'dataframe_transacoes_limpo.csv'

# --- 2. Carregamento dos Dados ---
print(f"Carregando dados do arquivo: {input_csv_path}")
try:
    df = pd.read_csv(input_csv_path)
    print("Dados carregados com sucesso!")
    print("Amostra inicial dos dados:")
    print(df.head())
    print(f"\nShape inicial do DataFrame: {df.shape}")

except FileNotFoundError:
    print(f"Erro: O arquivo '{input_csv_path}' não foi encontrado.")
    print("Por favor, verifique o caminho do arquivo e tente novamente.")
    # Em um script para GitHub, você pode querer sair ou levantar uma exceção aqui
    # exit() # Descomente para sair do script em caso de erro
    raise # Levanta a exceção para indicar que o script não pode continuar

# --- 3. Alterações e Preparação dos Dados ---

# 3.1. Renomeação de Colunas (para português)
print("\nRenomeando colunas...")
df = df.rename(columns={
    'internal_store_id': 'id_loja_interna',
    'internal_product_id': 'id_produto_interno',
    'distributor_id': 'id_distribuidor',
    'transaction_date': 'data_transacao',
    'reference_date': 'data_referencia',
    'quantity': 'quantidade',
    'gross_value': 'valor_bruto',
    'net_value': 'valor_liquido',
    'gross_profit': 'lucro_bruto',
    'discount': 'desconto',
    'taxes': 'impostos'
})
print("Colunas renomeadas com sucesso!")
print("Colunas do DataFrame após renomeação:")
print(df.columns.tolist())


# 3.2. Formatação de Colunas Numéricas (para duas casas decimais)
print("\nFormatando colunas numéricas para 2 casas decimais...")
# Identificar colunas numéricas, excluindo IDs que são inteiros grandes
numerical_cols_to_format = df.select_dtypes(include=np.number).columns.tolist()
id_cols = ['id_loja_interna', 'id_produto_interno', 'id_distribuidor']
# Remover IDs da lista de colunas a formatar se estiverem presentes
numerical_cols_to_format = [col for col in numerical_cols_to_format if col not in id_cols]

for col in numerical_cols_to_format:
    # Verificar se a coluna existe no DataFrame antes de tentar formatar
    if col in df.columns:
      df[col] = df[col].round(2)
print("Colunas numéricas formatadas com sucesso!")
print("Amostra dos dados após formatação:")
print(df.head())


# 3.3. Seleção de Features para Detecção de Outliers e Tratamento de Nulos
print("\nPreparando dados para detecção de outliers...")
# Selecionando as features numéricas para análise.
# IDs e datas não entram diretamente no modelo de detecção de anomalias numéricas.
features_for_outlier_detection = ['quantidade', 'valor_bruto', 'valor_liquido', 'lucro_bruto', 'desconto', 'impostos']
X = df[features_for_outlier_detection]

# Verificar e tratar valores nulos. Usar a mediana é mais robusto a outliers.
if X.isnull().sum().any():
    print("Valores nulos encontrados. Preenchendo com a mediana.")
    # Calcular a mediana apenas das colunas relevantes
    median_values = X.median()
    X = X.fillna(median_values)
    print("Valores nulos preenchidos.")
else:
    print("Nenhum valor nulo encontrado nas features selecionadas.")


# --- 4. Detecção de Outliers com Isolation Forest ---
print("\nConfigurando e treinando o modelo Isolation Forest para detecção de outliers...")
# Instanciação do Modelo
# n_estimators: Número de árvores na floresta.
# contamination: Proporção de outliers esperada. 'auto' é comum, mas um valor fixo (ex: 0.01)
#                pode ser usado se houver uma hipótese sobre a proporção.
# random_state: Para reprodutibilidade.
# n_jobs=-1: Usar todos os processadores disponíveis.
model = IsolationForest(
    n_estimators=100,
    contamination='auto', # Pode ser ajustado para um valor float (ex: 0.01) se souber a proporção esperada
    random_state=42,
    n_jobs=-1
)

# Treinamento e Predição
# O método fit_predict treina o modelo e retorna as predições (-1 para outlier, 1 para inlier).
df['outlier_flag'] = model.fit_predict(X)

# A coluna 'decision_function' pode ser útil para ver o "score" de anomalia.
# Scores mais baixos indicam maior probabilidade de ser um outlier.
df['outlier_score'] = model.decision_function(X)

print("Detecção de outliers concluída.")
print(f"Total de transações analisadas: {len(df)}")
total_outliers = df[df['outlier_flag'] == -1].shape[0]
print(f"Total de outliers identificados: {total_outliers}")
if len(df) > 0:
    percentage_outliers = (total_outliers / len(df)) * 100
    print(f"Percentual de outliers: {percentage_outliers:.2f}%")

# --- 5. Remoção de Outliers ---
print("\nRemovendo outliers do DataFrame...")
# Filtrar o DataFrame para remover os outliers (outlier_flag == -1)
df_cleaned = df[df['outlier_flag'] == 1].copy()
print("Outliers removidos.")
print(f"Novo shape do DataFrame (limpo): {df_cleaned.shape}")
print(f"Número de linhas removidas: {len(df) - len(df_cleaned)}")


# --- 6. Salvamento do DataFrame Limpo ---
print(f"\nSalvando DataFrame limpo em: {output_cleaned_csv_path}")
try:
    df_cleaned.to_csv(output_cleaned_csv_path, index=False)
    print("DataFrame limpo salvo com sucesso!")
    # Opcional: Verificar o tamanho do arquivo salvo
    cleaned_size = os.path.getsize(output_cleaned_csv_path)
    cleaned_size_mb = cleaned_size / (1024 * 1024)
    print(f"Tamanho do arquivo limpo salvo (MB): {cleaned_size_mb:.2f} MB")

except Exception as e:
    print(f"Erro ao salvar o arquivo limpo: {e}")


print("\nProcessamento de dados e detecção/remoção de outliers concluído.")
