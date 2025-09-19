import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

datasetsFolder = "./Datasets"

db1 = datasetsFolder + "/part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy.parquet"
db1 = pd.read_parquet(db1)

db1.rename(
    columns=
    {
    'pdv': 'id_pdv',
    'premise': 'tipo_local',
    'zipcode': 'cep'
    },
    inplace=True
)

db1.head()

db2 = datasetsFolder + "/part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet"
db2 = pd.read_parquet(db2)

db2.rename(
    columns=
    {
    'internal_store_id': 'id_loja',
    'internal_product_id': 'id_produto',
    'distributor_id': 'id_distribuidor',
    'transaction_date': 'data_transacao',
    'reference_date': 'data_referencia',
    'quantity': 'quantidade',
    'gross_value': 'valor_bruto',
    'net_value': 'valor_liquido',
    'gross_profit': 'lucro_bruto',
    'discount': 'desconto',
    'taxes': 'impostos'
    },
    inplace=True
)

print(db2.head())
print(db2.groupby("id_loja").size())

db2.info()

db3 = datasetsFolder + "/part-00000-tid-7173294866425216458-eae53fbf-d19e-4130-ba74-78f96b9675f1-4-1-c000.snappy.parquet"
db3 = pd.read_parquet(db3)

db3.rename(
    columns=
    {
    'tipos': 'tipo',
    'label': 'rotulo',
    'produto': 'id_produto'
    },
    inplace=True
)

print(db3.head())
print(db3.groupby("categoria").size())

num_lojas_transacoes = db1['id_pdv'].nunique()
num_lojas_cadastro_pdv = db2['id_loja'].nunique()

print(f"Lojas únicas nas Transações: {num_lojas_transacoes}")
print(f"Lojas únicas no Cadastro de PDVs: {num_lojas_cadastro_pdv}")
print(f"Lojas 'órfãs' (sem cadastro): {num_lojas_cadastro_pdv - num_lojas_transacoes}")

# --- Passo 1.1: Preparar a tabela de PDVs ---
# Renomeia a coluna 'id_pdv' em db1 para 'id_loja' para padronizar
print("Renomeando 'id_pdv' para 'id_loja' na tabela de cadastro...")
db1 = db1.rename(columns={'id_pdv': 'id_loja'})
print("Coluna renomeada com sucesso.")
print("-" * 50)


# --- Passo 1.2: Unificar Transações com Produtos ---
# Juntamos as transações (db2) com os detalhes dos produtos (df_produtos)
# Usamos 'how="left"' para garantir que nenhuma transação seja perdida
print("Juntando transações com os detalhes dos produtos...")
df_unificado_parcial = pd.merge(
    left=db2,
    right=db3,
    on='id_produto',
    how='left'
)
print(f"Merge parcial concluído. O DataFrame agora tem {df_unificado_parcial.shape[1]} colunas.")
print("-" * 50)


# --- Passo 1.3: Unificar o resultado com os dados das Lojas (PDVs) ---
# Agora, juntamos o resultado anterior com os dados cadastrais das lojas (db1)
print("Juntando o resultado com os dados cadastrais das lojas...")
df_final = pd.merge(
    left=df_unificado_parcial,
    right=db1,
    on='id_loja',
    how='left'
)
print("Merge final concluído!")
print("-" * 50)


# --- Verificação Final ---
print(f"O seu DataFrame final unificado tem {df_final.shape[0]} linhas e {df_final.shape[1]} colunas.")
print("\nColunas do DataFrame final:")
print(df_final.columns.tolist())

print("\nVisualização das primeiras linhas do DataFrame unificado:")
print(df_final.head())

# Verificando as transações de lojas 'órfãs' (que ficaram com dados nulos)
lojas_orfas = df_final[df_final['categoria_pdv'].isnull()]
print(f"\nEncontradas {len(lojas_orfas)} transações de lojas sem cadastro no PDV (valores nulos).")

# Faz uma cópia do DataFrame para segurança (boa prática)
df_tratado = df_final.copy()

# Estratégia 1: Tratar nulos das colunas de PDV
print("Tratando nulos das colunas de PDV...")
df_tratado['tipo_local'] = df_tratado['tipo_local'].fillna('Desconhecido')
df_tratado['categoria_pdv'] = df_tratado['categoria_pdv'].fillna('Desconhecido')
df_tratado['cep'] = df_tratado['cep'].fillna(0) # Usando 0 para CEP desconhecido

# Estratégia 2: Tratar nulos das colunas de Produtos
print("Tratando nulos das colunas de Produtos...")
df_tratado['rotulo'] = df_tratado['rotulo'].fillna('Nao Informado')
df_tratado['subcategoria'] = df_tratado['subcategoria'].fillna('Nao Informado')

print("\nTratamento de valores nulos concluído sem avisos!")

# Verificação Final
contagem_final_nulos = df_tratado.isnull().sum()
print("\nContagem de nulos após o tratamento (apenas colunas com nulos):")
print(contagem_final_nulos[contagem_final_nulos > 0])

# Garante que o DataFrame está ordenado por data
df_tratado = df_tratado.sort_values(by='data_transacao')

# 1. Agrupa por categoria e soma a quantidade
top_categorias = df_tratado.groupby('categoria')['quantidade'].sum().sort_values(ascending=False).head(10)

# 2. Cria o gráfico
plt.figure(figsize=(12, 7)) # Define o tamanho da figura
sns.barplot(x=top_categorias.index, y=top_categorias.values, palette='viridis')

plt.title('Top 10 Categorias de Produtos Mais Vendidos', fontsize=16)
plt.xlabel('Categoria do Produto', fontsize=12)
plt.ylabel('Quantidade Total Vendida', fontsize=12)
plt.xticks(rotation=45, ha='right') # Rotaciona os rótulos do eixo X para não sobrepor
plt.tight_layout() # Ajusta o layout para tudo caber direitinho
plt.show()

# Garante que o DataFrame está ordenado por data (boa prática, mas não estritamente necessário para este gráfico de soma total)
df_tratado = df_tratado.sort_values(by='data_transacao')

# 1. Agrupa por categoria e soma o lucro bruto
top_lucro_categorias = df_tratado.groupby('categoria')['lucro_bruto'].sum().sort_values(ascending=False).head(10)

# 2. Cria o gráfico
plt.figure(figsize=(12, 7)) # Define o tamanho da figura
sns.barplot(x=top_lucro_categorias.index, y=top_lucro_categorias.values, palette='viridis')

plt.title('Top 10 Categorias de Produtos por Lucro Bruto', fontsize=16)
plt.xlabel('Categoria do Produto', fontsize=12)
plt.ylabel('Lucro Bruto Total', fontsize=12)
plt.xticks(rotation=45, ha='right') # Rotaciona os rótulos do eixo X para não sobrepor
plt.tight_layout() # Ajusta o layout para tudo caber direitinho
plt.show()

# Certifique-se de que 'data_transacao' está em formato datetime
df_tratado['data_transacao'] = pd.to_datetime(df_tratado['data_transacao'])

# Extrai o mês da data
df_tratado['mes_transacao'] = df_tratado['data_transacao'].dt.month

# Agrupa por mês e soma a quantidade e o lucro bruto
vendas_por_mes = df_tratado.groupby('mes_transacao').agg(
    total_quantidade=('quantidade', 'sum'),
    total_lucro_bruto=('lucro_bruto', 'sum')
).reset_index()

# Mapeia os números dos meses para nomes para melhor legibilidade no gráfico
nomes_meses = {
    1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
    7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
}
vendas_por_mes['mes_nome'] = vendas_por_mes['mes_transacao'].map(nomes_meses)

# Cria os subplots (dois gráficos um ao lado do outro)
fig, axes = plt.subplots(1, 2, figsize=(18, 7)) # 1 linha, 2 colunas

# Gráfico de Quantidade por Mês
sns.barplot(x='mes_nome', y='total_quantidade', data=vendas_por_mes, palette='Blues', ax=axes[0])
axes[0].set_title('Quantidade Total Vendida por Mês', fontsize=16)
axes[0].set_xlabel('Mês', fontsize=12)
axes[0].set_ylabel('Quantidade Total', fontsize=12)

# Gráfico de Lucro Bruto por Mês
sns.barplot(x='mes_nome', y='total_lucro_bruto', data=vendas_por_mes, palette='Greens', ax=axes[1])
axes[1].set_title('Lucro Bruto Total por Mês', fontsize=16)
axes[1].set_xlabel('Mês', fontsize=12)
axes[1].set_ylabel('Lucro Bruto Total', fontsize=12)

plt.tight_layout()
plt.show()

# Certifique-se de que 'data_transacao' está em formato datetime
df_tratado['data_transacao'] = pd.to_datetime(df_tratado['data_transacao'])

# Extrai o dia da semana (0=Segunda, 6=Domingo)
df_tratado['dia_semana'] = df_tratado['data_transacao'].dt.dayofweek

# Agrupa por dia da semana e soma a quantidade e o lucro bruto
vendas_por_dia = df_tratado.groupby('dia_semana').agg(
    total_quantidade=('quantidade', 'sum'),
    total_lucro_bruto=('lucro_bruto', 'sum')
).reset_index()

# Mapeia os números dos dias da semana para nomes
nomes_dias = {
    0: 'Segunda', 1: 'Terça', 2: 'Quarta', 3: 'Quinta',
    4: 'Sexta', 5: 'Sábado', 6: 'Domingo'
}
vendas_por_dia['dia_nome'] = vendas_por_dia['dia_semana'].map(nomes_dias)

# Cria os subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Gráfico de Quantidade por Dia da Semana
sns.barplot(x='dia_nome', y='total_quantidade', data=vendas_por_dia, palette='Purples', ax=axes[0])
axes[0].set_title('Quantidade Total Vendida por Dia da Semana', fontsize=16)
axes[0].set_xlabel('Dia da Semana', fontsize=12)
axes[0].set_ylabel('Quantidade Total', fontsize=12)

# Gráfico de Lucro Bruto por Dia da Semana
sns.barplot(x='dia_nome', y='total_lucro_bruto', data=vendas_por_dia, palette='Oranges', ax=axes[1])
axes[1].set_title('Lucro Bruto Total por Dia da Semana', fontsize=16)
axes[1].set_xlabel('Dia da Semana', fontsize=12)
axes[1].set_ylabel('Lucro Bruto Total', fontsize=12)

plt.tight_layout()
plt.show()

# Agrupa por categoria de PDV e conta as ocorrências (ou soma algo, como quantidade)
top_categorias_pdv = df_tratado['categoria_pdv'].value_counts().head(10)

plt.figure(figsize=(12, 7))
sns.barplot(x=top_categorias_pdv.index, y=top_categorias_pdv.values, palette='coolwarm')
plt.title('Top 10 Categorias de PDV (Estabelecimento) por Contagem de Transações', fontsize=16)
plt.xlabel('Categoria do PDV', fontsize=12)
plt.ylabel('Número de Transações', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
















import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ==========================================
# STEP 1: Additional Data Quality Checks
# ==========================================
print("Checking for outliers in numerical columns...")
numeric_cols = ['quantidade', 'valor_bruto', 'valor_liquido', 'lucro_bruto', 'desconto', 'impostos']

for col in numeric_cols:
    q1 = df_tratado[col].quantile(0.25)
    q3 = df_tratado[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df_tratado[(df_tratado[col] < lower_bound) | (df_tratado[col] > upper_bound)]
    print(f"Column {col}: Found {len(outliers)} outliers ({len(outliers)/len(df_tratado)*100:.2f}%)")

# Check for inconsistencies (negative values where inappropriate)
print("\nChecking for negative values in key columns...")
for col in ['quantidade', 'valor_bruto', 'valor_liquido']:
    negatives = df_tratado[df_tratado[col] < 0]
    print(f"Column {col}: Found {len(negatives)} negative values")

# Verify data types
print("\nCurrent data types:")
print(df_tratado.dtypes)

# ==========================================
# STEP 2: Fix Data Issues
# ==========================================
print("\nFixing data issues...")

# Handle negative quantities (either drop or investigate)
if len(df_tratado[df_tratado['quantidade'] < 0]) > 0:
    print(f"Removing {len(df_tratado[df_tratado['quantidade'] < 0])} rows with negative quantities")
    df_tratado = df_tratado[df_tratado['quantidade'] >= 0]

# Ensure proper date types
df_tratado['data_transacao'] = pd.to_datetime(df_tratado['data_transacao'])
df_tratado['data_referencia'] = pd.to_datetime(df_tratado['data_referencia'])

# Save this cleaned version before feature engineering
df_limpo = df_tratado.copy()
df_limpo.to_parquet('./Datasets/dados_limpos.parquet')
print(f"Clean dataset saved with {df_limpo.shape[0]} rows and {df_limpo.shape[1]} columns")

# ==========================================
# STEP 3: Create Time Features
# ==========================================
print("\nCreating time features...")

# Extract time components
df_tratado['ano'] = df_tratado['data_transacao'].dt.year
df_tratado['mes'] = df_tratado['data_transacao'].dt.month
df_tratado['dia'] = df_tratado['data_transacao'].dt.day
df_tratado['dia_semana'] = df_tratado['data_transacao'].dt.dayofweek
df_tratado['semana_ano'] = df_tratado['data_transacao'].dt.isocalendar().week
try:
    df_tratado['hora'] = df_tratado['data_transacao'].dt.hour
except:
    print("Time information not available in transaction data")
    
# Create weekend flag
df_tratado['is_weekend'] = df_tratado['dia_semana'].isin([5, 6]).astype(int)

# Create month-end flag
df_tratado['is_month_end'] = df_tratado['data_transacao'].dt.is_month_end.astype(int)

# ==========================================
# STEP 4: Create Derived Business Metrics
# ==========================================
print("\nCreating derived business metrics...")

# Calculate profit margin
df_tratado['margem_lucro'] = df_tratado['lucro_bruto'] / df_tratado['valor_bruto']
df_tratado['margem_lucro'] = df_tratado['margem_lucro'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Calculate discount percentage
df_tratado['pct_desconto'] = df_tratado['desconto'] / df_tratado['valor_bruto']
df_tratado['pct_desconto'] = df_tratado['pct_desconto'].replace([np.inf, -np.inf], np.nan).fillna(0)

# ==========================================
# STEP 5: Create Aggregated Features
# ==========================================
print("\nCreating aggregated features...")

# Product popularity metrics
product_popularity = df_tratado.groupby('id_produto')['quantidade'].sum().reset_index()
product_popularity.columns = ['id_produto', 'total_quantidade_produto']
df_tratado = pd.merge(df_tratado, product_popularity, on='id_produto', how='left')

# PDV transaction volume
pdv_volume = df_tratado.groupby('id_loja').size().reset_index()
pdv_volume.columns = ['id_loja', 'total_transacoes_pdv']
df_tratado = pd.merge(df_tratado, pdv_volume, on='id_loja', how='left')

# Category performance
category_performance = df_tratado.groupby('categoria')['margem_lucro'].mean().reset_index()
category_performance.columns = ['categoria', 'margem_media_categoria']
df_tratado = pd.merge(df_tratado, category_performance, on='categoria', how='left')

# ==========================================
# STEP 6: Save Dataset with Features
# ==========================================
print("\nSaving dataset with engineered features...")
df_tratado.to_parquet('./Datasets/dados_com_features.parquet')
print(f"Feature-engineered dataset saved with {df_tratado.shape[0]} rows and {df_tratado.shape[1]} columns")

print("\nData preparation complete. Ready for modeling!")

# Optional: Display summary of created features
print("\nFeatures created:")
new_features = [col for col in df_tratado.columns if col not in df_limpo.columns]
for feature in new_features:
    print(f"- {feature}")