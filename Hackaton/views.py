import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

def create_output_dir():
    """Create output directory for plots if it doesn't exist"""
    if not os.path.exists('plots'):
        os.makedirs('plots')

def load_data(file_path='wrangler_engineered.csv'):
    """Load the engineered data"""
    df = pd.read_csv(file_path)
    # Convert date columns to datetime
    if 'data_transacao' in df.columns:
        df['data_transacao'] = pd.to_datetime(df['data_transacao'])
    if 'data_referencia' in df.columns:
        df['data_referencia'] = pd.to_datetime(df['data_referencia'])
    return df

def basic_statistics(df):
    """Display basic statistics of the dataset"""
    print("==================================================")
    print("ESTATÍSTICAS BÁSICAS")
    print("==================================================")
    
    # Dataset dimensions
    print(f"Número de registros: {df.shape[0]}")
    print(f"Número de variáveis: {df.shape[1]}\n")
    
    # Data types
    print("Tipos de dados:")
    print(df.dtypes)
    print()
    
    # Summary statistics for numerical columns
    print("Estatísticas descritivas:")
    print(df.describe().T)
    print()
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("Valores ausentes:")
        print(missing[missing > 0])
    else:
        print("Não há valores ausentes no dataset.")
    print()

def temporal_analysis(df):
    """Analyze temporal patterns in the data"""
    print("==================================================")
    print("ANÁLISE TEMPORAL")
    print("==================================================")
    
    if 'data_transacao' not in df.columns:
        print("Coluna 'data_transacao' não encontrada. Pulando análise temporal.")
        return
    
    # Aggregate by date
    daily_transactions = df.groupby(df['data_transacao'].dt.date).size()
    daily_value = df.groupby(df['data_transacao'].dt.date)['valor_liquido'].sum()
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot number of transactions over time
    axes[0].plot(daily_transactions.index, daily_transactions.values, 
                color='royalblue', linewidth=1.5)
    axes[0].set_title('Número de Transações por Dia')
    axes[0].set_ylabel('Número de Transações')
    axes[0].grid(True, alpha=0.3)
    
    # Plot total value over time
    axes[1].plot(daily_value.index, daily_value.values, 
                color='mediumseagreen', linewidth=1.5)
    axes[1].set_title('Valor Líquido Total por Dia')
    axes[1].set_ylabel('Valor Líquido Total')
    axes[1].set_xlabel('Data')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/temporal_analysis.png', dpi=300)
    plt.close()
    
    # Monthly aggregation
    if 'mes' in df.columns and 'ano' in df.columns:
        df['year_month'] = df['ano'].astype(str) + '-' + df['mes'].astype(str).str.zfill(2)
        monthly = df.groupby('year_month').agg(
            transactions=('valor_liquido', 'count'),
            total_value=('valor_liquido', 'sum'),
            avg_value=('valor_liquido', 'mean')
        )
        
        print("Estatísticas mensais:")
        print(monthly)
        print()
    
    # Day of week analysis
    if 'dia_da_semana' in df.columns:
        dow_map = {0:'Segunda', 1:'Terça', 2:'Quarta', 3:'Quinta', 
                  4:'Sexta', 5:'Sábado', 6:'Domingo'}
        
        df_dow = df.copy()
        df_dow['dia_semana_nome'] = df_dow['dia_da_semana'].map(dow_map)
        
        dow_stats = df_dow.groupby('dia_semana_nome').agg(
            transactions=('valor_liquido', 'count'),
            avg_value=('valor_liquido', 'mean'),
            total_value=('valor_liquido', 'sum')
        )
        
        # Reorder days of week
        dow_order = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        dow_stats = dow_stats.reindex(dow_order)
        
        print("Estatísticas por dia da semana:")
        print(dow_stats)
        print()
        
        # Visualize weekday patterns
        plt.figure(figsize=(12, 6))
        sns.barplot(x=dow_stats.index, y=dow_stats['transactions'], color='cornflowerblue')
        plt.title('Número de Transações por Dia da Semana')
        plt.ylabel('Número de Transações')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/weekday_transactions.png', dpi=300)
        plt.close()

def distributor_analysis(df):
    """Analyze patterns by distributor"""
    print("==================================================")
    print("ANÁLISE POR DISTRIBUIDOR")
    print("==================================================")
    
    if 'id_distribuidor' not in df.columns:
        print("Coluna 'id_distribuidor' não encontrada. Pulando análise por distribuidor.")
        return
    
    # Group by distributor
    dist_stats = df.groupby('id_distribuidor').agg(
        transactions=('valor_liquido', 'count'),
        total_value=('valor_liquido', 'sum'),
        avg_value=('valor_liquido', 'mean'),
        avg_profit=('lucro_bruto', 'mean'),
        profit_margin=('margem', 'mean')
    )
    
    print("Estatísticas por distribuidor:")
    print(dist_stats)
    print()
    
    # Visualize distributor patterns
    plt.figure(figsize=(14, 8))
    
    # Create a boxplot of valor_liquido by distributor
    sns.boxplot(x='id_distribuidor', y='valor_liquido', data=df, palette='viridis')
    plt.title('Distribuição do Valor Líquido por Distribuidor')
    plt.xlabel('ID do Distribuidor')
    plt.ylabel('Valor Líquido')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/distributor_boxplot.png', dpi=300)
    plt.close()
    
    # Create a bar chart comparing average values
    plt.figure(figsize=(12, 6))
    dist_stats['avg_value'].plot(kind='bar', color='steelblue')
    plt.title('Valor Médio por Distribuidor')
    plt.xlabel('ID do Distribuidor')
    plt.ylabel('Valor Médio')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/distributor_avg_value.png', dpi=300)
    plt.close()

def correlation_analysis(df):
    """Analyze correlations between variables"""
    print("==================================================")
    print("ANÁLISE DE CORRELAÇÃO")
    print("==================================================")
    
    # Select numerical columns for correlation analysis
    numeric_cols = ['quantidade', 'valor_bruto', 'valor_liquido', 'lucro_bruto', 
                    'desconto', 'impostos', 'outlier_score']
    
    # Add engineered features if they exist
    additional_cols = ['margem', 'pct_desconto', 'pct_imposto', 'valor_medio_item']
    numeric_cols.extend([col for col in additional_cols if col in df.columns])
    
    # Correlation matrix
    corr = df[numeric_cols].corr()
    print("Matriz de correlação:")
    print(corr)
    print()
    
    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, 
                mask=mask, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5)
    
    plt.title('Matriz de Correlação', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png', dpi=300)
    plt.close()
    
    # Scatter plot for key variables
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    sns.scatterplot(x='valor_bruto', y='lucro_bruto', 
                    data=df.sample(min(5000, len(df))), 
                    alpha=0.6, ax=axes[0, 0])
    axes[0, 0].set_title('Lucro Bruto vs Valor Bruto')
    axes[0, 0].grid(alpha=0.3)
    
    sns.scatterplot(x='quantidade', y='valor_bruto', 
                    data=df.sample(min(5000, len(df))), 
                    alpha=0.6, ax=axes[0, 1])
    axes[0, 1].set_title('Valor Bruto vs Quantidade')
    axes[0, 1].grid(alpha=0.3)
    
    sns.scatterplot(x='desconto', y='valor_liquido', 
                    data=df.sample(min(5000, len(df))), 
                    alpha=0.6, ax=axes[1, 0])
    axes[1, 0].set_title('Valor Líquido vs Desconto')
    axes[1, 0].grid(alpha=0.3)
    
    if 'margem' in df.columns:
        sns.scatterplot(x='margem', y='valor_liquido', 
                        data=df.sample(min(5000, len(df))), 
                        alpha=0.6, ax=axes[1, 1])
        axes[1, 1].set_title('Valor Líquido vs Margem')
        axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/scatter_plots.png', dpi=300)
    plt.close()

def outlier_analysis(df):
    """Analyze outliers in the data"""
    print("==================================================")
    print("ANÁLISE DE OUTLIERS")
    print("==================================================")
    
    if 'outlier_flag' not in df.columns or 'outlier_score' not in df.columns:
        print("Colunas de outliers não encontradas. Pulando análise de outliers.")
        return
    
    # Outlier distribution
    outlier_stats = df.groupby('outlier_flag').agg(
        count=('outlier_flag', 'count'),
        avg_score=('outlier_score', 'mean'),
        avg_valor=('valor_liquido', 'mean')
    )
    
    print("Estatísticas de outliers:")
    print(outlier_stats)
    print()
    
    print(f"Percentual de outliers: {df['outlier_flag'].mean()*100:.2f}%")
    print()
    
    # Plot outlier score distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['outlier_score'], bins=50, kde=True, color='purple')
    plt.title('Distribuição do Outlier Score')
    plt.xlabel('Outlier Score')
    plt.ylabel('Frequência')
    plt.grid(alpha=0.3)
    plt.savefig('plots/outlier_score_distribution.png', dpi=300)
    plt.close()
    
    # Scatter plot of outlier scores
    plt.figure(figsize=(14, 6))
    plt.scatter(range(len(df)), df['outlier_score'], 
                alpha=0.5, s=3, c=df['outlier_flag'], cmap='viridis')
    plt.title('Outlier Score para cada Transação')
    plt.xlabel('Índice da Transação')
    plt.ylabel('Outlier Score')
    plt.colorbar(label='outlier_flag')
    plt.grid(alpha=0.3)
    plt.savefig('plots/outlier_scatter.png', dpi=300)
    plt.close()

def dimensionality_reduction(df):
    """Perform PCA for visualization"""
    print("==================================================")
    print("REDUÇÃO DE DIMENSIONALIDADE (PCA)")
    print("==================================================")
    
    # Select numerical columns for PCA
    numeric_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
    
    # Exclude ID columns, date columns, and categorical columns
    exclude_cols = ['id_loja_interna', 'id_produto_interno', 'id_distribuidor',
                   'ano', 'mes', 'dia', 'dia_da_semana']
    
    numeric_cols = [col for col in numeric_cols 
                   if col not in exclude_cols and not col.startswith('distribuidor_')]
    
    # Run PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[numeric_cols])
    
    print(f"Variância explicada pelos dois primeiros componentes: {sum(pca.explained_variance_ratio_):.2f}")
    print()
    
    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['id_distribuidor'] = df['id_distribuidor']
    
    if 'outlier_flag' in df.columns:
        pca_df['outlier_flag'] = df['outlier_flag']
    
    # Plot PCA results by distributor
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='id_distribuidor', 
                    data=pca_df.sample(min(5000, len(pca_df))), 
                    palette='viridis', alpha=0.7)
    plt.title('PCA - Agrupamento por Distribuidor')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
    plt.legend(title='Distribuidor')
    plt.grid(alpha=0.3)
    plt.savefig('plots/pca_distributor.png', dpi=300)
    plt.close()
    
    # Plot PCA results by outlier flag if available
    if 'outlier_flag' in df.columns:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='PC1', y='PC2', hue='outlier_flag', 
                        data=pca_df.sample(min(5000, len(pca_df))), 
                        palette=['blue', 'red'], alpha=0.7)
        plt.title('PCA - Identificação de Outliers')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        plt.legend(title='Outlier')
        plt.grid(alpha=0.3)
        plt.savefig('plots/pca_outliers.png', dpi=300)
        plt.close()

def main():
    # Create output directory
    create_output_dir()
    
    # Load data
    df = load_data()
    
    print("==================================================")
    print("ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("==================================================")
    print(f"Número de registros: {df.shape[0]}")
    print(f"Número de colunas: {df.shape[1]}")
    print()
    
    # Perform analyses
    basic_statistics(df)
    temporal_analysis(df)
    distributor_analysis(df)
    correlation_analysis(df)
    outlier_analysis(df)
    dimensionality_reduction(df)
    
    print("==================================================")
    print("ANÁLISE EXPLORATÓRIA CONCLUÍDA")
    print("==================================================")
    print("Todos os gráficos foram salvos no diretório 'plots/'")

if __name__ == "__main__":
    main()