import streamlit as st
import pandas as pd
import numpy as np
import warnings

# Ignorar FutureWarning do Pandas/NumPy no Streamlit
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configurações Iniciais do Streamlit ---
st.set_page_config(layout="wide", page_title="Calculadora de Propriedades Elásticas Editável e Diagnóstica")
st.title("🔬 Análise Interativa de Propriedades Elásticas de Madeira")
st.write("Carregue seu CSV contendo **Distância (cm)** e **Tempo (μs)**. A velocidade será calculada. Edite os dados e visualize os resultados no formato de sua preferência (Tabela ou Matriz LaTeX).")

# Lista de todas as direções de velocidade necessárias
DIRECOES_NECESSARIAS = [
    'LL', 'RR', 'TT',
    'LR', 'RL', 'LT', 'TL', 'RT', 'TR',
    'RL2', 'RT1', 'RT2', 'TL1', 'TL2'
]

# --- Função Auxiliar: Geração de Matriz LaTeX (Mantida) ---
def to_latex_matrix(np_array, precision=2, is_flexibility=False):
    """
    Converte uma matriz numpy 6x6 em uma string de matriz LaTeX.
    """
    lines = []
    
    if is_flexibility:
        formatter = lambda x: f'{x:.{precision}e}'.replace('e', r' \cdot 10^{').replace('+', '').replace('-', '{-}') + '}'
    else:
        formatter = lambda x: f'{x:.{precision}f}'
        
    for row in np_array:
        formatted_row = [formatter(val) for val in row]
        lines.append(' & '.join(formatted_row) + r' \\')
        
    latex_string = r'\begin{bmatrix}' + '\n' + '\n'.join(lines) + '\n' + r'\end{bmatrix}'
    return latex_string


# --- Funções de Verificação e Cálculo (Ajustadas para st.table) ---

def verificar_dados_amostra(df_amostra):
    # (Mantida)
    direcoes_presentes = set(df_amostra['Direção'].unique())
    faltando_direcoes = [d for d in DIRECOES_NECESSARIAS if d not in direcoes_presentes]
    if df_amostra['Densidade (kg/m³)'].nunique() != 1:
        if 'Densidade (kg/m³)' not in [d for d in df_amostra['Direção'].unique()]:
            return faltando_direcoes + ["Densidade (valor inconsistente)"]
    return faltando_direcoes

def verificar_condicoes_de_ordem(C_matrix):
    # (Ajustada para retornar uma lista de dicionários para st.table)
    C11, C22, C33 = C_matrix[0, 0], C_matrix[1, 1], C_matrix[2, 2]
    C44, C55, C66 = C_matrix[3, 3], C_matrix[4, 4], C_matrix[5, 5]
    C12, C13, C23 = C_matrix[0, 1], C_matrix[0, 2], C_matrix[1, 2]

    diagnostico_list = [
        {
            'Condição': r'$\mathbf{C}_{ii} \text{ (Normal)}$',
            'Regra Esperada': r'$C_{11} > C_{22} > C_{33}$',
            'Valor da Condição': f'{C11:.2f} > {C22:.2f} > {C33:.2f}',
            'Conformidade': '✅ Atendida' if (C11 > C22 and C22 > C33) else '❌ Não Atendida'
        },
        {
            'Condição': r'$\mathbf{C}_{ij} \text{ (Cisalhamento)}$',
            'Regra Esperada': r'$C_{44} < C_{55} < C_{66}$',
            'Valor da Condição': f'{C44:.2f} < {C55:.2f} < {C66:.2f}',
            'Conformidade': '✅ Atendida' if (C66 > C55 and C55 > C44) else '❌ Não Atendida'
        },
        {
            'Condição': r'$\mathbf{C}_{ij} \text{ (Mútua)}$',
            'Regra Esperada': r'$C_{12} > C_{13} > C_{23}$',
            'Valor da Condição': f'{C12:.2f} > {C13:.2f} > {C23:.2f}',
            'Conformidade': '✅ Atendida' if (C12 > C13 and C13 > C23) else '❌ Não Atendida'
        }
    ]
        
    return diagnostico_list

@st.cache_data
def calcular_propriedades_elasticas(df_amostra, nome_amostra):
    
    rho_kg_m3 = df_amostra['Densidade (kg/m³)'].iloc[0]
    rho = rho_kg_m3
    velocidades = df_amostra.set_index('Direção')['Velocidade (m/s)'].to_dict()
    fator_conversao = 1e-6 # Pa para MPa

    # Cálculo da Matriz C (C_matrix)
    C11 = rho * (velocidades['LL']**2) * fator_conversao
    C22 = rho * (velocidades['RR']**2) * fator_conversao
    C33 = rho * (velocidades['TT']**2) * fator_conversao
    
    V6_puro = (velocidades['LR'] + velocidades['RL']) / 2
    V5_puro = (velocidades['LT'] + velocidades['TL']) / 2
    V4_puro = (velocidades['RT'] + velocidades['TR']) / 2
    
    C66 = rho * (V6_puro**2) * fator_conversao
    C55 = rho * (V5_puro**2) * fator_conversao
    C44 = rho * (V4_puro**2) * fator_conversao
    
    n1 = n2 = n3 = 0.7071067812
    n_quadrado = 0.5
    
    V_LR_45 = velocidades['RL2']
    rho_VLR_quadrado = rho * (V_LR_45**2) * fator_conversao
    I_12 = C11 * (n1**2) + C66 * (n2**2) - rho_VLR_quadrado
    II_12 = C66 * (n1**2) + C22 * (n2**2) - rho_VLR_quadrado 
    C12 = (np.sqrt(I_12 * II_12) / n_quadrado) - C66
    
    V_RT_45 = (velocidades['RT1'] + velocidades['RT2']) / 2
    rho_VRT_quadrado = rho * (V_RT_45**2) * fator_conversao
    I_23 = C22 * (n2**2) + C44 * (n2**2) - rho_VRT_quadrado
    II_23 = C44 * (n2**2) + C33 * (n3**2) - rho_VRT_quadrado
    C23 = (np.sqrt(I_23 * II_23) / n_quadrado) - C44

    V_LT_45 = (velocidades['TL1'] + velocidades['TL2']) / 2
    rho_VLT_quadrado = rho * (V_LT_45**2) * fator_conversao
    I_13 = C11 * (n1**2) + C55 * (n3**2) - rho_VLT_quadrado
    II_13 = C55 * (n3**2) + C33 * (n3**2) - rho_VLT_quadrado
    C13 = (np.sqrt(I_13 * II_13) / n_quadrado) - C55
    
    C_matrix = np.array([
        [C11, C12, C13, 0.0, 0.0, 0.0],
        [C12, C22, C23, 0.0, 0.0, 0.0],
        [C13, C23, C33, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, C44, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, C55, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, C66]
    ])

    S_matrix = np.linalg.inv(C_matrix)
    
    S11, S12, S13 = S_matrix[0, 0], S_matrix[0, 1], S_matrix[0, 2]
    S22, S23 = S_matrix[1, 1], S_matrix[1, 2]
    S33 = S_matrix[2, 2]
    S44, S55, S66 = S_matrix[3, 3], S_matrix[4, 4], S_matrix[5, 5]

    # 6. Cálculo das 9 Constantes Elásticas (E, G, nu)
    EL = 1 / S11; ER = 1 / S22; ET = 1 / S33
    GLR = 1 / S66; GLT = 1 / S55; GRT = 1 / S44
    
    nu_LR = -EL * S12; nu_LT = -EL * S13
    nu_RL = -ER * S12; nu_RT = -ER * S23
    nu_TL = -ET * S13; nu_TR = -ET * S23
    
    # --- NOVO: Formato Lista de Dicionários com LaTeX para st.table ---
    propriedades_list = [
        {"Constante": r'$E_L$', "Valor": f"{EL:.4f}", "Unidade": 'MPa'},
        {"Constante": r'$E_R$', "Valor": f"{ER:.4f}", "Unidade": 'MPa'},
        {"Constante": r'$E_T$', "Valor": f"{ET:.4f}", "Unidade": 'MPa'},
        {"Constante": r'$G_{LR}$', "Valor": f"{GLR:.4f}", "Unidade": 'MPa'},
        {"Constante": r'$G_{LT}$', "Valor": f"{GLT:.4f}", "Unidade": 'MPa'},
        {"Constante": r'$G_{RT}$', "Valor": f"{GRT:.4f}", "Unidade": 'MPa'},
        {"Constante": r'$\nu_{LR}$', "Valor": f"{nu_LR:.4f}", "Unidade": '-'},
        {"Constante": r'$\nu_{LT}$', "Valor": f"{nu_LT:.4f}", "Unidade": '-'},
        {"Constante": r'$\nu_{RL}$', "Valor": f"{nu_RL:.4f}", "Unidade": '-'},
        {"Constante": r'$\nu_{RT}$', "Valor": f"{nu_RT:.4f}", "Unidade": '-'},
        {"Constante": r'$\nu_{TL}$', "Valor": f"{nu_TL:.4f}", "Unidade": '-'},
        {"Constante": r'$\nu_{TR}$', "Valor": f"{nu_TR:.4f}", "Unidade": '-'}
    ]
    
    df_rigidez = pd.DataFrame(C_matrix, index=['C1', 'C2', 'C3', 'C4', 'C5', 'C6'], columns=['C1', 'C2', 'C3', 'C4', 'C5', 'C6'])
    df_flexibilidade = pd.DataFrame(S_matrix, index=['S1', 'S2', 'S3', 'S4', 'S5', 'S6'], columns=['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])
    
    # Chamada para a função que retorna lista de dicts
    diagnostico_list = verificar_condicoes_de_ordem(C_matrix)

    return propriedades_list, df_rigidez, df_flexibilidade, diagnostico_list, C_matrix, S_matrix


# --- SEÇÃO PRINCIPAL DO STREAMLIT ---
uploaded_file = st.sidebar.file_uploader("1. Arraste e solte o arquivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Carregar o CSV e pré-processar
        df_completo = pd.read_csv(uploaded_file, decimal=',', sep=',')
        colunas_esperadas = ['Amostra', 'Densidade (kg/m³)', 'Direção', 'Distância (cm)', 'Tempo (μs)']
        df_completo = df_completo[[col for col in colunas_esperadas if col in df_completo.columns]].copy()
        df_completo['Densidade (kg/m³)'] = pd.to_numeric(df_completo['Densidade (kg/m³)'], errors='coerce')
        df_completo['Distância (cm)'] = pd.to_numeric(df_completo['Distância (cm)'], errors='coerce')
        df_completo['Tempo (μs)'] = pd.to_numeric(df_completo['Tempo (μs)'], errors='coerce')
        
        # --- 2. EDITA O DATAFRAME COMPLETO ---
        st.header("2. Edite os Dados de Entrada")
        st.info("Ajuste Densidade, Distância (cm) e Tempo (μs).")
        
        df_editado = st.data_editor(df_completo, num_rows="dynamic", width='stretch',
            column_config={
                "Distância (cm)": st.column_config.NumberColumn(format="%.3f"),
                "Tempo (μs)": st.column_config.NumberColumn(format="%.2f"),
                "Densidade (kg/m³)": st.column_config.NumberColumn(format="%.2f")
            }
        )
        
        df_editado.dropna(subset=['Densidade (kg/m³)', 'Distância (cm)', 'Tempo (μs)'], inplace=True)
        df_editado['Velocidade (m/s)'] = (df_editado['Distância (cm)'] / df_editado['Tempo (μs)']) * 10000
        
        todas_amostras = sorted(df_editado['Amostra'].unique())
        
        # --- 5. CONTROLES NA BARRA LATERAL ---
        st.sidebar.markdown("---")
        amostras_selecionadas = st.sidebar.multiselect(
            "3. Selecione as Amostras para Análise:",
            options=todas_amostras,
            default=todas_amostras
        )
        use_latex = st.sidebar.checkbox("Visualizar Matrizes em formato LaTeX", value=False)

        if st.sidebar.button("4. Executar Análise"):
            if not amostras_selecionadas:
                st.warning("Por favor, selecione pelo menos uma amostra para análise.")
                st.stop()
            
            st.header(f"Resultados da Análise para {len(amostras_selecionadas)} Amostra(s)")
            tabs = st.tabs(amostras_selecionadas)
            
            for i, amostra in enumerate(amostras_selecionadas):
                df_amostra = df_editado[df_editado['Amostra'] == amostra].copy()
                
                with tabs[i]:
                    st.subheader(f"Amostra: {amostra}")
                    
                    if faltando := verificar_dados_amostra(df_amostra):
                        st.error("⚠️ Dados insuficientes para o cálculo ortotrópico completo.")
                        st.info(f"Amostra ignorada. Faltando: {', '.join(faltando)}")
                        continue 

                    try:
                        resultados = calcular_propriedades_elasticas(df_amostra, amostra)
                        propriedades_list, df_rigidez, df_flexibilidade, diagnostico_list, C_matrix, S_matrix = resultados

                        st.success("✅ Cálculo das 9 Constantes Elásticas concluído.")

                        # Relatório de Diagnóstico (AGORA USANDO st.table)
                        with st.expander("Diagnóstico de Conformidade da Rigidez", expanded=True):
                             st.markdown("**Verificação das Condições de Ordem da Madeira:**")
                             # Note: st.table renderiza o LaTeX, mas a estilização de fundo é perdida.
                             st.table(diagnostico_list)

                        # Resultados em Colunas
                        col_E, col_C = st.columns(2)
                        
                        with col_E:
                            st.markdown("##### Módulos e Coeficientes de Poisson")
                            # PROPRIEDADES: USANDO st.table
                            st.table(propriedades_list)

                        with col_C:
                            st.markdown("##### Matrizes de Rigidez e Flexibilidade")
                            
                            if use_latex:
                                # Opção LaTeX
                                latex_C_code = r"\mathbf{C}_{ij} \text{ [MPa]} = " + to_latex_matrix(C_matrix, precision=2)
                                with st.expander("Copiar Código LaTeX da Matriz de Rigidez ($C_{ij}$)", expanded=False):
                                    st.code(latex_C_code, language='latex')
                                st.latex(latex_C_code)
                                
                                latex_S_code = r"\mathbf{S}_{ij} \text{ [MPa}^{-1}\text{]} = " + to_latex_matrix(S_matrix, precision=2, is_flexibility=True)
                                with st.expander("Copiar Código LaTeX da Matriz de Flexibilidade ($S_{ij}$)", expanded=False):
                                    st.code(latex_S_code, language='latex')
                                st.latex(latex_S_code)
                                
                            else:
                                # Opção DataFrame (Padrão, sem renderização LaTeX)
                                st.markdown("###### Matriz de Rigidez ($C_{ij}$) [MPa]")
                                st.dataframe(df_rigidez.style.format(precision=2), width='stretch')

                                st.markdown("###### Matriz de Flexibilidade ($S_{ij}$) [$MPa^{-1}$]")
                                st.dataframe(df_flexibilidade.style.format('{:.2e}'), width='stretch')
                            
                    except np.linalg.LinAlgError:
                        st.error("Erro de Inversão de Matriz. Inconsistência grave nos dados de rigidez.")
                    except Exception as e:
                        st.error(f"Erro inesperado durante o cálculo: {e}")
                        
    except Exception as e:
        st.error("Erro fatal ao carregar ou processar o arquivo CSV. Verifique se as colunas estão corretas e o separador decimal é a vírgula (',').")
        st.exception(e)