import numpy as np

import pandas as pd

import streamlit as st

import plotly.express as px



st.set_page_config(page_title="Matriz de Rigidez Ortotrópica", layout="wide")

st.title("📊 Cálculo das Matrizes [C] e [S] — Material Ortotrópico")



# ========================

# Entradas

# ========================

rho = st.number_input("Densidade ρ (kg/m³)", value=1028.972)

a = np.deg2rad(st.slider("Ângulo entre direções (°)", 0.0, 90.0, 45.0))



st.markdown("#### 📏 Dados Experimentais — Distância e Tempo de Propagação")

st.caption("Edite as amostras diretamente. Velocidades serão calculadas automaticamente.")



# ------------------------

# Tabela editável

# ------------------------

dados_iniciais = {

    "Amostra": [

        "LL", "RR", "TT", "LR", "LT", "RL", "RT", "TL", "TR",

        "LT1", "LT2", "RL1", "RL2", "RT1", "RT2"

    ],

    "Distância (cm)": [4.681, 4.608, 4.593, 4.681, 4.681, 4.608,

        4.608, 4.593, 4.593, 4.374, 4.373, 4.366, 4.368, 4.341, 4.332

    ],                    

    "Tempo (µs)": [8.67, 20.85, 23.45, 39.47, 39.37, 36.26, 51.76,

        36.71, 52.19, 28.33, 28.87, 30.98, 28.7 ,35.45, 37.4

    ]              

}



df = pd.DataFrame(dados_iniciais)

df_edit = st.data_editor(df, num_rows="fixed", use_container_width=True)



# ------------------------

# Cálculo das velocidades

# ------------------------

df_edit["Velocidade (m/s)"] = (

    df_edit["Distância (cm)"] * 1e-2 / (df_edit["Tempo (µs)"] * 1e-6)

).round(2)



st.dataframe(df_edit, use_container_width=True)



# Converter para dicionário de velocidades

vel = dict(zip(df_edit["Amostra"], df_edit["Velocidade (m/s)"]))



# ------------------------

# Médias das velocidades fora das direções principais

# ------------------------

V_12a = (vel["RL1"] + vel["RL2"]) / 2

V_13a = (vel["LT1"] + vel["LT2"]) / 2

V_23a = (vel["RT1"] + vel["RT2"]) / 2



# ------------------------

# Cálculo das matrizes

# ------------------------

C_11 = rho * vel["LL"]**2

C_22 = rho * vel["RR"]**2

C_33 = rho * vel["TT"]**2

C_44 = rho * ((vel["RT"] + vel["TR"]) / 2)**2

C_55 = rho * ((vel["LT"] + vel["TL"]) / 2)**2

C_66 = rho * ((vel["LR"] + vel["RL"]) / 2)**2



n1, n2, n3 = np.sin(a), np.cos(a), np.cos(a)

C_12 = (((C_11*n1**2 + C_66*n2**2 - rho*V_12a**2) *

         (C_66*n1**2 + C_22*n2**2 - rho*V_12a**2))**0.5 - C_66*n1*n2) / (n1*n2)

C_13 = (((C_11*n1**2 + C_55*n3**2 - rho*V_13a**2) *

         (C_55*n1**2 + C_33*n3**2 - rho*V_13a**2))**0.5 - C_55*n1*n3) / (n1*n3)

C_23 = (((C_22*n2**2 + C_44*n3**2 - rho*V_23a**2) *

         (C_44*n2**2 + C_33*n3**2 - rho*V_23a**2))**0.5 - C_44*n2*n3) / (n2*n3)



# Matrizes

C = np.array([

    [C_11, C_12, C_13, 0, 0, 0],

    [C_12, C_22, C_23, 0, 0, 0],

    [C_13, C_23, C_33, 0, 0, 0],

    [0, 0, 0, C_44, 0, 0],

    [0, 0, 0, 0, C_55, 0],

    [0, 0, 0, 0, 0, C_66]

]) / 1e6  # MPa



S = np.linalg.inv(C)



# ------------------------

# Propriedades

# ------------------------

E_L, E_R, E_T = 1/S[0,0]/1e3, 1/S[1,1]/1e3, 1/S[2,2]/1e3

G_RT, G_LT, G_LR = 1/S[3,3]/1e3, 1/S[4,4]/1e3, 1/S[5,5]/1e3



v_LR = -S[0,1]/S[0,0]

v_LT = -S[0,2]/S[0,0]

v_RL = -S[0,1]/S[1,1]

v_TL = -S[0,2]/S[2,2]

v_RT = -S[1,2]/S[1,1]

v_TR = -S[1,2]/S[2,2]



# ------------------------

# Gráfico: Módulos E e G

# ------------------------

props_df = pd.DataFrame({

    "Propriedade": ["E_L", "E_R", "E_T", "G_RT", "G_LT", "G_LR"],

    "Valor (GPa)": [E_L, E_R, E_T, G_RT, G_LT, G_LR],

    "Tipo": ["E","E","E","G","G","G"]

})



fig_props = px.bar(

    props_df, x="Propriedade", y="Valor (GPa)", color="Tipo",

    title="Módulos de Elasticidade e Cisalhamento", text="Valor (GPa)"

)

fig_props.update_layout(yaxis_title="GPa", xaxis_title="Propriedade")

st.plotly_chart(fig_props, use_container_width=True)



# ------------------------

# Gráfico: Coeficientes de Poisson

# ------------------------

poisson_df = pd.DataFrame({

    "Coeficiente": ["ν_LR", "ν_LT", "ν_RL", "ν_TL", "ν_RT", "ν_TR"],

    "Valor": [v_LR, v_LT, v_RL, v_TL, v_RT, v_TR]

})



fig_poisson = px.bar(

    poisson_df, x="Coeficiente", y="Valor", text="Valor",

    title="Coeficientes de Poisson", color="Coeficiente"

)

fig_poisson.update_layout(yaxis_title="Valor", xaxis_title="Coeficiente")

st.plotly_chart(fig_poisson, use_container_width=True)