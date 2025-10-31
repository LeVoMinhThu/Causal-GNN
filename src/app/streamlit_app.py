import os, streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt
from src.cgxai.utils.io import prepare_sample
from src.cgxai.pipeline.evaluate import evaluate_demo
from src.cgxai.pipeline.infer import infer_demo
from src.cgxai.xai.shap_utils import simple_shap_like
st.set_page_config(page_title='Causal GNN XAI – Fraud Monitor', layout='wide')
st.title('Causal GNN with XAI – Stock Market Fraud Monitor (Demo)')
if st.sidebar.button('Prepare sample data'):
    prepare_sample()
tab1, tab2, tab3 = st.tabs(['Overview', 'Firm Detail', 'XAI – Feature Impact'])
with tab1:
    st.subheader('Quick Metrics (Demo)')
    m=evaluate_demo(); c1,c2,c3=st.columns(3)
    c1.metric('AUROC', f"{m['auroc']:.3f}"); c2.metric('AUPRC', f"{m['auprc']:.3f}")
    k=[x for x in m.keys() if x.startswith('f1@')][0]; c3.metric(k, f"{m[k]:.3f}")
    st.subheader('Recent Risk Flags (Demo)')
    st.dataframe(pd.DataFrame(infer_demo()))
with tab2:
    st.subheader('Load Combined CSV')
    p='data/interim/combined.csv'
    if os.path.exists(p):
        df=pd.read_csv(p); st.write(f'Rows: {len(df)}'); st.dataframe(df.head(20))
    else:
        st.info("Click 'Prepare sample data' to create data/interim/combined.csv")
with tab3:
    st.subheader('Global Feature Impact (Toy)')
    features=np.array([1.0,3.2,0.5,2.1,0.9]); scores=simple_shap_like(features)
    import matplotlib.pyplot as plt
    fig=plt.figure(); plt.bar(range(len(scores)), scores)
    plt.xticks(range(len(scores)), ['Vol','CAR','Delay','Insider','BondGap']); plt.ylabel('Impact weight')
    st.pyplot(fig)
