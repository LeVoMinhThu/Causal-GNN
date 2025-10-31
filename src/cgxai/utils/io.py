import os, argparse, pandas as pd
S1='data/raw/scc_thanhtra_extration.csv'
S2='data/raw/fsc_articles_extraction.csv'

def prepare_sample():
    os.makedirs('data/interim', exist_ok=True)
    df1=pd.read_csv(S1) if os.path.exists(S1) else pd.DataFrame()
    df2=pd.read_csv(S2) if os.path.exists(S2) else pd.DataFrame()
    df=pd.concat([df1,df2], ignore_index=True, sort=False)
    out='data/interim/combined.csv'; df.to_csv(out, index=False)
    print(f'Wrote {out} with {len(df)} rows')
