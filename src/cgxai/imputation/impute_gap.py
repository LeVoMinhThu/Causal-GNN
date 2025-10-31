def impute_gap(df, columns, weights=None):
    df=df.copy()
    for col in columns:
        cd=df[col].ffill().bfill()
        br=df[col].fillna(df[col].rolling(5, min_periods=1).mean())
        w1,w2=(0.62,0.38) if not weights or col not in weights else weights[col]
        df[col]=w1*cd + w2*br
    return df
