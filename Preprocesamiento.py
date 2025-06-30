import pandas as pd

df2 = pd.read_csv("C:/Users/pelic/.cache/kagglehub/datasets/najzeko/steam-reviews-2021/versions/1/steam_reviews.csv")

def removeSp(x):
    return str(x).replace("\n", "").replace("\r", "").replace("\t", "")

df2['review']=df2['review'].apply(removeSp)

df2=df2[df2["language"]=="english"]

df2=df2.head(2000000)

df2.to_csv('steamDataF3.tsv', sep="\t")