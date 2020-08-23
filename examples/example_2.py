from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import datasets

import pandas as pd

from nonparam.classification import NonParamGapClf

x, y = datasets.load_iris(return_X_y=True)
features = [i for i in range(4)]

df = pd.DataFrame(data=x, columns=features)
df['target'] = y

noclf = NonParamGapClf(
    k=13,
    kernel='parabolic',
    con_features=features,
    cat_features=[]
)

x, y = df[features], df['target']
x_tr, x_ts, y_tr, y_ts = train_test_split(
    x, y,
    test_size=0.2,
    random_state=0
)

noclf.fit(x_tr, y_tr)
y_pred_proba = noclf.predict_proba(x_ts)

y_pred = []
for _, row in y_pred_proba.iterrows():
    tmp = sorted(row.to_dict().items(), key=lambda x: x[1], reverse=True)[0]
    y_pred.append(tmp[0])

f1 = f1_score(y_ts, y_pred, average='macro')

print(f1)
