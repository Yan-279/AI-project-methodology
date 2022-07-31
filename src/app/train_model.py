from pathlib import Path
from joblib import dump
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.app.preprocess import preprocess
DATA_DIR = Path(r'../../data/')
MODEL_DIR=Path(r'../../models/')
df_master= pd.read_csv(DATA_DIR/"train.csv",index_col='ID')

def build_model(data: pd.DataFrame):
    target = data['y']
    train = data.drop(['y'], axis=1)
    processedtrain= preprocess(train, MODEL_DIR)
    x_train , x_test , y_train , y_test = train_test_split(processedtrain,target,test_size=0.2)
    rfc = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    rfc.fit(x_train, y_train)
    dump(rfc,MODEL_DIR/'RFC.joblib')
    predict = rfc.predict(x_test)
    print(f"The accuracy score is : {accuracy_score(y_test, predict) * 100}%")
    print (classification_report(y_test,predict))
    return

build_model(df_master)