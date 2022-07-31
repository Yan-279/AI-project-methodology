import sys
sys.path.append('..')
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
from src.app.preprocess import preprocess

from pathlib import Path
DATA_DIR = Path(r'../../data/')
MODEL_DIR=Path(r'../../models/')
df_master= pd.read_csv(DATA_DIR/"train.csv",index_col='ID')

def predictions(data: pd.DataFrame):
    target = data['y']
    train = data.drop(['y'], axis=1)
    processedtrain = preprocess(train, MODEL_DIR)
    x_train, x_test, y_train, y_test = train_test_split(processedtrain, target, test_size=0.2)
    k = load(MODEL_DIR / 'RFC.joblib')
    predict = k.predict(x_test)
    # print(predict)
    newdf = pd.DataFrame(predict, columns=['y'])
    submissiondf = newdf.rename_axis('ID').reset_index()
    print(submissiondf.head())
    return submissiondf.head()