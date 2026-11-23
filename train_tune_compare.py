"""Task 04: supervised model comparison, CV tuning, evaluation and serialization."""
from pathlib import Path
import json, joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report, RocCurveDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"data"/"synthetic_fraud_classification.csv")
X=df.drop(columns=["fraud_flag","record_id"])
y=df["fraud_flag"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=404,stratify=y)
num=X.select_dtypes(include="number").columns.tolist()
cat=X.select_dtypes(exclude="number").columns.tolist()
prep=ColumnTransformer([
 ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
 ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)
])
models={
 "LogisticRegression":(LogisticRegression(max_iter=1500,class_weight="balanced"),{"model__C":[.1,1,10]}),
 "DecisionTree":(DecisionTreeClassifier(class_weight="balanced",random_state=404),{"model__max_depth":[3,6,None],"model__min_samples_leaf":[2,5]}),
 "RandomForest":(RandomForestClassifier(class_weight="balanced",random_state=404,n_jobs=-1),{"model__n_estimators":[100,200],"model__max_depth":[None,8],"model__min_samples_leaf":[1,3]}),
 "SVC":(SVC(probability=True,class_weight="balanced",random_state=404),{"model__C":[.5,1,2],"model__kernel":["rbf","linear"]})
}
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=404)
results=[]
best_name=None; best_score=-1; best_model=None; curves=[]
for name,(est,param_grid) in models.items():
 pipe=Pipeline([("prep",prep),("model",est)])
 search=GridSearchCV(pipe,param_grid,scoring="roc_auc",cv=cv,n_jobs=-1,refit=True)
 search.fit(X_train,y_train)
 fitted=search.best_estimator_
 pred=fitted.predict(X_test)
 proba=fitted.predict_proba(X_test)[:,1]
 score=roc_auc_score(y_test,proba)
 results.append({"model":name,"best_cv_roc_auc":float(search.best_score_),"test_precision":float(precision_score(y_test,pred,zero_division=0)),"test_recall":float(recall_score(y_test,pred,zero_division=0)),"test_f1":float(f1_score(y_test,pred,zero_division=0)),"test_roc_auc":float(score),"best_params":search.best_params_})
 curves.append((name,y_test,proba))
 print("\\n",name,"best params:",search.best_params_)
 print(classification_report(y_test,pred,zero_division=0))
 if search.best_score_>best_score:
  best_score=search.best_score_;best_name=name;best_model=fitted
OUT=ROOT/"reports";OUT.mkdir(exist_ok=True)
pd.DataFrame(results).sort_values("best_cv_roc_auc",ascending=False).to_csv(OUT/"model_comparison.csv",index=False)
with open(OUT/"model_comparison.json","w") as f: json.dump(results,f,indent=2)
print("\\nChampion selected by CV ROC-AUC:",best_name,"score:",best_score)
champ_pred=best_model.predict(X_test)
pd.DataFrame(confusion_matrix(y_test,champ_pred),index=["Actual 0","Actual 1"],columns=["Predicted 0","Predicted 1"]).to_csv(OUT/"champion_confusion_matrix.csv")
fig,ax=plt.subplots()
for name,yt,pr in curves:
 RocCurveDisplay.from_predictions(yt,pr,name=name,ax=ax)
ax.set_title("ROC Curves — Synthetic Test Set")
fig.tight_layout();fig.savefig(OUT/"roc_curves.png",dpi=160)
MODELS=ROOT/"models";MODELS.mkdir(exist_ok=True)
joblib.dump(best_model,MODELS/"champion_model.joblib")
print("Saved model:",MODELS/"champion_model.joblib")
print("Warning: synthetic metrics are not deployment evidence.")
