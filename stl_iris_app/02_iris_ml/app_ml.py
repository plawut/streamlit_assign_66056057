import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

iris_df = pd.read_csv('iris.csv')
iris_df.dropna(inplace=True)
output = iris_df['variety']
features = iris_df[["sepal.length","sepal.width","petal.length","petal.width"]]
features = pd.get_dummies(features)
output,uniques = pd.factorize(output)

X_train, X_test, y_train, y_test = train_test_split(features,output,test_size=0.8)
rfc_model = RandomForestClassifier(random_state=15)
rfc_model.fit(X_train,y_train)
y_pred = rfc_model.predict(X_test)
score = accuracy_score(y_pred,y_test)
print(f'model accuracy score: {score}')

rfc_pickle = open('random_forest_iris.pickle','wb')
pickle.dump(rfc_model,rfc_pickle)
rfc_pickle.close()
output_pickle = open('output_iris.pickle','wb')
pickle.dump(uniques,output_pickle)
output_pickle.close()

fig,ax = plt.subplots()
ax = sns.barplot(x = rfc_model.feature_importances_,  y= features.columns)
plt.title('Which feature in iris dataset are the most important for variety prediction ??')
plt.xlabel('Important')
plt.ylabel('Features')
plt.tight_layout()
fig.savefig('feature_importance.png')

#%%
