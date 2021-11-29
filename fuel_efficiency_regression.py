import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


import numpy as np

import pandas as pd

np.set_printoptions(precision=3, suppress=True)

mpg_data = pd.read_csv('auto-mpg.csv.gz', sep='|')

print("Fuel efficiency input data:")
print(mpg_data)

print("Fuel efficiency input data info:")
print(mpg_data.info())

print("Origin column distinct values:", mpg_data['Origin'].unique())

mpg_data = mpg_data.drop('Car Name', axis=1)

sns.pairplot(mpg_data[['MPG', 'Cylinders', 'Displacement', 'Weight', 'Horsepower']], diag_kind='kde')
plt.show()

corr = mpg_data.corr()
sns.heatmap(corr, annot=True)
plt.show()

mpg_data_labels = mpg_data['MPG'].copy().to_numpy()
mpg_data_features = mpg_data.drop('MPG', axis=1)


num_tr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

cat_tr_pipeline = Pipeline([
        ('ordinal_encoder', OrdinalEncoder()),
        ('one_hot_encoder', OneHotEncoder()),
    ])


num_attribs = [col for col in mpg_data_features.columns if col != 'Origin']
cat_attribs = ['Origin']

full_pipeline = ColumnTransformer([
        ("num_tr_pipeline", num_tr_pipeline, num_attribs),
        ("cat_tr_pipeline", cat_tr_pipeline, cat_attribs),
    ])

mpg_transformed = full_pipeline.fit_transform(mpg_data_features)

mpg_train_data, mpg_test_data, mpg_train_labels, mpg_test_labels = train_test_split(mpg_transformed, mpg_data_labels, test_size=0.3, random_state=0)


lin_reg = LinearRegression()
lin_reg.fit(mpg_train_data, mpg_train_labels)
mpg_test_predicted = lin_reg.predict(mpg_test_data)
lin_reg_rmse = np.sqrt(mean_squared_error(mpg_test_labels, mpg_test_predicted, squared=True))
print("Linear Regression RMSE: ", lin_reg_rmse)

forest_reg = RandomForestRegressor(n_estimators=50, random_state=0)
forest_reg.fit(mpg_train_data, mpg_train_labels)
mpg_test_predicted = forest_reg.predict(mpg_test_data)
forest_reg_rmse = np.sqrt(mean_squared_error(mpg_test_predicted, mpg_test_labels, squared=True))
print("Random Forest Regression RMSE: ", forest_reg_rmse)

