# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn import svm
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict,GridSearchCV
from sklearn import preprocessing
import scipy.stats as stats
# from sklearn.decomposition import PCA

# Show all columns
pd.set_option('display.max_columns',None)

# Import data and basic insight:types
train = pd.read_csv('/Kaggle/houseprice/train.csv',index_col='Id')
test = pd.read_csv('/Kaggle/houseprice/test.csv',index_col='Id')
# all_data = pd.concat([train,test])
train.head()
# train.info()

# test.info()
# EDA:visualization;distribution;check outliers;Year
train.describe()
# sns.distplot(train['SalePrice'])
# plt.show()

train.drop(train[(train['LotArea'] > 70000) & (train['SalePrice'] < 400000)].index,inplace=True)
train.drop(train[train['LotFrontage'] > 300].index,inplace=True)

# Data preprocessing/cleaning

# test data dealing with:
# check: test[test[columns].isnull()]
# fill with mode
dealt_list1 = ['MSZoning','Utilities','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType']
for columns in dealt_list1:
	test[columns].fillna(test[columns].value_counts().idxmax(),inplace=True)
# KitchenQual 1458/1459
# test.groupby(['KitchenAbvGr'])['KitchenQual'].value_counts()
# fill with 0
dealt_list2 = ['BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']
for columns in dealt_list2:
	test[columns].fillna(0,inplace=True)

# train data dealing with
# Electrical(1459/1460)
train[train['Electrical'].isnull()] # find the NaN: id=1380
train[train['YearBuilt']>2000]['Electrical'].value_counts() # according to the YearBuilt to find the mode type of Electrical - SBrkr
train['Electrical'].replace(np.nan,'SBrkr',inplace=True)
# BsmtFinType2(1422)&BsmtFinType1(1423) - drop 1 entry
# train[train['BsmtFinType2'].isnull() & train['BsmtFinType1'].notnull()]# id=333 - type1 is GLQ and type2 area is 479&no basement; .index
# train.drop(train.index[332],inplace=True) # id=333 but index=332/from 0
# BsmtExposure(1422)&BsmtFinType1(1423) - normal data
train[train['BsmtExposure'].isnull()][['BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']] # id=949 - no basement; but type1&2 choose unfinished, area=936

df_dealing = [train,test]
for df in df_dealing:
	# Data formatting: astype('int'): GarageYrBlt
	df['GarageYrBlt'].fillna(0,inplace=True)
	df['GarageYrBlt']=df['GarageYrBlt'].astype('int')
	# Missing values: None is not missing values
	'''
	from data_description,here,NA means No equipment
	check real NaN: LotFrontage;BsmtExposure;BsmtFinType2;Electrical;
	after dealing with real NaN, 1460->1459 entries
	'''
	# LotFrontage(1200/1460,width) - possible influenced variables:Neighborhood/MSZoning
	# reference: https://www.gimme-shelter.com/frontage-50043/
	df[df['LotFrontage'].isnull()].head()
	df['LotFrontage'].fillna(df.groupby(['Neighborhood'])['LotFrontage'].transform('median'),inplace=True) # groupby+transform/apply
	# MasVnrType & MasVnrArea
	df['MasVnrType'].mode()
	df['MasVnrType'].fillna('None',inplace=True)
	df['MasVnrArea'].fillna(0,inplace=True)
	# Other Missing Values
	df.fillna('None',inplace=True)
	# transform year data

# Deal with categorical values
# reference: https://pbpython.com/categorical-encoding.html
# LabelEncoder: label.classes_
label = preprocessing.LabelEncoder()
# One-Hot Encoding
# label = preprocessing.OneHotEncoder()
label_list = ['MSZoning','Alley','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','Foundation','MasVnrType','Heating','CentralAir','Electrical','Functional','GarageType','GarageFinish','PavedDrive','MiscFeature','SaleType','SaleCondition']
for df in df_dealing:
	for columns in label_list:
		df[columns] = label.fit_transform(df[columns].values)

# ordinal variable:ExterQual;ExterCond;BsmtQual;BsmtCond;BsmtExposure;BsmtFinType1;BsmtFinType2;HeatingQC;KitchenQual;FireplaceQu;GarageQual;GarageCond;PoolQC;Fence
ordinal_dict1 = {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
ordinal_list1 = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']
for df in df_dealing:
	for columns in ordinal_list1:
		df[columns] = df[columns].map(ordinal_dict1)
	df['BsmtExposure'] = df['BsmtExposure'].map({'None':0,'No':1,'Mn':2,'Av':3,'Gd':4})
	df['BsmtFinType1'] = df['BsmtFinType1'].map({'None':0,'Unf':1,'LwQ':2,'TA':3,'Rec':4,'BLQ':5,'ALQ':6,'GLQ':7})
	df['BsmtFinType2'] = df['BsmtFinType2'].map({'None':0,'Unf':1,'LwQ':2,'TA':3,'Rec':4,'BLQ':5,'ALQ':6,'GLQ':7})
	df['Fence'] = df['Fence'].map({'None':0,'MnWw':1,'GdWo':2,'MnPrv':3,'GdPrv':4})

# Feature Selection
'''
reference: https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
'''

# correlation: for continuous variable
corr = train.corr() 
corr[(corr['SalePrice'] > 0.5) | (corr['SalePrice'] < -0.5)].index
# sns.heatmap(corr,cmap="RdYlGn",linewidths=.5) # correlation heatmap
# plt.show()
'''
0.5['OverallQual', 'YearBuilt', 'YearRemodAdd', 'ExterQual', 'BsmtQual',
'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'KitchenQual',
'TotRmsAbvGrd', 'FireplaceQu', 'GarageCars', 'GarageArea']
'''

# ANOVA: for categorial variable
# df_test = train[['MSZoning','SalePrice']]

# Modelling
X = train[['LotFrontage','LotArea','OverallQual', 'YearBuilt', 'YearRemodAdd', 'ExterQual', 'BsmtQual',
'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'KitchenQual',
'TotRmsAbvGrd', 'FireplaceQu', 'GarageCars', 'GarageArea']]
y = train['SalePrice']
# data scaling
# scale = preprocessing.StandardScaler()
# preprocessing.StandardScaler.fit_transform(scale,X)
scale_columns = ['LotFrontage','LotArea','YearBuilt', 'YearRemodAdd','TotalBsmtSF', '1stFlrSF', 'GrLivArea','GarageArea']
for column in scale_columns:
	X[column] = (X[column]-X[column].min())/(X[column].max() - X[column].min())
X = np.array(X)
y = np.log(y)

# train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)
# Multiple Linear Regression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.score(X_test,y_test))
# cross validation with mse
# sorted(sklearn.metrics.SCORERS.keys())
Rcross_lm = np.sqrt(-cross_val_score(lm,X,y,cv=5,scoring='neg_mean_squared_error'))
print(Rcross_lm.mean())
# distribution plot for y and y-estimated
y_predict = cross_val_predict(lm,X,y,cv=5)
ax1 = sns.distplot(y,hist=True,color='r',label='actual')
sns.distplot(y_predict,hist=False,color='b',label='predicted_lm',ax=ax1)
# plt.show()

# predict the test value
test_data = test[['LotFrontage','LotArea','OverallQual', 'YearBuilt', 'YearRemodAdd', 'ExterQual', 'BsmtQual','TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'KitchenQual','TotRmsAbvGrd', 'FireplaceQu', 'GarageCars', 'GarageArea']]
for column in scale_columns:
	test_data[column] = (test_data[column]-test_data[column].min())/(test_data[column].max() - test_data[column].min())
test_data = np.array(test_data)

import math

test['SalePrice'] = math.e ** lm.predict(test_data)
test[['SalePrice']].to_csv('/Kaggle/houseprice/submission.csv',index=True)
