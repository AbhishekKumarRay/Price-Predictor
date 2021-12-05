import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
import pickle
import json


# Loading data
df1 = pd.read_csv("..\Data\Bengaluru_House_Data.csv")
# print(df1.head(),df1.shape)
# print(df1['society'].value_counts())


# Removing unnecessary features that are not required to build our model.
df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
# print(df1.shape,df2.shape)


# Handeling NAN values
# print(df2.isnull().sum())
df3=df2.dropna()  # removed all nan values instead of filling it with fillna().
# df3=df2.fillna(0)
# print(df3.isnull().sum())

# Adding new feature for bhk
df4=df3.copy()
df4['BHK']=df4['size'].apply(lambda x: int(x.split(' ')[0]))
# print(df4.head(10))

# Explore total_sqft feature
def is_float(x):
    try:
        float(x)
    except:
        return True
    return False
# print(df4[df4['total_sqft'].apply(is_float)].head(10))

# Above shows that total_sqft can be a range (e.g. 2100-2850).
# For such case we can just take average of min and max value in the range.
# There are other cases such as 34.46Sq. Meter which one can convert to square ft using unit cpiponversion.
# I am going to just drop such corner cases to keep things simple

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
# Using none so that all the values like 34.46Sq. Meter, 300Sq. Yards, etc could be replace with none/NULL.

df5 = df4.copy()
df5.total_sqft = df5.total_sqft.apply(convert_sqft_to_num)
df5 = df5[df5.total_sqft.notnull()]
# print(df5.head(31))


# Add new feature called price per square feet
df6 = df5.copy()
df6['price_per_sqft'] = df6['price']*100000/df6['total_sqft']
# print(df6.head())


# Examine locations. We need to apply dimensionality reduction technique to reduce number of locations.
df6.location = df6.location.apply(lambda x: x.strip())
location_stats = df6['location'].value_counts(ascending=False)
# print(location_stats,df6.shape)

# print(location_stats.values.sum())
# print(len(location_stats))
# print(len(location_stats[location_stats>10]))
# print(len(location_stats[location_stats<=10]))


# Any location having less than 10 data points should be tagged as "other" location.
location_stats_less_than_10 = location_stats[location_stats<=10]
df6.location = df6.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
# print(df6.head(10))



# Typically 300sqft =1BHK. So accordingly we will remove the data which seems suspicious like 1200=8BKS and all.
df7 = df6[~(df6.total_sqft/df6.BHK<300)]
# print(df7.shape)
# print(df7.price_per_sqft.describe())


# The min price per sqft is 267 rs/sqft whereas max is 176470. Now remove outliers per location using mean and standard deviation.
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)

    return df_out
df8 = remove_pps_outliers(df7)
# print(df8.price_per_sqft.describe())


# How does the 2 BHK and 3 BHK property prices look like
def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.BHK == 2)]
    bhk3 = df[(df.location == location) & (df.BHK == 3)]
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
# plot_scatter_chart(df8, "Rajaji Nagar")
# plt.show()

# Remove 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df9 = remove_bhk_outliers(df8)

# print(df9.shape)
# plot_scatter_chart(df9,"Rajaji Nagar")
# # plt.show()



# Outlier Removal Using Bathrooms Feature
# Normally we have 2 more bathrooms than number of bedrooms. Removing those which have more

# print(df9.bath.unique())
# print(df9[df9.bath>df9.BHK+2])

df10 = df9[df9.bath<df9.BHK+2]
# print(df10.shape)

df11 = df10.drop(['size','price_per_sqft'],axis='columns')
# print(df11.head())


# Using Dummies for location
dummies = pd.get_dummies(df11.location)
df12 = pd.concat([df11,dummies.drop('other',axis='columns')],axis='columns')
# All 0 will be used for representing other feature. Like 1000.. is for 1st Block Jayanagar, so 00000... for others.
df13 = df12.drop('location',axis='columns')


# Building a MODEL
X = df13.drop(['price'],axis='columns')
y = df13.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
score=lr_clf.score(X_test,y_test)
# print(score)


# Use K Fold cross validation to measure accuracy of our LinearRegression model
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
score2=cross_val_score(LinearRegression(), X, y, cv=cv)
# print(score2)

# Find best model using GridSearchCV

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

# print(find_best_model_using_gridsearchcv(X,y))
# print(X.columns)
# print(np.where(X.columns=="2nd Phase Judicial Layout")[0][0])

# Test the model
def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

# Testing the model
print(predict_price('1st Phase JP Nagar',1000, 2, 2))
print(predict_price('1st Phase JP Nagar',1000, 3, 3))
print(predict_price('Indira Nagar',1000, 2, 2))
print(predict_price('Indira Nagar',1000, 3, 3))

# Export the tested model to a pickle file and also the excel sheet.
with open('Bengaluru_House_Data_Model.pickle','wb') as f:
    pickle.dump(lr_clf,f)

df13.to_csv('..\Data\Final_Data_df13.csv',index=False)


# Export location and column information to a file.
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))
