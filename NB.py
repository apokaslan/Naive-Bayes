import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization
!pip install category_encoders
import category_encoders as ce
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


df=pd.read_csv('/content/drive/MyDrive/file.csv')

df.shape
df.head()
col_names = ['credit_history', 'credit_amount', 'employment', 'property_magnitude', 'age', 'class']
df.columns = col_names
df.columns
df.info


# find categorical variables
categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)


# Tüm değişkenlerin sıklık görüntülenmesi

for var in categorical: 
    
    print(df[var].value_counts())

# Dağılımları
for var in categorical: 
    
    print(df[var].value_counts()/np.float(len(df)))

# ? ile girilen değerleri NaN olarak değiştiriyorum.Python ? olan değerleri "missing value" olduğunu belirttim
df['credit_history'].replace('?', np.NaN, inplace=True)
df['credit_amount'].replace('?', np.NaN, inplace=True)
df['employment'].replace('?', np.NaN, inplace=True)
df['property_magnitude'].replace('?', np.NaN, inplace=True)
df['age'].replace('?', np.NaN, inplace=True)
df['class'].replace('?', np.NaN, inplace=True)

#kardinalite kontrolü
for var in categorical:
    
    print(var, ' contains ', len(df[var].unique()), ' labels')


X = df.drop(['class'], axis=1)

y = df['class']

#birleştirilen csv dosyasını 0.75 ve 0.25 olarak bölüyoruym
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

X_train.shape, X_test.shape

#kategorik değişkenlere bakmak
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical

#nümerik değişkenlere bakmak
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical

# train setindeki NaN değerlerin yüzdesi
X_train[categorical].isnull().mean()

#eksik değerlerin empoze edilmesi
for df2 in [X_train, X_test]:
    df2['credit_history'].fillna(X_train['credit_history'].mode()[0], inplace=True)
    df2['credit_amount'].fillna(X_train['credit_amount'].mode()[0], inplace=True)
    df2['employment'].fillna(X_train['employment'].mode()[0], inplace=True)
    df2['property_magnitude'].fillna(X_train['property_magnitude'].mode()[0], inplace=True)
    df2['age'].fillna(X_train['age'].mode()[0], inplace=True)

#tekrar eksik değer kontrolü
X_train[categorical].isnull().sum()
#test değişkeninde eksik değer kontrolü
X_test[categorical].isnull().sum()

categorical
#kategorik değişkenlerin encode ile nümerike cevrilmesi
encoder = ce.OneHotEncoder(cols=['credit_history', 'credit_amount', 'employment', 'property_magnitude', 'age'
                                 ])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

X_train.shape

cols = X_train.columns
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

#Model oluşturma
gnb = GaussianNB()


# Modeli atama işlemi
gnb.fit(X_train, y_train)


# Sonucların tahmin edilmesi
y_pred = gnb.predict(X_test)

y_pred


from sklearn.metrics import accuracy_score
# accuaracynin tahmin edilmesi
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

y_pred_train = gnb.predict(X_train)

y_pred_train

#Training set accuracy skoru 
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))

y_test.value_counts()


cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


#tp rate
true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))


#fp rate
false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
