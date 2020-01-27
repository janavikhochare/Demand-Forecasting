import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('GEFCom2014.csv')

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# for i in df.columns:
#     print(i, ': ', df[i].unique())
#
# print(df['Load'].isna().sum())  # 35064


df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfYear'] = df['Date'].dt.dayofyear
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['WeekOfYear'] = df['Date'].dt.weekofyear
df = df.drop(['Date'], axis=1)

# print(df.columns)
# print(df.info())


df = df[np.isfinite(df['Load'])]
df = df.reset_index()

df_load = df['Load']
df = df.drop(['Load'], axis=1)

# scaler = MinMaxScaler()
dfy = df['Year']
df = df.drop(['Year'], axis=1)
# cols = df.columns

print(df.head())
# df = scaler.fit_transform(df)
# df = pd.DataFrame(df)
# print(df.head())


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


df = normalize(df)

# df.columns = cols
# print(df.head())

df = pd.DataFrame(df)
df = df.join(dfy)
df = df.join(df_load)


ld = df['Load'].tolist()
ld_7 = []

for i in range(len(ld)):
    if i >= 7:
        ld_7.append(ld[i-7])
    else:
        ld_7.append(np.mean(ld[:7]))

df['load_d7'] = pd.DataFrame(ld_7)

print('load: ', len(ld))
print('load_d7: ', len(ld_7))
print('df: ', len(df))

print('load: ', ld)
print('load_d7: ', ld_7)

print('COLUMN VALUES:', df.columns)

is_not_2011 = df['Year'] != 2011
train = df[is_not_2011]

is_2011 = df['Year'] == 2011
test = df[is_2011]

train = train.drop(['Year'], axis=1)
test = test.drop(['Year'], axis=1)

print(df.shape)
print(train.shape)
print(test.shape)

# print(train.head())
# print(test.head())
#
# print(train['Year'].unique())
# print(test['Year'].unique())
#
# model = tf.keras.Sequential()


X_train = pd.DataFrame(train.drop(['Load'], axis=1))
Y_train = pd.DataFrame(train['Load'])

X_test = pd.DataFrame(test.drop(['Load'], axis=1))
Y_test = pd.DataFrame(test['Load'])

print('train:\n', train[:15])
print('test:\n', test.head())
print('X_train:\n', X_train.head())
print('Y_train:\n', Y_train.head())
print('X_test:\n', X_test.head())
print('Y_test:\n', Y_test.head())

inp = X_train.shape[1]
inputs = tf.keras.Input(shape=(inp,))
h1 = tf.keras.layers.Dense(16, activation=tf.nn.relu)(inputs)  # 16
h2 = tf.keras.layers.Dense(8, activation=tf.nn.relu)(h1)  # 4
h3 = tf.keras.layers.Dense(4, activation=tf.nn.relu)(h2)
outputs = tf.keras.layers.Dense(1)(h3)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))


model.fit(X_train.values, Y_train.values, epochs=200, shuffle=True, verbose=2)

y_pred = model.predict(X_test)

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)

df1 = pd.DataFrame(y_pred)

df1.to_csv('y_pred.csv')
Y_test.to_csv('y_val.csv')

y_pred = np.array(y_pred.tolist())
y_val = np.array(Y_test['Load'])


def r2_keras(y_true, y_pred):
    SS_res = np.sum(np.square(y_true - y_pred))
    SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
    print(SS_res, SS_tot, tf.keras.backend.epsilon())
    return 1-SS_res/(SS_tot + tf.keras.backend.epsilon())


# print('r2 score: ', r2_keras(y_val, y_pred))


print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))


