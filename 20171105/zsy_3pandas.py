from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 带有索引的numpy1
# obj = Series([4, 7, -5, 3])
# print(obj)
# print(obj.values)
# print(obj.index)
# 2
# obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
# print(obj2)
# print(obj2.index)
# print(obj2['a'])
# obj2['d'] = 6
# print(obj2.values)
# print(obj2[obj2 > 0])
# print(obj2 * 2)
# print(np.exp(obj2))
# print('b' in obj2)
# print('e' in obj2)
# 3
# sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
# obj3 = Series(sdata)
# # print(obj3)
# states = ['Califorina', 'Ohio', 'Oregon', 'Texas']
# obj4 = Series(sdata, index=states)
# print(obj4)
# print(pd.isnull(obj4))
# print(pd.notnull(obj4))
# print(obj4.isnull())
# print(obj3+obj4)
# obj4.name = 'population'
# obj4.index.name = 'state'
# # print(obj4)
# obj.index = ['bob', 'steve', 'jeff', 'ryan']
# print(obj)
# 4
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
# print(frame)
# print(DataFrame(data, columns=['year', 'state', 'pop']))
frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                   index=['one', 'two', 'three', 'four', 'five'])
# print(frame2)
# print(frame2.columns)
# print(frame2['state'])
# print(frame2.year)
# print(frame2.ix['three'])
# frame2['debt'] = 16.5
# print(frame2)
# frame2['debt'] = np.arange(5.)
# print(frame2)
val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
# print(frame2)
frame2['eastern'] = frame2.state == 'Ohio'
# print(frame2)
# del frame2['eastern']
# print(frame2.columns)
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop)
# print(frame3)
# print(frame3.T)
# print(DataFrame(pop, index=[2001, 2002, 2003]))
pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
# print(DataFrame(pdata))
frame3.index.name = 'year'; frame3.columns.name = 'state'
# print(frame3)
# print(frame3.values)
# print(frame2.values)
# 5
obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
# print(index)
# print(index[1:])
# print(frame3)
# print('Ohio' in frame3.columns)
# print(2002 in frame3.index)
# 6
obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
# print(obj)
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
# print(obj2)
obj3 = obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)
# print(obj3)
obj4 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
# print(obj4.reindex(range(6), method='ffill'))
frame = DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'],
                  columns=['Ohio', 'Texas', 'California'])
# print(frame)
frame2 =frame.reindex(['a', 'b', 'c', 'd'])
# print(frame2)
states = ['California', 'Texas', 'Utah']
# print(frame.reindex(columns=states))
# print(frame.reindex(index=['a', 'b', 'c', 'd'], method='ffill'))
# print(frame.ix[['a', 'b', 'c', 'd']])
# 7
obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop('c')
# print(obj)
# print(new_obj)
# print(obj.drop(['d', 'c']))
data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
# print(data)
# print(data.drop(['Colorado', 'Ohio']))
# print(data.drop('two', axis=1))
# print(data.drop(['two', 'four'], axis=1))
# 8
obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
# print(obj)
# print(obj['b'])
# print(obj[1])
# print(obj[2:4])
# print(obj[['b', 'a', 'd']])
# print(obj[[1, 3]])
# print(obj[obj < 2])
# print(obj['b':'c'])
obj['b':'c'] = 5
# print(obj)
# 9
data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
# print(data)
# print(data['two'])
# print(data[['three', 'one']])
# print(data[:2])
# print(data[data['three'] > 5])
# print(data < 5)
# data[data < 5] = 0
# print(data)
# print(data.ix['Colorado', ['two', 'three']])
# print(data.ix[['Colorado', 'Utah'], [3, 0, 1]])
# print(data.ix[2])
# print(data.ix[:'Utah', 'two'])
# print(data.ix[data.three > 5, :3])
# 10
s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
# print(s1)
# print(s2)
# print(s1+s2)
df1 = DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
                index=['ohio', 'texas', 'colorado'])
df2 = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                index=['utah', 'ohio', 'texas', 'oregon'])
# print(df1)
# print(df2)
# print(df1+df2)
df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
# print(df1)
# print(df2)
# print(df1+df2)
# print(df1.add(df2, fill_value=0))
# print(df1.reindex(columns=df2.columns, fill_value=0))
# 11
arr = np.arange(12.).reshape((3, 4))
# print(arr)
# print(arr[0])
# print(arr-arr[0])
frame = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                  index=['utah', 'ohio', 'texas', 'oregon'])
series = frame.ix[0]
# print(frame)
# print(series)
# print(frame-series)
series2 = Series(range(3), index=['b', 'e', 'f'])
# print(series2)
# print(frame+series2)
series3 = frame['d']
# print(series3)
# print(frame.sub(series3, axis=0))
# 12
frame = DataFrame(np.random.randn(4, 3), columns=list('bde'),
                  index=['utah', 'ohio', 'texas', 'oregon'])
# print(frame)
# print(np.abs(frame))
# f = lambda x: x.max() - x.min()
# print(frame.apply(f))
# print(frame.apply(f, axis=1))


def f(x):
    return Series([x.min(), x.max()], index=['min', 'max'])
# print(frame.apply(f))
format = lambda x: '%.2f' % x
# print(frame.applymap(format))
# print(frame['e'].map(format))
# 13
obj = Series(range(4), index=['d', 'a', 'b', 'c'])
# print(obj)
# print(obj.sort_index())

frame = DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'],
                  columns=['d', 'a', 'b', 'c'])
# print(frame)
# print(frame.sort_index())
# print(frame.sort_index(axis=1))
# print(frame.sort_index(axis=1, ascending=False))

obj = Series([4, np.nan, 7, np.nan, -3, 2])
# print(obj.sort_values())

frame = DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
# print(frame)
# print(frame.sort_index(by='b'))
# print(frame.sort_index(by=['a', 'b']))
# 14排名
obj = Series([7, -5, 7, 4, 2, 0, 4])
# print(obj)
# print(obj.rank())
# print(obj.rank(method='first'))
# print(obj.rank(ascending=False, method='max'))

frame = DataFrame({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1],
                   'c': [-2, 5, 8, -2.5]})
# print(frame)
# print(frame.rank(axis=1))
# 15带有重复值的轴索引
obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
# print(obj)
# print(obj.index.is_unique)
# print(obj['a'])
# print(obj['b'])
# print(obj['c'])

df = DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
# print(df)
# print(df.ix['b'])
# 16汇总和计算描述统计（约简型）
df = DataFrame([[1.4, np.nan], [7.1, -4.5],
                [np.nan, np.nan], [0.75, -1.3]],
               index=['a', 'b', 'c', 'd'],
               columns=['one', 'two'])
# print(df)
# print(df.sum())
# print(df.sum(axis=1))
# print(df.mean(axis=1, skipna=False))
# print(df.idxmax())
# print(df.idxmin())
# print(df.cumsum())
# print(df.describe())

obj = Series(['a', 'a', 'b', 'c'] * 4)
# print(obj)
# print(obj.describe())
# 17
obj = Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
uniques = obj.unique()
# print(uniques)
# print(obj.value_counts())
# print(pd.value_counts(obj.values, sort=False))

mask = obj.isin(['b', 'c'])
# print(mask)
# print(obj[mask])

data = DataFrame({'Qu1': [1, 3, 4, 3, 4],
                  'Qu2': [2, 3, 1, 2, 3],
                  'Qu3': [1, 5, 2, 4, 4]})
# print(data)
# result = data.apply(pd.value_counts).fillna(0)
# print(result)
# 18处理缺失数据
string_data = Series(['aardvark', 'artichoke', np.nan, 'avocado'])
# print(string_data)
# print(string_data.isnull())
string_data[0] = None
# print(string_data)
# print(string_data.isnull())
# 19 过滤缺失数据
from numpy import nan as NA

data = Series([1, NA, 3.5, NA, 7])
# print(data)
# print(data.dropna())
# print(data[data.notnull()])

data = DataFrame([[1., 6.5, 3.], [1., NA, NA],
                  [NA, NA, NA], [NA, 6.5, 3.]])
cleaned = data.dropna()
# print(data)
# print(cleaned)
# print(data.dropna(how='all'))
data[4] = NA
# print(data)
# print(data.dropna(axis=1, how='all'))

df = DataFrame(np.random.randn(7, 3))
# print(df)
df.ix[:4, 1] = NA
df.ix[:2, 2] = NA
# print(df)
# print(df.dropna(thresh=1))
# print(df.dropna(thresh=2))
# print(df.dropna(thresh=3))
# 20 填充缺失数据
# print(df.fillna(0))
# print(df.fillna({1: 0.5, 2: -1}))
# 总是返回被填充对象的引用
# _ = df.fillna(0, inplace=True)
# print(df)

df = DataFrame(np.random.randn(6, 3))
df.ix[:2, 1] = NA
df.ix[:4, 2] = NA
# print(df)

data = Series([1., NA, 3.5, NA, 7])
# print(data.fillna(data.mean()))
# 21 层次化索引
data = Series(np.random.randn(10),
              index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'],
                     [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])
# print(data)
# print(data.index)
# print(data['b'])
# print(data['b':'c'])
# print(data.ix[['b', 'd']])
# print(data[:, 2])
# print(data.unstack())
# print(data.unstack().stack())

frame = DataFrame(np.arange(12).reshape((4, 3)),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=[['ohio', 'ohio', 'colorrado'],
                           ['green', 'red', 'green']])
# print(frame)
frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']

# print(frame)
# print(frame['ohio'])
# print(frame.swaplevel('key1', 'key2'))
# print(frame.sort_index(level=1))
# print(frame.swaplevel(0, 1).sort_index(level=1))
# print(frame.sum(level='key2'))
# print(frame.sum(level='color', axis=1))
# 22
frame = DataFrame({'a': range(7), 'b': range(7, 0, -1),
                   'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                   'd': [0, 1, 2, 0, 1, 2, 3]})
# print(frame)
frame2 = frame.set_index(['c', 'd'])
# print(frame2)
# print(frame.set_index(['c', 'd'], drop=False))
# print(frame2.reset_index())
# 23整数索引
ser3 = Series(range(3), index=[-5, 1, 3])
frame = DataFrame(np.arange(6).reshape(3, 2), index=[2, 0, 1])
print(frame)

# 23面板数据
