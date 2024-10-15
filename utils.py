import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from collections import defaultdict

datasets = {
    'adult': ('adult.csv'),
    'marketing': ('bank-additional.csv'),
    'churn': ('churn.csv'),
    'compas': ('compas-scores-two-years.csv'),
    'fico': ('fico.csv'),
    'german': ('german_credit.csv'),
    'wisconsin':('breast.csv')
}


def get_tabular_dataset(name, path='./dataset/', test_size=0.2, random_state=None):

    get_dataset_fn = dataset_read_function_map[name]

    filename = path + datasets[name]

    df, class_name = get_dataset_fn(filename)

    df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = prepare_dataset(df, class_name, 'onehot')

    X = df[feature_names].astype(np.float64)
    y = df[class_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    # if len(numeric_columns)>0:
    #     scaler = MinMaxScaler()
    #     numeric_idx = [idx for idx,f in enumerate(feature_names) if f in numeric_columns]
    #     X_train.iloc[:, numeric_idx] = scaler.fit_transform(X_train.iloc[:, numeric_idx])
    #     X_test.iloc[:, numeric_idx] = scaler.transform(X_test.iloc[:, numeric_idx])

    data = {
        'seed':random_state,
        'name': name,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'class_name': class_name,
        'class_values': class_values,
        'feature_names': feature_names,
        'real_feature_names': real_feature_names,
        'numeric_columns': numeric_columns,
        'features_map': features_map,
        'df':df,
        'rdf':rdf
        }

    return data


def prepare_dataset(df, class_name, encdec):

    df = remove_missing_values(df)

    numeric_columns = get_numeric_columns(df)

    if class_name in numeric_columns:
        numeric_columns.remove(class_name)

    rdf = df.copy()

    if encdec == 'onehot' or encdec == 'none':
        df, feature_names, class_values = one_hot_encoding(df, class_name)

        real_feature_names = get_real_feature_names(rdf, numeric_columns, class_name)

        rdf = rdf[real_feature_names + (class_values if isinstance(class_name, list) else [class_name])]

        features_map = get_features_map(feature_names, real_feature_names)

    elif encdec == 'target':
        feature_names = df.columns.values
        feature_names = np.delete(feature_names, np.where(feature_names == class_name))
        class_values = np.unique(df[class_name]).tolist()
        numeric_columns = list(df._get_numeric_data().columns)
        real_feature_names = [c for c in df.columns if c in numeric_columns and c != class_name]
        real_feature_names += [c for c in df.columns if c not in numeric_columns and c != class_name]
        #print('feat names and real ', len(feature_names), len(real_feature_names))
        features_map = dict()
        for f in range(0, len(real_feature_names)):
            features_map[f] = dict()
            features_map[f][real_feature_names[f]] = np.where(feature_names == real_feature_names[f])[0][0]
    return df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map


def get_features_map(feature_names, real_feature_names):
    features_map = defaultdict(dict)
    i = 0
    j = 0

    while i < len(feature_names) and j < len(real_feature_names):
        if feature_names[i] == real_feature_names[j]:
            #print('in if ', feature_names[i], real_feature_names[j])
            features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
            i += 1
            j += 1

        elif feature_names[i].startswith(real_feature_names[j]):
            #print('in elif ', feature_names[i], real_feature_names[j])
            features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
            i += 1
        else:
            j += 1
    return features_map


def get_real_feature_names(rdf, numeric_columns, class_name):
    if isinstance(class_name, list):
        real_feature_names = [c for c in rdf.columns if c in numeric_columns and c not in class_name]
        real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c not in class_name]
    else:
        real_feature_names = [c for c in rdf.columns if c in numeric_columns and c != class_name]
        real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c != class_name]
    return real_feature_names


def one_hot_encoding(df, class_name):
    if not isinstance(class_name, list):
        dfX = pd.get_dummies(df[[c for c in df.columns if c != class_name]], prefix_sep='=')
        class_name_map = {v: k for k, v in enumerate(sorted(df[class_name].unique()))}
        dfY = df[class_name].map(class_name_map)
        df = pd.concat([dfX, dfY], axis=1)
        df =df.reindex(dfX.index)
        feature_names = list(dfX.columns)
        class_values = sorted(class_name_map)
    else: # isinstance(class_name, list)
        dfX = pd.get_dummies(df[[c for c in df.columns if c not in class_name]], prefix_sep='=')
        # class_name_map = {v: k for k, v in enumerate(sorted(class_name))}
        class_values = sorted(class_name)
        dfY = df[class_values]
        df = pd.concat([dfX, dfY], axis=1)
        df = df.reindex(dfX.index)
        feature_names = list(dfX.columns)
    return df, feature_names, class_values


def remove_missing_values(df):
    for column_name, nbr_missing in df.isna().sum().to_dict().items():
        if nbr_missing > 0:
            if column_name in df._get_numeric_data().columns:
                mean = df[column_name].mean()
                df[column_name].fillna(mean, inplace=True)
            else:
                mode = df[column_name].mode().values[0]
                df[column_name].fillna(mode, inplace=True)
    return df


def get_numeric_columns(df):
    numeric_columns = list(df._get_numeric_data().columns)
    return numeric_columns


def get_adult_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True, na_values='?', keep_default_na=True)
    df.drop(['fnlwgt', 'education-num'], axis=1, inplace=True)
    return df, class_name


def get_marketing_dataset(filename):
    class_name = 'y'
    df = pd.read_csv(filename, sep=';', skipinitialspace=True, keep_default_na=True)
    return df, class_name


def get_churn_dataset(filename):
    class_name = 'churn'
    df = pd.read_csv(filename, skipinitialspace=True, na_values='?', keep_default_na=True)
    df.drop(['phone number'], inplace=True, axis=1)
    return df, class_name


def get_compas_dataset(filename, binary=True):

    df = pd.read_csv(filename, delimiter=',', skipinitialspace=True)

    columns = ['age', 'age_cat', 'sex', 'race',  'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
               'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score', 'score_text']

    df = df[columns]

    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])

    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])

    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)

    df['length_of_stay'] = df['length_of_stay'].astype(int)
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)

    if binary:
        def get_class(x):
            if x < 7:
                return 'Medium-Low'
            else:
                return 'High'
        df['class'] = df['decile_score'].apply(get_class)
    else:
        df['class'] = df['score_text']

    del df['c_jail_in']
    del df['c_jail_out']
    del df['decile_score']
    del df['score_text']

    class_name = 'class'
    return df, class_name


def get_fico_dataset(filename):
    class_name = 'RiskPerformance'
    df = pd.read_csv(filename, skipinitialspace=True, keep_default_na=True)
    return df, class_name


def get_german_dataset(filename):
    class_name = 'default'
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = [c.replace('=', '') for c in df.columns]
    return df, class_name


def get_wisconsin_dataset(filename):
    names = ['ID', 'diagnosis', 
       'mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture','worst perimeter', 'worst area', 
       'worst smoothness','worst compactness', 'worst concavity', 
       'worst concave points', 'worst symmetry', 'worst fractal dimension']

    class_name = 'diagnosis'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True, names=names)
    df.drop(['ID'], axis=1, inplace=True)

    return df, class_name

dataset_read_function_map = {
    'adult': get_adult_dataset,
    'marketing': get_marketing_dataset,
    'churn': get_churn_dataset,
    'compas': get_compas_dataset,
    'fico': get_fico_dataset,
    'german': get_german_dataset,
    'wisconsin':get_wisconsin_dataset
}