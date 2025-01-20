import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(data):
    trans_df = data.copy()

    #extracting features:
    # trans_df['pcr_date'] = pd.to_datetime(trans_df['pcr_date'])
    # trans_df['month'] = trans_df['pcr_date'].dt.month
    # trans_df['year'] = trans_df['pcr_date'].dt.year

    trans_df['blood_type_group'] = trans_df["blood_type"].isin(["O+", "B+"])

    #dropping columns:
    trans_df.drop(columns = ["blood_type",'current_location', 'patient_id','pcr_date'], inplace = True)

    #converting catarogical columns:
    trans_df['sex'] = trans_df['sex'].map({'M': 1, 'F': -1})
    trans_df['blood_type_group'] = trans_df['blood_type_group'].map({True: 1, False: -1})

    #filling missing values:
    trans_df.fillna(trans_df[['household_income', 'PCR_02']].median(), inplace = True)

    return trans_df

def prepare_data(training_data, new_data):
    trans_df = new_data.copy()

    #normalizing data:
    # pcr_cols = set(training_data.columns[training_data.columns.str.startswith("PCR")])
    minmax_cols = set(["PCR_04", "PCR_06", "PCR_03",'conversations_per_day','household_income'])
    
    exclude_cols = {'sex','blood_type_group', 'contamination_level'}
    zscore_cols = set(training_data.columns) - minmax_cols - exclude_cols

    minmax_scaler = MinMaxScaler((-1,1))
    minmax_mask = training_data.columns.isin(minmax_cols)
    minmax_scaler.fit(training_data.loc[:, minmax_mask])
    trans_df.loc[:, minmax_mask] = minmax_scaler.transform(trans_df.loc[:, minmax_mask])

    zscore_scaler = StandardScaler()
    zsocre_mask = training_data.columns.isin(zscore_cols)
    zscore_scaler.fit(training_data.loc[:, zsocre_mask])
    trans_df.loc[:, zsocre_mask] = zscore_scaler.transform(trans_df.loc[:, zsocre_mask])


    return trans_df


if __name__ == "__main__":
    virus_data = pd.read_csv('data_HW3.csv')

    train_df, test_df = train_test_split(virus_data, test_size = 0.2, random_state = 134)

    train_df, test_df = preprocess_data(train_df), preprocess_data(test_df)

    # Prepare training set according to itself
    train_df_prepared = prepare_data(train_df, train_df)
    train_df_prepared.to_csv("train_prepared.csv", index = False)

    # Prepare test set according to the raw training set
    test_df_prepared = prepare_data(train_df, test_df)
    test_df_prepared.to_csv("test_prepared.csv", index = False)