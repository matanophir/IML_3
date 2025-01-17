import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data(training_data, new_data):
    trans_df = new_data.copy()

    #filling missing values:
    trans_df.fillna(training_data[['household_income', 'PCR_02']].median(), inplace = True)
    trans_df.fillna(training_data[['household_income', 'PCR_02']].median(), inplace = True)

    #normalizing data:
    pcr_cols = set(training_data.columns[training_data.columns.str.startswith("PCR")])
    minmax_cols = set(["PCR_04", "PCR_06", "PCR_03"])

    minmax_scaler = MinMaxScaler((-1,1))
    minmax_mask = training_data.columns.isin(minmax_cols)
    minmax_scaler.fit(training_data.loc[:, minmax_mask])
    trans_df.loc[:, minmax_mask] = minmax_scaler.transform(trans_df.loc[:, minmax_mask])

    zscore_scaler = StandardScaler()
    zsocre_mask = training_data.columns.isin(pcr_cols.difference(minmax_cols))
    zscore_scaler.fit(training_data.loc[:, zsocre_mask])
    trans_df.loc[:, zsocre_mask] = zscore_scaler.transform(trans_df.loc[:, zsocre_mask])

    #adding and dropping features:
    trans_df['blood_type_group'] = trans_df["blood_type"].isin(["O+", "B+"])
    trans_df.drop(columns = ["blood_type"], inplace = True)

    return trans_df


if __name__ == "__main__":
    virus_data = pd.read_csv('virus_data.csv')

    train_df, test_df = train_test_split(virus_data, test_size = 0.2, random_state = 134)

    # Prepare training set according to itself
    train_df_prepared = prepare_data(train_df, train_df)
    train_df_prepared.to_csv("train_prepared.csv", index = False)

    # Prepare test set according to the raw training set
    test_df_prepared = prepare_data(train_df, test_df)
    test_df_prepared.to_csv("test_prepared.csv", index = False)