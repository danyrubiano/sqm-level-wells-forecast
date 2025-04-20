import os
import yaml
import utm
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datetime import date, timedelta
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_error, mean_squared_error
import warnings

# Filter the specific FutureWarning about force_all_finite
warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.",
    category=FutureWarning
)


class Preprocess:

    def __init__(self, path: str='data/', context: str='local', max_days: int=25, len_history: int=1826):

        self.max_days = max_days
        self.len_history = len_history
        self.path = path
        self.context = context

        if self.context == 'google_drive':
            from google.colab import drive
            drive.mount('/content/drive')

        self.data_wells = self.read_parquet('wells_data')
        self.data_meteo = self.read_parquet('meteo_data')
        self.wells_info = self.read_info_yaml('wells')
        self.meteo_info = self.read_info_yaml('meteo')
        self.meteo_well_comb = None
        self.data = None
        self.data_train = None
        self.data_test = None
        self.inputation_info = {}
        self.metrics_predictors = None


    def read_info_yaml(self, station_type: str):
        """
        Reads information from a YAML file and returns it as a pandas DataFrame.
        Args:
            station_type (str): The type of station, used to determine the YAML file to read.
                                 The file is expected to be located at `self.path` with the
                                 filename format `{station_type}.yaml`.
        Returns:
            pd.DataFrame: A DataFrame containing the data loaded from the YAML file.
        """
        yaml_path = f"{self.path}{station_type}.yaml"
        try:
            if self.context in ['local','google_drive']:
                with open(yaml_path, 'r') as file:
                    data = yaml.safe_load(file)
                return data

            elif self.context == 'vertex':
                blob_name = f'config-data/{station_type}.yaml'
                client = storage.Client()
                bucket = client.bucket(self.bucket)
                blob = bucket.blob(blob_name)
                contenido_yaml = blob.download_as_text()
                data = yaml.safe_load(contenido_yaml)
                return data
            else:
                logging.error(f"Unknown context: {self.context}. Cannot read info YAML.")
                return None

        except FileNotFoundError:
            logging.error(f"YAML file not found: {yaml_path}")
            return None
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file {yaml_path}: {e}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred while reading YAML from GCS: {e}")
            return None


    def read_parquet(self, file_path: str):
        """
        Read data from a parquet file and return it as a dataframe
        Args:
            file_path (str): the path of the parquet file.
        Returns:
            pd.DataFrame: A DataFrame containing the data loaded.
        """
        path = f"{self.path}{file_path}.parquet"
        data = pd.read_parquet(path)
        return data


    def save_as_excel(self, data: pd.DataFrame, file_path: str):
        """
        Saves the processed data to a excel file.
        The file is saved in the specified path.
        Args:
            data (pd.DataFrame): the data in dataframe format
            file_path (str): the path of the parquet file.
        """
        path = f"{self.path}{file_path}.xlsx"
        data.to_excel(path, index=False)
        print(f"Data saved to {path}")


    def fill_levels_wells(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in the 'manual_level' column with values from the
        'continue_level' column, renames the 'manual_level' column to 'level',
        and drops the 'continue_level' column.
        Args:
            data (pd.DataFrame): A pandas DataFrame containing the columns
                'manual_level' and 'continue_level'.
        Returns:
            pd.DataFrame: A pandas DataFrame with the 'manual_level' column
            filled, renamed to 'level', and the 'continue_level' column removed.
        """
        data['manual_level'] = np.where(
            data['manual_level'].isnull(), data['continue_level'], data['manual_level']
        )
        data = (
            data
            .rename(columns={'manual_level':'level'})
            .drop(columns=['continue_level'])
        )
        return data


    def inputation(self, data: pd.DataFrame, station_type: str) -> pd.DataFrame:
        """
        Imputes missing values and smooths data for the dataset.
        This method processes the dataset by iterating over unique well IDs, imputing missing values,
        applying smoothing to numerical columns, and handling specific conditions for precipitation data.
        The processed data is then concatenated and returned as a DataFrame.
        Args:
            data (pd.DataFrame): the data in dataframe format
            station_type (str): the type of station (well_id or meteo_id)
        Returns:
            pd.DataFrame: The processed DataFrame with missing values imputed and data smoothed.
        Steps:
            1. Iterates through unique well IDs in the dataset.
            2. Drops null values from the dataset, excluding protected and categorical columns.
            3. Imputes missing values for numerical columns using STL decomposition.
            4. Applies a rolling mean with a window of 7 to smooth numerical columns, excluding certain columns.
            5. Ensures precipitation values ('prec' and 'prec_1') are non-negative.
            6. Concatenates the processed data for all wells.
            7. Drops rows with null values based on a threshold, excluding protected and categorical columns.
            8. Updates the class's dataset with the processed data.
        Notes:
            - Categorical columns are excluded from imputation and smoothing.
            - Protected columns are excluded from null value dropping.
            - The method uses helper functions `Preprocess.drop_nulls` and `Preprocess.stl_imputation`.
        """
        print("- Imputing missing values and smotthing data")
        categorical_cols = ['system_well', 'well_id', 'meteo_id', 'system_meteo']
        protected_cols = ['ds', 'prec', 't_prom', 'evap', 'velvto']
        station_col = 'well_id' if station_type=='wells' else 'meteo_id'
        data_list = []

        for station in tqdm(data[station_col].unique()):
            data_temp = data[data[station_col] == station].copy()
            data_temp = Preprocess.drop_nulls(data_temp, exclude_cols=protected_cols+categorical_cols)
            data_temp = data_temp.set_index('ds')
            inputation_info ={}
            inputation_info['shape'] = data_temp.shape
            for column in data_temp.columns:
                if data_temp[column].isnull().any() and column not in categorical_cols:
                    df_imputed, inputation_index = Preprocess.stl_imputation(data_temp, column)
                    inputation_info[column] = inputation_index
                    data_temp[column] = df_imputed

            """
            if column not in categorical_cols and column not in ['ds','prec','prec_1']:
                    data_temp[column] = data_temp[column].rolling(window=7, min_periods=1).mean()

            if 'prec' in data_temp.columns:
                data_temp['prec'] = np.where(data_temp['prec']<0, 0, data_temp['prec'])
                data_temp['prec_1'] = np.where(data_temp['prec_1']<0, 0, data_temp['prec_1'])
            """
            data_temp = data_temp.reset_index()

            self.inputation_info[station] = inputation_info
            data_list.append(data_temp)

        final_data = pd.concat(data_list, ignore_index=False)

        final_data = Preprocess.drop_nulls(final_data, threshold=1 ,exclude_cols=protected_cols+categorical_cols)
        return final_data


    def smooth_meteo(self):
        """
        Smooths meteorological data by applying a rolling mean to columns with missing values.
        This method processes the `self.data_meteo` DataFrame, which is expected to contain
        meteorological data. For each unique `meteo_id`, it applies a rolling mean with a
        window size of 7 to columns that have missing values, excluding columns listed in
        `protected_cols`. The smoothed data is then concatenated and returned.
        Returns:
            pd.DataFrame: The updated `self.data_meteo` DataFrame with smoothed values.
        """

        protected_cols = ['meteo_id','system_meteo']
        data_list = []

        for meteo_id in self.data_meteo['meteo_id'].unique():
            data_temp = self.data_meteo[self.data_meteo['meteo_id'] == meteo_id].copy()
            data_temp = data_temp.set_index('ds')
            for meteo_var in data_temp.columns:
                if data_temp[meteo_var].isnull().any() and meteo_var not in protected_cols:
                    data_temp[meteo_var] = data_temp[meteo_var].rolling(window=7, min_periods=1).mean()

                data_temp = data_temp.reset_index()
            data_list.append(data_temp)

        self.data_meteo = pd.concat(data_list, ignore_index=False)
        return self.data_meteo


    def drop_continuos_nulls(self, data: pd.DataFrame, name_id: str, col_ref: str) -> pd.DataFrame:
        """
        Drops rows from a DataFrame where a specified column contains continuous null values
        exceeding a threshold and returns the cleaned DataFrame.
        Args:
            data (pd.DataFrame): The input DataFrame to process.
            name_id (str): The column name used to group the data (e.g., an identifier column).
            col_ref (str): The column name to check for null values.
        Returns:
            pd.DataFrame: A DataFrame with rows containing continuous null values exceeding
                        the threshold removed.
        Notes:
            - The function identifies groups of continuous null values in the specified column (`col_ref`).
            - If a group of continuous null values exceeds 14 rows, all rows in the group are removed.
            - Intermediate columns used for computation are dropped in the final output.
        """
        data['null_count'] =  np.where((data[col_ref].isnull()), 1, np.nan)
        data['count_cumsum'] = data['null_count'].groupby(data['null_count'].isna().cumsum()).cumsum()
        data['total_nulls_in_group'] = data.groupby(data['null_count'].isna().cumsum())['count_cumsum'].transform('max')
        data['total_nulls_in_group'] = np.where(data['null_count'].isna(), np.nan, data['total_nulls_in_group'])
        data['group_number'] = data.groupby([name_id, data['null_count'].isna().cumsum()]).ngroup()
        data['group_number'] = np.where(data['null_count'].isna(), np.nan, data['group_number'])
        filter_ds = (
            data[data['total_nulls_in_group']>self.max_days]
            .groupby(name_id)['ds']
            .max()
            .reset_index()
            .rename(columns={'ds': 'min_ds'})
        )
        data = data.merge(filter_ds, on=name_id, how='left')
        data['filter'] = np.where(data['ds'] <= data['min_ds'], 1, 0)
        data_final = data[data['filter'] == 0].copy()
        data_final = data_final.drop(columns=[
            'null_count','count_cumsum','total_nulls_in_group','group_number','min_ds','filter'
        ])
        return data_final
    
    
    def set_train_test(self, data: pd.DataFrame, set_index_date: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataset into training and testing sets based on specified date ranges.
        Args:
            set_index_date (bool, optional): If True, sets the 'ds' column as the index 
                for both the training and testing datasets. Defaults to False.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
                - data_train: The training dataset, filtered based on the specified 
                  training date range.
                - data_test: The testing dataset, filtered based on the specified 
                  testing date range.
        """
        data['ds'] = pd.to_datetime(data['ds'])
        data_train = data[data['ds'].between(
            pd.to_datetime(self.train_end - timedelta(days=self.len_train)),
            pd.to_datetime(self.train_end)
        )]
        data_test = data[data['ds'].between(
            pd.to_datetime(self.train_end + timedelta(days=1)),
            pd.to_datetime(self.train_end + timedelta(days=self.h+1))
        )]
        if set_index_date:
            data_train = data_train.set_index('ds')
            data_test = data_test.set_index('ds')
        
        return data_train, data_test+


    def preprocess_wells(self) -> pd.DataFrame:
        """
        Preprocess well data by consolidating, filling levels, and dropping continuous nulls.
        This method performs the following steps:
        1. Consolidates well data using the `consolide_data` method with 'wells' as the argument.
        2. Fills missing levels in the well data using the `fill_levels_wells` method.
        3. Drops continuous null values from the well data using the `drop_continuos_nulls` method,
           with 'well_id' as the identifier and 'level' as the reference column.
        Returns:
            pd.DataFrame: The preprocessed well data.
        """
        print("- Preprocessing wells data")
        self.data_wells['ds'] = pd.to_datetime(self.data_wells['ds'])
        min_date = self.data_wells['ds'].max() - pd.Timedelta(self.len_history, unit='D')
        self.data_wells = self.data_wells[self.data_wells['ds']<=min_date].copy()

        self.data_wells = self.fill_levels_wells(self.data_wells)
        self.data_wells = self.drop_continuos_nulls(self.data_wells, name_id='well_id', col_ref='level')
        self.data_wells = self.inputation(self.data_wells, station_type='wells')
        return self.data_wells


    def preprocess_meteo(self) -> pd.DataFrame:
        """
        Preprocess meteorological data by consolidating it into a single DataFrame.
        Returns:
            pd.DataFrame: A DataFrame containing the consolidated meteorological data.
        """
        print("- Preprocessing meteo data")
        self.data_meteo['ds'] = pd.to_datetime(self.data_meteo['ds'])
        min_date = self.data_meteo['ds'].max() - pd.Timedelta(self.len_history, unit='D')
        self.data_meteo = self.data_meteo[self.data_meteo['ds']<=min_date].copy()

        self.data_meteo = self.inputation(self.data_meteo, station_type='meteo')
        self.data_meteo = self.smooth_meteo()
        data_train, data_test = self.set_train_test(self.data_meteo)
        data_test, self.metrics_predictors = Preprocess.forecast_meteo_variables()
        
        self.data_meteo  = pd.concat([data_train, data_test], ignore_index=True)
        self.data_meteo ['ds'] = pd.to_datetime(self.data_meteo ['ds'])
        self.data_meteo  = data.sort_values('ds')

        self.data_meteo ['prec_acum'] = self.data_meteo .groupby(['meteo_id'])['prec'].transform(
            lambda x: x.groupby((x != 0).cumsum()).cumsum()
        )
        self.data_meteo ['evap_acum'] = self.data_meteo .groupby(['meteo_id'])['evap'].transform(
            lambda x: x.where(self.data_meteo .loc[x.index, 'prec'] > 0, 0).cumsum()
        )
        self.data_meteo ['rel_prec_evap'] = self.data_meteo ['prec_acum']/(self.data_meteo ['evap_acum']+1e-6)
        self.data_meteo ['rel_t_evap'] = self.data_meteo ['evap']/self.data_meteo ['t_prom']

        self.data_meteo ['month'] = pd.to_datetime(self.data_meteo ['ds']).dt.month
        self.data_meteo ['summer'] = np.where(self.data_meteo ['month'].isin([12,1,2]), 1, 0)
        self.data_meteo ['fall'] = np.where(self.data_meteo ['month'].isin([3,4,5]), 1, 0)
        self.data_meteo ['winter'] = np.where(self.data_meteo ['month'].isin([6,7,8]), 1, 0)
        self.data_meteo ['spring'] = np.where(self.data_meteo ['month'].isin([9,10,11]), 1, 0)
        self.data_meteo ['high_plateau'] = np.where(self.data_meteo ['month'].isin([12,1,2,3]), 1, 0)

        self.data_meteo  = pd.get_dummies(self.data_meteo , columns=['meteo_id'], drop_first=True)
        self.data_meteo  = self.data_meteo.drop(columns=['month','system_meteo'], axis=1)
        return self.data_meteo


    def merge_well_meteo(self) -> pd.DataFrame:
        """
        Merges well and meteorological data based on proximity and temporal alignment.
        This method performs the following steps:
        1. Calculates distances between wells and meteorological stations.
        2. Combines well and meteorological data using the calculated distances.
        3. Aggregates the combined data to compute the number of unique dates (`ds`)
           and the count of null values in the temperature column (`t_prom`).
        4. Calculates a balance ratio based on distance, null counts, and unique dates.
        5. Selects the best meteorological station for each well based on the balance ratio.
        6. Merges the selected meteorological data with the well data.
        Returns:
            pd.DataFrame: A DataFrame containing the merged well and meteorological data
                          with the best meteorological station for each well.
        """
        print("- Merging well and meteo data")
        distances = self.get_distances_well_meteo()
        combined = (
            distances
            .merge(self.data_wells, on='well_id', how='inner')
            .merge(self.data_meteo, on=['meteo_id', 'ds'], suffixes=('_well', '_meteo'))
        )
        combination = (
            combined
            .groupby(['well_id','meteo_id','distance'])[['ds','t_prom']]
            .agg({'ds':'nunique', 't_prom':lambda x: x.isnull().sum()})
            .reset_index()
        )
        combination['balance_ratio'] = (combination['distance']*combination['t_prom']) / (1 + (combination['ds'] - combination['t_prom'])**2)
        combination = combination.sort_values(['well_id','balance_ratio'])
        self.meteo_well_comb = combination.groupby('well_id').first().reset_index()

        self.data = (
            self.data_wells
            .merge(self.meteo_well_comb[['well_id','meteo_id']], on='well_id', how='inner')
            .merge(self.data_meteo, on=['meteo_id', 'ds'], suffixes=('_well', '_meteo'))
        )
        self.data['ds'] = pd.to_datetime(self.data['ds'])
        return self.data


    def set_cluster(self) -> pd.DataFrame:
        """
        Clusters wells by level and updates the data with cluster assignments.

        This method performs the following steps:
        1. Prepares the data for clustering by pivoting it.
        2. Determines the optimal number of clusters (k) using a search within a specified range.
        3. Visualizes the k search results.
        4. Clusters the data using the optimal number of clusters.
        5. Visualizes the resulting clusters.
        6. Merges the cluster assignments back into the original data.

        Returns:
            pd.DataFrame: The updated DataFrame with an additional 'cluster' column
                          indicating the cluster assignment for each well.
        """
        print("- Clustering wells by level")
        k_means_data = Preprocess.pivot_data(self.data)
        print("\t- Finding optimal number of clusters")
        k_search = Preprocess.find_k(k_means_data, min_k=5, max_k=14)
        Preprocess.plot_k_search(k_search)
        print("\t- Aplying clusterization")
        clustered_data = Preprocess.cluster_data(k_means_data, n_clusters=k_search['best_k'])
        Preprocess.plot_clusters(clustered_data, n_clusters=k_search['best_k'])
        self.data = (
            self.data
            .merge(clustered_data[['cluster']].reset_index(), how='left', on='well_id')
        )
        return self.data


    def preprocess(self) -> pd.DataFrame:
        """
        Preprocesses meteorological and well data, merges them, and performs imputation
        for missing values using STL decomposition.
        Returns:
            pd.DataFrame: A DataFrame containing the preprocessed and imputed data
            for all wells
        """
        print("Preprocessing data")
        _ = self.preprocess_meteo()
        _ = self.preprocess_wells()
        _ = self.merge_well_meteo()
        _ = self.inputation()
        _ = self.set_cluster()
        self.save_data()
        print("Preprocessing completed")
        return self.data


    @staticmethod
    def drop_nulls(data: pd.DataFrame, threshold: float=0.7, exclude_cols=[]) -> pd.DataFrame:
        """
        Drops columns from a DataFrame where the percentage of null values exceeds a specified threshold.
        Args:
            data (pd.DataFrame): The input DataFrame to process.
            threshold (float, optional): The threshold for null value percentage. Columns with a null percentage
                                          greater than this value will be dropped. Defaults to 0.7.
            exclude_cols (list, optional): A list of columns to exclude from the null value check. Defaults to [].
        Returns:
            pd.DataFrame: A DataFrame with columns containing excessive null values removed.
        """
        null_percentage = data[data.columns.difference(exclude_cols)].isnull().mean()
        columns_to_drop = null_percentage[null_percentage >= threshold].index
        data = data.drop(columns=columns_to_drop)
        return data

    @staticmethod
    def stl_imputation(data: pd.DataFrame, variable: str) -> Tuple[pd.DataFrame, List[date]]:
        """
        Perform STL (Seasonal-Trend decomposition using LOESS) based imputation on a specified variable in a DataFrame.
        This function identifies missing values in the specified variable, applies STL decomposition to separate
        the seasonal component, and performs linear interpolation on the deseasonalized data to fill in the missing values.
        The imputed values are then recombined with the seasonal component to produce the final imputed series.
        Args:
            data (pd.DataFrame): The input DataFrame containing the data to be imputed.
            variable (str): The name of the column in the DataFrame to perform imputation on.
        Returns:
            Tuple[pd.DataFrame, List[date]]:
                - A DataFrame containing the imputed values for the specified variable.
                - A list of indices with dates where the imputation was performed.
        """
        imputed_indices = data[data[variable].isnull()].index
        try:
            stl = STL(data[variable].interpolate())
            res = stl.fit()
            seasonal_component = res.seasonal
            df_deseasonalised = data[variable] - seasonal_component
            df_deseasonalised_imputed = df_deseasonalised.interpolate(method="linear")
            df_imputed = df_deseasonalised_imputed + seasonal_component
        except:
            df_imputed = data[variable]
            print(f"Imputation failed for {variable}")
            pass
        return df_imputed, imputed_indices

    @staticmethod
    def plot_inputation(data: pd.DataFrame, imputed_indices: List[date],  variable: str):
        """
        Plots a time series variable with imputed values highlighted.
        Args:
            data (pd.DataFrame): A pandas DataFrame containing the time series data.
        imputed_indices (List[date]): A list of indices (dates) where the data has been imputed.
        variable (str): The name of the variable/column in the DataFrame to be plotted.
        """
        plt.figure(figsize=[12, 6])
        data[variable].plot(style='.-',  label='hr')
        plt.scatter(imputed_indices, data.loc[imputed_indices, variable], color='red')
        plt.title(f"{variable} with STL Imputation")
        plt.ylabel(variable)
        plt.xlabel("ds")
        plt.show()

    @staticmethod
    def pivot_data(data: pd. DataFrame) -> pd.DataFrame:
        """
        Transforms and scales a DataFrame by pivoting it on 'ds' as the index and 'well_id' as columns.
        Args:
            data (pd.DataFrame): Input DataFrame containing the columns 'ds', 'well_id', and 'level'.
        Returns:
            pd.DataFrame: A transformed DataFrame where:
        """
        data['well_id'] = data['well_id'].astype(str)
        data_pivot = (
            data
            .pivot(index='ds', columns='well_id', values='level')
            .dropna()
        )
        scaler = StandardScaler()
        scaler.fit(data_pivot)
        data_pivot.iloc[:,:] = scaler.transform(data_pivot)
        data_pivot = data_pivot.T
        return data_pivot

    @staticmethod
    def find_k(data: pd.DataFrame, min_k: int=2, max_k: int=15, seed: int=42) -> dict:
        """
        Determines the optimal number of clusters (k) for time series data using
        the TimeSeriesKMeans algorithm with the Dynamic Time Warping (DTW) metric.
        Args:
            data (pd.DataFrame): The input time series data to cluster.
            min_k (int, optional): The minimum number of clusters to evaluate. Default is 2.
            max_k (int, optional): The maximum number of clusters to evaluate. Default is 15.
            seed (int, optional): Random seed for reproducibility. Default is 42.
        Returns:
            dict: A dictionary containing the following keys:
                - 'k_values': List of evaluated k values.
                - 'silhouette_scores': List of silhouette scores for each k.
                - 'inertia': List of inertia values for each k.
                - 'best_k': The optimal number of clusters based on combined scores.
        """
        inertia_values = []
        silhouette_values = []

        for n_clusters in tqdm(range(min_k, max_k + 1)):
            model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=10, random_state=seed)
            clusters = model.fit_predict(data)
            inertia = model.inertia_
            inertia_values.append(inertia)
            silhouette_avg = silhouette_score(data, clusters)
            silhouette_values.append(silhouette_avg)

        k_values = list(range(min_k, max_k + 1))
        silhouette_scores = silhouette_values
        inertia_scores = inertia_values

        # Normalize silhouette and inertia scores
        silhouette_norm = (np.array(silhouette_scores) - np.min(silhouette_scores)) / (np.max(silhouette_scores) - np.min(silhouette_scores))
        inertia_norm = (np.max(inertia_scores) - np.array(inertia_scores)) / (np.max(inertia_scores) - np.min(inertia_scores))

        # Combine normalized scores to balance silhouette and inertia
        combined_scores = silhouette_norm + inertia_norm
        best_k = k_values[np.argmax(combined_scores)]
        return {'k_values': k_values, 'silhouette_scores': silhouette_scores, 'inertia':inertia_scores, 'best_k': best_k}

    @staticmethod
    def plot_k_search(k_search: dict):
        """
        Plots the evaluation of the number of clusters (k) using silhouette scores and inertia.
        This function creates a dual-axis plot:
        - The primary y-axis (left) shows the silhouette scores for different values of k.
        - The secondary y-axis (right) shows the inertia for the same values of k.
        - A vertical dashed line indicates the best k value.
        Args:
            k_search (dict): A dictionary containing the following keys:
                - 'k_values' (list of int): The list of k values evaluated.
                - 'silhouette_scores' (list of float): The silhouette scores corresponding to each k value.
                - 'inertia' (list of float): The inertia values corresponding to each k value.
                - 'best_k' (int): The optimal number of clusters (k) determined.
        """
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Silhouette Score', color=color)
        ax1.plot(k_search['k_values'], k_search['silhouette_scores'], color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Inertia', color=color)
        ax2.plot(k_search['k_values'], k_search['inertia'], color=color, marker='x')
        ax2.tick_params(axis='y', labelcolor=color)

        ax1.axvline(x=k_search['best_k'], color='green', linestyle='--', linewidth=2)

        plt.title('Evaluation of Number of Clusters (k)')
        plt.show()

    @staticmethod
    def cluster_data(data: pd.DataFrame, n_clusters: int, seed: int=42) -> pd.DataFrame:
        """
        Cluster time series data using the TimeSeriesKMeans algorithm with Dynamic Time Warping (DTW) as the distance metric.
        Args:
            data (pd.DataFrame): A pandas DataFrame containing the time series data to be clustered.
                                    Each row is expected to represent a time series.
            n_clusters (int): The number of clusters to form.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        Returns:
            pd.DataFrame: The input DataFrame with an additional column 'cluster' indicating the cluster assignment for each row.
        """
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=10, random_state=seed)
        clusters = model.fit_predict(data)
        data['cluster'] = clusters
        return data

    @staticmethod
    def plot_clusters(data: pd.DataFrame, n_clusters: int):
        """
        Plots time series data grouped by clusters.
        This function visualizes the time series data for each cluster in a grid layout.
        Each subplot corresponds to a cluster, and the time series within that cluster
        are plotted together.
        Args:
            data (pd.DataFrame): A DataFrame containing the time series data. It must include
                a 'cluster' column that indicates the cluster assignment for each row.
            n_clusters (int): The number of clusters to plot.
        """
        num_columnas = 3
        num_filas = int(np.ceil(n_clusters / num_columnas))

        plt.figure(figsize=(20, num_filas * 4))

        for cluster_id in range(n_clusters):
            plt.subplot(num_filas, num_columnas, cluster_id + 1)
            cluster_data = data[data['cluster'] == cluster_id].drop('cluster', axis=1)

            for idx, row in cluster_data.iterrows():
                plt.plot(row, label=f'Serie {idx}')

            plt.title(f'Cluster {cluster_id}')
            plt.legend(fontsize='small')
            plt.grid()

            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def calculate_metrics(true_values, predicted_values):
        metrics = {}
        metrics['MAE'] = mean_absolute_error(true_values, predicted_values)
        metrics['wMAPE'] = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
        metrics['RMSE'] = np.sqrt(mean_squared_error(true_values, predicted_values))
        metrics['MSE'] = mean_squared_error(true_values, predicted_values)
        return metrics


    @staticmethod
    def forecast_predictors(data_train: pd.DataFrame, data_test: pd.DataFrame, predictor: 'str', plot_compon: bool=False, plot_predict: bool=False):
        model = Prophet(daily_seasonality=False, weekly_seasonality=False, holidays_prior_scale=0)
        train = data_train[['ds', predictor]].copy()
        train = train.rename(columns={predictor:'y'})
        test = data_test[['ds', predictor]].copy()
        test = test.rename(columns={predictor:'y'})
        test['ds'] = pd.to_datetime(test['ds'])

        model.fit(train)
        forecast = model.predict(test)

        if plot_compon:
            fig_compon = model.plot_components(forecast)
        if plot_predict:
            fig_predict = model.plot(forecast)
            ax = fig_predict.get_axes()
            ax[0].plot(test['ds'], test['y'], 'r--', label='Observed data test', zorder=1)
            ax[0].set_xlabel('fecha')
            ax[0].set_ylabel(predictor)
            ax[0].set_title(f'Forecast {predictor}')
            ax[0].legend()

        return model, forecast
    
    
    @staticmethod
    def forecast_meteo_variables(data_train: pd.DataFrame, data_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        meteo_ids = data_train['meteo_id'].unique()
        meteo_variables = [col for col in data_train.columns if col not in ['ds','well_id','system_well','reference_level','level','meteo_id','system_meteo','cluster']]
        non_forecasted_columns = [col for col in data_test.columns if col not in meteo_variables]
        
        new_data_test_list = []
        evaluation_results = []

        for meteo_id in meteo_ids:
            print(f"Forecasting variables for meteo_id {meteo_id}")
            train_subset = data_train[data_train['meteo_id'] == meteo_id].copy()
            test_subset = data_test[data_test['meteo_id'] == meteo_id].copy()
            new_test = test_subset[non_forecasted_columns].copy()

            for variable in tqdm(meteo_variables):
                if train_subset[variable].isna().all():
                    continue
                _, forecast = Preprocess.forecast_predictors(
                    data_train=train_subset,
                    data_test=test_subset,
                    predictor=variable
                )
                if variable in ['prec','prec_1']:
                    forecast['yhat'] = np.where(forecast['yhat']<0, 0, forecast['yhat'])

                new_test = new_test.merge(
                    forecast[['ds', 'yhat']].rename(columns={'yhat': variable}),
                    on='ds', how='left'
                )
                eval = (
                    test_subset[['ds', variable]]
                    .merge(forecast[['ds', 'yhat']], how='left', on='ds')
                    .dropna()
                )
                metrics = Preprocess.calculate_metrics(eval[variable], eval['yhat'])
                metrics['meteo_id'] = meteo_id
                metrics['variable'] = variable
                evaluation_results.append(metrics)

            new_data_test_list.append(new_test)
        
        print(f"Creating new data test with predicted foreacsted variables")
        new_data_test = pd.concat(new_data_test_list, ignore_index=True)
        evaluation_df = pd.DataFrame(evaluation_results)

        return new_data_test, evaluation_df

