import os
import pandas as pd
from google.cloud import storage
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Consolidate:
    def __init__(self, path: str='data/', bucket_name: str='sqm-data-bucket', context: str='local'):
        self.bucket_name = bucket_name
        self.path = path
        self.context = context
        logging.info(f"Initializing Consolidate in context: '{self.context}'")

        if self.context == 'google_drive' and not os.path.exists('/content/drive/'):
            logging.info("Mounting Google Drive...")
            from google.colab import drive
            drive.mount('/content/drive')
            logging.info("Google Drive mounted successfully.")

    def list_files_in_gcs_dir(self, directory: str):
        """
        Lists the files within a specific directory in a Google Cloud Storage bucket.
        Returns:
            list: A list of strings, where each string is the name of a file
                  within the specified directory. Returns None if an error occurs.
        """
        logging.info(f"Listing files in GCS directory: 'gs://{self.bucket_name}/{directory}'")
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blobs = bucket.list_blobs(prefix=directory)
            file_names = [blob.name for blob in blobs if not blob.name.endswith('/')]
            logging.info(f"Found {len(file_names)} files in 'gs://{self.bucket_name}/{directory}'.")
            return file_names

        except Exception as e:
            logging.error(f"Error listing files in GCS: {e}")
            return None


    def save_to_parquet(self, data: pd.DataFrame, file_name: str):
        """
        Saves the processed data to a Parquet file.
        The file is saved in the specified path.
        """
        if self.context in ['local','google_drive']:
            path = f"{self.path}{file_name}.parquet"
        elif self.context == 'vertex':
            path = f"gs://{self.bucket_name}/preprocessed-data/{file_name}.parquet"
        data.to_parquet(path, index=False)
        logging.info(f"Data saved to {path}")


    def consolide_data(self, station_type: str) -> pd.DataFrame:
        """
        Consolidates data from Excel files for a specified station type.
        This method reads Excel files from a directory corresponding to the given
        station type ('wells' or 'meteo'), processes the data, and returns a
        consolidated DataFrame. The processing includes filling missing dates,
        grouping by relevant columns, and calculating mean values for numeric data.
        Args:
            station_type (str): The type of station to process.
                                 Must be either 'wells' or 'meteo'.
        Returns:
            pd.DataFrame: A consolidated DataFrame containing processed data
                          for the specified station type.
        """
        logging.info(f'Consolidating {station_type} data')
        data = pd.DataFrame()
        station_col = 'well_id' if station_type=='wells' else 'meteo_id'

        base_path = ''
        file_names = []
        if self.context in ['local','google_drive']:
            base_path = f'{self.path}{station_type}'
            try:
                file_names = os.listdir(base_path)
                logging.info(f"Found {len(file_names)} local files in '{base_path}'.")
            except FileNotFoundError:
                logging.error(f"Local directory '{base_path}' not found.")
                return pd.DataFrame()
            except Exception as e:
                logging.error(f"Error listing local files in '{base_path}': {e}")
                return pd.DataFrame()
        elif self.context == 'vertex':
            base_path = f'gs://{self.bucket_name}'
            file_names = self.list_files_in_gcs_dir(f'raw-data/{station_type}')
            if file_names is None:
                logging.warning(f"No files found or error occurred while listing GCS directory 'gs://{self.bucket_name}/raw-data/{station_type}'.")
                return pd.DataFrame()

        logging.info(f"Processing {len(file_names)} files for {station_type}.")
        for name in file_names:
            temp = pd.DataFrame()
            if name.endswith('.xlsx'):
                file_path = f'{base_path}/{name}'
                logging.info(f"Reading Excel file: '{file_path}'")
                try:
                    temp = pd.read_excel(file_path)
                    logging.info(f"Successfully read {name} with {temp.shape[0]} rows and {temp.shape[1]} columns.")
                    
                    if 'ds' in temp.columns:
                        temp['ds'] = pd.to_datetime(temp['ds']).dt.date
                        base_dt = pd.DataFrame({'ds':pd.date_range(temp.ds.min(), temp.ds.max())})
                        base_dt['ds'] = pd.to_datetime(base_dt['ds']).dt.date
                        temp = base_dt.merge(temp, how='left', on='ds').sort_values('ds').reset_index(drop=True)
                        if not temp.empty and station_col in temp.columns and not pd.isna(temp.at[0, station_col]):
                            temp[station_col] = temp.at[0, station_col]
                        elif not temp.empty and station_col not in temp.columns:
                            logging.warning(f"Column '{station_col}' not found in {name}.")
                        elif not temp.empty and station_col in temp.columns and pd.isna(temp.at[0, station_col]):
                            logging.warning(f"'{station_col}' is NaN in the first row of {name}.")

                        if not temp.empty and 'system' in temp.columns and not pd.isna(temp.at[0, 'system']):
                            temp['system'] = temp.at[0,'system']
                        elif not temp.empty and 'system' in temp.columns and pd.isna(temp.at[0, 'system']):
                            temp['system'] = None
                            logging.warning(f"'system' is NaN in the first row of {name}.")
                        elif not temp.empty and 'system' not in temp.columns:
                            temp['system'] = None
                            logging.info(f"Column 'system' not found in {name}, set to None.")

                        group_cols = ['ds',station_col,'system']
                        if station_type == 'wells' and 'reference_level' in temp.columns:
                            if not pd.isna(temp.at[0,'reference_level']):
                                temp['reference_level'] = temp.at[0,'reference_level']
                                group_cols.append('reference_level')
                            else:
                                temp['reference_level'] = None
                                logging.warning(f"'reference_level' is NaN in the first row of {name}.")
                        elif station_type == 'wells' and 'reference_level' not in temp.columns:
                            temp['reference_level'] = None
                            logging.info(f"Column 'reference_level' not found in {name}, set to None.")

                        if not temp.empty:
                            temp = temp.groupby(group_cols, as_index=False).mean()
                            data = pd.concat([data, temp], ignore_index=True)
                    else:
                        logging.warning(f"Column 'ds' not found in {name}, skipping file.")

                except FileNotFoundError:
                    logging.error(f"File not found: '{file_path}'")
                except Exception as e:
                    logging.error(f"Error reading or processing '{file_path}': {e}")

        if not data.empty and station_col in data.columns:
            data[station_col] = data[station_col].astype('string')
            logging.info(f"Consolidated {station_type} data with {data.shape[0]} rows and {data.shape[1]} columns.")
        else:
            logging.warning(f"No data consolidated for {station_type}.")

        return data

if __name__ == "__main__":
    logging.info("Starting data consolidation process.")
    conso = Consolidate(bucket_name='sqm-data-bucket', context='vertex')

    logging.info("Consolidating wells data...")
    wells_data = conso.consolide_data(station_type='wells')
    if not wells_data.empty:
        conso.save_to_parquet(wells_data, file_name='wells_data')
    else:
        logging.warning("No wells data to save.")

    logging.info("Consolidating meteo data...")
    meteo_data = conso.consolide_data(station_type='meteo')
    if not meteo_data.empty:
        conso.save_to_parquet(meteo_data, file_name='meteo_data')
    else:
        logging.warning("No meteo data to save.")

    logging.info("Data consolidation process finished.")