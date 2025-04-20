import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os
from google.cloud import storage
from unidecode import unidecode
from requests import Session
from requests.adapters import HTTPAdapter
from pandas import read_excel
from io import BytesIO
import yaml


class Ingestor:
    def __init__(self, station_type: str, path: str='data/', bucket: str='sqm-data-bucket', context: str='local', timeout: int=15, retries: int=4, expiration_days: int=15):
        self.context = context
        self.path = path
        self.station_type = station_type
        self.not_found = []
        self.timeout = timeout
        self.retries = retries
        self.expiration_days = expiration_days

        if self.context == 'google_drive' and not os.path.exists('/content/drive/'):
            from google.colab import drive
            drive.mount('/content/drive')

        self.data_info = self.read_info_yaml()
    
    
    def read_info_yaml(self):
        """
        Reads information from a YAML file and returns it as a pandas DataFrame.
        Args:
            station_type (str): The type of station, used to determine the YAML file to read.
                               The file is expected to be located at `self.path` with the
                               filename format `{station_type}.yaml`.
        Returns:
            pd.DataFrame: A DataFrame containing the data loaded from the YAML file.
        """
        if self.context in ['local','google_drive']:
            with open(f"{self.path}{station_type}.yaml", 'r') as file:
                data = pd.DataFrame(yaml.safe_load(file))
            return data
        
        elif self.context = 'vertex':
            blob_name = f'config-data/{station_type}.yaml'
            try:
                client = storage.Client()
                bucket = client.bucket(self.bucket_name)
                blob = bucket.blob(blob_name)
                contenido_yaml = blob.download_as_text()
                data = pd.DataFrame(yaml.safe_load(contenido_yaml))
                return data
            except Exception as e:
                print(f"Error al leer el archivo YAML desde GCS: {e}")
                return None
            except yaml.YAMLError as e:
                print(f"Error al parsear el archivo YAML: {e}")
                return None


    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans and formats a given text string.
        Args:
            text (str): The input text string to be cleaned.
        Returns:
            str: The cleaned and formatted text string.
        """
        text = unidecode(text)
        return text.lower().replace('-', '').replace('  ', '_').replace(' ', '_')


    def download_excel(self, url: str) -> dict:
        """
        Downloads an Excel file from the specified URL and reads its content into a dictionary of DataFrames.
        Args:
            url (str): The URL of the Excel file to download.
            timeout (int, optional): The timeout in seconds for the HTTP request. Defaults to 10.
        Returns:
            dict or None: A dictionary where the keys are sheet names and the values are pandas DataFrames
                            containing the sheet data. Returns None if the download or parsing fails.
        """
        data = None
        mask = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36"
        }
        try:
            with Session() as session:
                adapter = HTTPAdapter(max_retries=self.retries)
                session.mount('http://', adapter)
                session.mount('https://', adapter)
                session.trust_env = False

                response = session.get(url, timeout=self.timeout, headers=mask, verify=False)
                print(response.status_code)
                response.raise_for_status()
                data = read_excel(BytesIO(response.content), sheet_name=None)

            if data is None:
                data = pd.read_excel(url, sheet_name=None)
        except:
            pass
        return data

    def get_data_id(self, url: str, station_id: str, system: str,  subsystem: str='') -> pd.DataFrame:
        """
        Retrieves and processes data from the URL associated to the station_id.
        Args:
            url (str): The URL to download the data from.
            station_id (str): The unique identifier for the station.
            system (str): The name of the system associated with the data.
            subsystem (str, optional): The name of the subsystem associated with the data. Defaults to an empty string.
        Returns:
            pd.DataFrame: A DataFrame containing the processed data for the station.
                          Returns None if the data could not be retrieved or processed.
        Raises:
            ValueError: If there are issues with data type conversions during processing.
        """

        data = pd.DataFrame()

        if self.station_type == 'wells':
            xls = pd.read_excel(url, sheet_name=None)  # Ingestor.download_excel(url)

            if xls is None:
                self.not_found.append(station_id)
                return None

            if 'Nivel Manual' in xls.keys():
                df_well_manual = xls['Nivel Manual'][['Fecha','Nivel (msnm)','Cota Punto Referencia (msnm)']]
                df_well_manual = df_well_manual.rename(columns={'Fecha':'ds','Nivel (msnm)':'manual_level','Cota Punto Referencia (msnm)':'reference_level'})
                df_well_manual['ds'] = pd.to_datetime(df_well_manual['ds']).dt.date
                df_well_manual = df_well_manual.astype({'reference_level':str})
                df_well_manual['reference_level'] = (
                    df_well_manual['reference_level']
                    .astype(str)
                    .str.replace(r'\.', '', regex=True)
                    .str.replace(r',', '.', regex=True)
                    .astype(float, errors='raise')
                )
                data = df_well_manual.groupby('ds', as_index=False)[['manual_level','reference_level']].mean()

            if 'Nivel Continuo' in xls.keys():
                df_well_continue = xls['Nivel Continuo'][['Fecha','Nivel (msnm)']]
                df_well_continue = df_well_continue.rename(columns={'Fecha':'ds','Nivel (msnm)':'continue_level'})
                df_well_continue['ds'] = pd.to_datetime(df_well_continue['ds']).dt.date
                df_well_continue['continue_level'] = pd.to_numeric(df_well_continue['continue_level'], errors='raise')
                df_well_continue = df_well_continue.groupby('ds', as_index=False)['continue_level'].mean()
                data = data.merge(df_well_continue, how='outer', on='ds')

            data['system'] = system
            data['subsystem'] = subsystem
            data['well_id'] = station_id


        if self.station_type == 'meteo':

            xls = pd.read_excel(url, sheet_name=None)

            if xls is None:
                self.not_found.append(station_id)
                return None

            for key in xls.keys():
                df = xls[key]
                df.columns = [Ingestor.clean_text(col) for col in df.columns]
                df = df[['fecha', 'valor']].copy()
                df = df.rename(columns={'valor':Ingestor.clean_text(key).replace('meteorologia_','')})
                df = df.rename(columns={'fecha':'ds'})
                df['ds'] = pd.to_datetime(df['ds']).dt.date
                xls[key] = df

            data = None
            for df in xls.values():
                if data is None:
                    data = df
                else:
                    data = data.merge(df, how='outer', on='ds')
            data = data.sort_values(['ds'])

            data['system'] = system
            data['meteo_id'] = station_id
            
        if self.context in ['local','google_drive']:
            data.to_excel(f"{self.path}{self.station_type}/{station_id}.xlsx", index=False)
        elif self.context == 'vertex':
            data.to_excel(f"gs://{self.bucket_name}/raw-data/{self.station_type}/{station_id}.xlsx", index=False)

        return data


    def get_all_info(self, data_info: list) -> None:
        """
        Retrieves and processes information for all stations listed in data_info.
        This method iterates through value of data_info, checks if the corresponding
        station data file exists and is up-to-date, and fetches new data if necessary. If the data
        file is missing or outdated (older than 15 days), it attempts to download the data using
        the `get_data_id` method. Stations for which data could not be retrieved are added to the
        `not_found` list.
        Args:
            data_info (list): A list of dictionaries containing information about the stations.
        """

        for value in tqdm(data_info):

            file_path = f"{self.station_type}/{value['station_id']}.xlsx"
            creation_date = None
            subsystem = value['subsystem'] if 'subsystem' in value else ''
            
            if context in ['local','gogle_cloud'] and os.path.exists(f'{self.path}{file_path}'):
                creation_time = os.path.getctime(f'{self.path}{file_path}')
                creation_date = pd.to_datetime(time.ctime(creation_time)).date()
            elif contex=='vertex':
                creation_date = self.get_file_creation_date(f'raw-data/{file_path}')

            if not creation_date or pd.to_datetime('today').date() - creation_date > pd.Timedelta(days=self.expiration_days):
                try:
                    data = self.get_data_id(url=value['url'], station_id=value['station_id'], system=value['system'], subsystem=subsystem)

                    if data and value['station_id'] in self.not_found:
                        self.not_found.remove(value['station_id'])
                except:
                    self.not_found.append(value['station_id'])
                    continue


    def ingest_by_chunks(self, chunk_size: int):
        """
        Splits the data into chunks and processes each chunk.
        This method divides the data into smaller chunks based on the specified
        chunk size and processes each chunk by calling the `get_all_info` method.

        Args:
            chunk_size (int): The number of rows per chunk.
            already_files (list, optional): A list of files that have already been processed.
                Defaults to an empty list.
        """

        data_size = len(self.data_info)
        n_chunks = int(np.ceil(data_size/chunk_size))
        chunks = [self.data_info[i*chunk_size:(i*chunk_size)+chunk_size] for i in range(0, n_chunks)]

        for chunk in chunks:
            self.get_all_info(chunk)


    def retry_not_found(self):
        """
        Retries fetching information for stations that were not found in the initial attempt.
        """

        info_not_found = [v for v in self.data_info if v['station_id'] in self.not_found]

        self.timeout=30
        if len(info_not_found) > 0:
            self.get_all_info(info_not_found)
    
    
    def get_file_creation_date(self, blob_name):
        """
        Gets the creation date of a file in a GCS bucket.
        Args:
            bucket_name: The name of the GCS bucket.
            blob_name: The name of the file (blob) in the bucket.
        Returns:
            The creation date of the file as a datetime object, or None if the file
            does not exist or an error occurs.
        """
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)

            if not blob.exists():
                print(f"File {blob_name} does not exist in bucket {self.bucket_name}")
                return None

            metadata = blob.metadata

            if 'timeCreated' in metadata:
                creation_date = pd.to_datetime(metadata['timeCreated'])
            elif 'creationTime' in metadata: 
                creation_date = pd.to_datetime(metadata['creationTime'])
            else:
                creation_date = blob.updated
                print("Creation date not found in metadata, using updated time instead.")

            return creation_date
        except Exception as e:
            print(f"Error getting creation date: {e}")
            return None


if __name__ == "__main__":
    
    ingest_wells = Ingestor(
        station_type='wells', 
        bucket_name='sqm-data-bucket', 
        context='vertex', 
        expiration_days=20
    )
    ingest_meteo.ingest_by_chunks(chunk_size=4)

    if len(ingest_meteo.not_found)>0:
        print('Retrying not found meteo...')
        ingest_meteo.retry_not_found()