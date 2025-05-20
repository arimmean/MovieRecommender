from .base import AbstractDataset
import pandas as pd
import os
import zipfile
import requests
from pathlib import Path

class KionDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'kion'

    @classmethod
    def url(cls):
        return 'https://github.com/irsafilo/KION_DATASET/raw/f69775be31fa5779907cf0a92ddedb70037fb5ae/data_en.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['data_en.zip']

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        
        print("Raw file doesn't exist. Downloading...")
        
        # Create folder if it doesn't exist
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Download data_en.zip
        data_en_path = folder_path.joinpath('data_en.zip')
        self._download_file(self.url(), data_en_path)
        
        # Extract data_en.zip
        with zipfile.ZipFile(data_en_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
        
        # Clean up
        data_en_path.unlink()
        print()

    def _download_file(self, url, filepath):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(filepath, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('data_en', 'interactions.csv')
        df = pd.read_csv(file_path)
        
        # Rename columns to match the expected format
        df = df.rename(columns={
            'user_id': 'uid',
            'item_id': 'sid',
            'last_watch_dt': 'timestamp'
        })
        
        # Convert timestamp to unix timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
        
        # Add rating column (all 1s for implicit feedback)
        df['rating'] = 1
        
        return df[['uid', 'sid', 'rating', 'timestamp']] 