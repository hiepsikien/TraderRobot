import wget
import config as cf
import os
import zipfile
import glob
import pandas as pd

def download_spot_monthly(symbol:str,year:int,month:int,interval:str):
    
    save_dir = f"{cf.SPOT_MONTHLY_KLINES_DATA_PATH}{symbol}/{interval}"
    filename = f"{save_dir}/{symbol}-{interval}-{year}-{month:02}.zip"
    if os.path.exists(filename):
        print(f"Already existed {filename}")
        return save_dir
    
    url = f"{cf.BINANCE_SPOT_MOTHLY_KLINES_URL}{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02}.zip"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        str = wget.download(url,save_dir)
        print(str)
        return save_dir
    except Exception as e:
        print(f"{e} {url}")
        return 0
    
def download_futures_monthly(symbol:str,year:int,month:int,interval:str):
    save_dir = f"{cf.FUTURE_MONTHLY_KLINES_DATA_PATH}{symbol}/{interval}"
    filename = f"{save_dir}/{symbol}-{interval}-{year}-{month:02}.zip"
    if os.path.exists(filename):
        # print(f"Already existed {filename}")
        return save_dir
    url = f"{cf.BINANCE_FUTURE_MOTHLY_KLINES_URL}{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02}.zip"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        str = wget.download(url,save_dir)
        print(str)
        return save_dir
    except Exception as e:
        # print(f"{e} {url}")
        return 0
    
def download_spot_bulk(symbol:str,interval:str,start_year:int,start_month:int,end_year:int,end_month:int):
    count = 0
    result = None
    for year in range(start_year,end_year+1,1):
        s_month = start_month if year == start_year else 1
        e_month = end_month if year == end_year else 12
        for month in range(s_month,e_month+1,1):
            result = download_spot_monthly(symbol,year,month,interval)
            if result != 0:
                count+=1
    print(f"Total downloaded {count} files")
    return result
    
def download_futures_bulk(symbol:str,interval:str,start_year:int,start_month:int,end_year:int,end_month:int):
    count = 0
    result = None
    for year in range(start_year,end_year+1,1):
        s_month = start_month if year == start_year else 1
        e_month = end_month if year == end_year else 12
        for month in range(s_month,e_month+1,1):
            result = download_futures_monthly(symbol,year,month,interval)
            if result != 0:
                count+=1
    print(f"Total downloaded {count} files")
    return result
    
def unzip(source_dir):
    files = glob.glob(f"{source_dir}/*.zip")
    for file in files:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(source_dir)
            print(f"{file}")
    return source_dir
            
def import_data_from_multiple_csv(dir):
    col_names = ["Open","High","Low","Close","Volumn","Close Time", "Quote Asset Volumn", "Number of Trades", "Taker Buy Base Asset Volumn", "Taker Buy Quote Asset Volumn", "Ignore" ]
    filenames = glob.glob(f"{dir}/*.csv")
    dfs = []
    
    for filename in filenames:
        print(filename)
        df = pd.read_csv(filename, names = col_names)
        if (df.iloc[0:1]["Open"].values[0] == "open"):
            df.drop(index=df.index[0],axis=0,inplace=True)
        dfs.append(df)

    data = pd.concat(dfs)
    data.index.name = "Date"
    data.index = data.index.astype(int)
    data["Open"] = data["Open"].astype(float)
    data["High"] = data["High"].astype(float)
    data["Low"] = data["Low"].astype(float)
    data["Close"] = data["Close"].astype(float)
    data["Volumn"] = data["Volumn"].astype(float)
    data["Close Time"] = data["Close Time"].astype(int)
    data["Quote Asset Volumn"] = data["Quote Asset Volumn"].astype(float)
    data["Number of Trades"] = data["Number of Trades"].astype(int)
    data["Taker Buy Base Asset Volumn"] = data["Taker Buy Base Asset Volumn"].astype(float)
    data["Taker Buy Quote Asset Volumn"] = data["Taker Buy Quote Asset Volumn"].astype(float)
    data["Ignore"] = data["Ignore"].astype(str)
    data = data.sort_index(ascending=True)
    data["Date Time"] = pd.to_datetime(data.index,unit="ms",utc=True)
    data.index.name = "Open Time"

    return data

def write_data_to_hdf(save_dir:str,data:pd.DataFrame,symbol:str,interval:str):
    data.to_hdf(f"{save_dir}/{symbol}-{interval}.h5",key="data")
    
def download_spot_and_save_to_hdf(symbol:str,interval:str,start_year:int,start_month:int,end_year:int,end_month:int):
    print("...downloading ...")
    dir = download_spot_bulk(symbol,interval,start_year,start_month,end_year,end_month)
    print("...unzipping...")
    dir = unzip(dir)
    print("...merging...")
    data = import_data_from_multiple_csv(dir)
    print("...saving...")
    save_dir = f"{cf.AGGREGATED_DATA_PATH}spot/"
    write_data_to_hdf(data=data,symbol=symbol,interval=interval,save_dir=save_dir)
    print("Completed")   
    
def download_futures_and_save_to_hdf(symbol:str,interval:str,start_year:int,start_month:int,end_year:int,end_month:int):
    print("...downloading ...")
    dir = download_futures_bulk(symbol,interval,start_year,start_month,end_year,end_month)
  
    if dir is None:
        return 0
    print("...unzipping...")
    dir = unzip(dir)
    if dir is None:
        return 0
    print("...merging...")
    data = import_data_from_multiple_csv(dir)
    print("...saving...")
    save_dir = f"{cf.AGGREGATED_DATA_PATH}futures/"
    write_data_to_hdf(data=data,symbol=symbol,interval=interval,save_dir=save_dir)
    print("Completed")   