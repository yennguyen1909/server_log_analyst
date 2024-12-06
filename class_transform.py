import pandas as pd
import numpy as np
import os
import re

class Transform:
    """
    Transforming a log file into a DataFrame involves reading the log file, parsing the necessary information, 
    and converting it into a structured format suitable for data analysis using a tool like Pandas, Vaex, ... 
    Pandas DataFrame is a two-dimensional size-mutable, potentially heterogeneous tabular data structure with 
    labeled axes (rows and columns).
    Vaex is a powerful Python library designed for handling large tabular datasets efficiently. It provides 
    out-of-core dataframes, which means it can process data that doesn't fit into memory by reading and processing 
    data in chunks. This makes it particularly useful for working with big data.

    Input: File data about Server log with 10 types information, including: std, remote_address, remote_user, 
    datetime, method, path, header,	status,	bytes_sent,	referer	user_agent 
    
    from parse: 
    %{IP:remote_address} - %{DATA:remote_user} \[%{HTTPDATE:time_local}\] \"%{WORD:http_method} %{URIPATHPARAM:path} 
    %{DATA:header}\" %{NUMBER:status} %{NUMBER:body_bytes_sent:long} \"%{DATA:http_referer}\" \"%{DATA:http_user_agent}\" 
    kong_request_id: \"%{WORD:kong_request_id}\"

    Output: A Pandas or Vaex Dataframe includes 10 columns 
    """
    file_log= r'D:\code-python\project_practice\server_log\0.log.20241201-211909\0.log.20241201-211909'
    log_pattern= r'^\S+ (?P<std>\S+) \S+ (?P<remote_address>\d+\.\d+\.\d+\.\d+) - (?P<remote_user>[^ ]*) \[(?P<datetime>[^\]]+)\] "(?P<method>\w+) (?P<path>[^\s]+) (?P<header>[^\"]+)" (?P<status>\d+) (?P<bytes_sent>\d+) "(?P<referer>[^\"]*)" "(?P<user_agent>[^\"]*)"'
    def read_log_file(file_path, pattern):
        with open(file_path, 'r') as file:
            log_lines = file.readlines()
            parsed_logs = [re.match(pattern, line).groupdict() for line in log_lines if re.match(pattern, line)]
        return parsed_logs
    
    def transform(parsed_logs):
        df = pd.DataFrame(parsed_logs)
        return df

parsed_logs = Transform.read_log_file(Transform.file_log, Transform.log_pattern) 
df = Transform.transform(parsed_logs) 

#Data Preprocessing

df['std'] = df['std'].replace({'stdout': 0, 'stderr': 1})
from datetime import datetime
df['datetime'] = df['datetime'].apply(lambda x: datetime.strptime(x, "%d/%b/%Y:%H:%M:%S %z"))
df = df.drop(['remote_user'], axis = 1)
df = df.drop(['header'], axis = 1)
df['method'] = df['method'].replace({'GET': 0, 'POST': 1})
df ['bytes_sent'] = pd.to_numeric(df['bytes_sent'])
df['status'] = pd.to_numeric(df['status'])
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df.isnull().sum()

print (df.head())
