import json
import os
from os import listdir
from os.path import isfile, join,dirname
from datetime import datetime
import torch
import numpy as np
import pandas as pd
def list_directories(pathname, predicate=None):
    if predicate is None:
        predicate = lambda x: True
    return [join(pathname, f) for f in listdir(pathname) if not isfile(join(pathname, f)) and predicate(f)]

def list_files(pathname, predicate=None):
    if predicate is None:
        predicate = lambda x: True
    return [join(pathname, f) for f in listdir(pathname) if isfile(join(pathname, f)) and predicate(f)]

def list_files_with_predicate(pathname, predicate):
    return [join(pathname, f) for f in listdir(pathname) if predicate(pathname, f)]

def mkdir_with_timestap(pathname):
    datetimetext = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    dir_name = "{}_{}".format(pathname, datetimetext)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

# 标准CSV字段顺序
CSV_FIELD_ORDER = [
    'epoch',
    'total_time',
    'time_per_epoch',
    'success_rate',
    'completion_rate',
    'undiscounted_returns',
    'sum_of_costs',
    'auc_completion',
    'auc_success',
    'training_time',
    'success_rates'
]

def save_csv(filename, data_dict, field_order=CSV_FIELD_ORDER):
    # 将列表数据转换为多行记录
    max_length = max(
        len(v) if isinstance(v, (list, np.ndarray)) else 1
        for v in data_dict.values()
    )
    
    records = []
    for i in range(max_length):
        record = {}
        for field in field_order:
            value = data_dict.get(field, None)
            
            if value is None:
                record[field] = np.nan
            elif isinstance(value, (list, np.ndarray)):
                try:
                    record[field] = round(value[i].item(),4) if hasattr(value[i], 'item') else round(float(value[i]),4) if i < len(value) else np.nan
                except IndexError:
                    record[field] = np.nan
            else:
                # 处理标量值
                if isinstance(value, float):
                    record[field] = round(value,4)
                elif 'time' in field.lower():
                    record[field] = f"{value:.2f}s"
                else:
                    record[field] = value
        records.append(record)
    
    df = pd.DataFrame.from_records(records)
    df.to_csv(filename, index=False)

def save_json(path, data, field_order=CSV_FIELD_ORDER):
    # 标准化JSON输出格式
    ordered_data = {field: data[field] for field in field_order if field in data}
    with open(path, 'w', encoding='utf-8') as data_file:
        json.dump(ordered_data, data_file, 
                 cls=TensorEncoder,
                 ensure_ascii=False,
                 indent=2)


def load_json(filename):
    data = None
    with open(filename) as data_file:    
        data = json.load(data_file)
    assert data is not None
    return data
def save_params(params, save_path, indent=4):
    """保存参数字典到指定路径"""
    os.makedirs(dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(params, f, indent=indent, cls=TensorEncoder)

class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)

def save_json(path, data):
    # 保留原始数据结构，仅转换Tensor类型
    if isinstance(data, list):
        processed_data = [
            {k: v.tolist() if isinstance(v, torch.Tensor) else v 
             for k, v in item.items()}
            for item in data
        ]
    else:
        processed_data = {
            k: v.tolist() if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
   