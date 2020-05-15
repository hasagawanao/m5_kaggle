import os
import gc
import warnings

import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

def read_data():

    print("Reading files...")

    calendar = pd.read_csv("m5-forecasting-accuracy-data/calendar.csv").pipe(reduce_mem_usage)
    prices = pd.read_csv("m5-forecasting-accuracy-data/sell_prices.csv").pipe(reduce_mem_usage)

    sales = pd.read_csv("m5-forecasting-accuracy-data/sales_train_validation.csv",).pipe(
        reduce_mem_usage
    )
    submission = pd.read_csv("m5-forecasting-accuracy-data/sample_submission.csv").pipe(
        reduce_mem_usage
    )

    print("sales shape:", sales.shape)
    print("prices shape:", prices.shape)
    print("calendar shape:", calendar.shape)
    print("submission shape:", submission.shape)

    # calendar shape: (1969, 14)
    # sell_prices shape: (6841121, 4)
    # sales_train_val shape: (30490, 1919)
    # submission shape: (60980, 29)

    return sales, prices, calendar, submission

sales, prices, calendar, submission = read_data()

NUM_ITEMS = sales.shape[0]  # 30490
DAYS_PRED = submission.shape[1] - 1  # 28

def encode_categorical(df, cols):
    for col in cols:
        # Leave NaN as it is.
        le = LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)

    return df

calendar["event_name_1"]=calendar["event_name_1"].fillna("nodata")
calendar["event_type_1"]=calendar["event_type_1"].fillna("nodata")
calendar["event_name_2"]=calendar["event_name_2"].fillna("nodata")
calendar["event_type_2"]=calendar["event_type_2"].fillna("nodata")
calendar = encode_categorical(
    calendar, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
).pipe(reduce_mem_usage)

sales = encode_categorical(
    sales, ["item_id", "dept_id", "cat_id", "store_id", "state_id"],
).pipe(reduce_mem_usage)

prices = encode_categorical(prices, ["item_id", "store_id"]).pipe(reduce_mem_usage)

def extract_num(ser):
    return ser.str.extract(r"(\d+)").astype(np.int16)


def reshape_sales(sales, submission, d_thresh=0, verbose=True):
    # melt sales data, get it ready for training
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # get product table.
    product = sales[id_columns]

    sales = sales.melt(id_vars=id_columns, var_name="d", value_name="demand",)
    sales = reduce_mem_usage(sales)

    # separate test dataframes.
    vals = submission[submission["id"].str.endswith("validation")]
    evals = submission[submission["id"].str.endswith("evaluation")]

    # change column names.
    vals.columns = ["id"] + [f"d_{d}" for d in range(1914, 1914 + DAYS_PRED)]
    evals.columns = ["id"] + [f"d_{d}" for d in range(1942, 1942 + DAYS_PRED)]

    # merge with product table
    evals["id"] = evals["id"].str.replace("_evaluation", "_validation")
    vals = vals.merge(product, how="left", on="id")
    evals = evals.merge(product, how="left", on="id")
    evals["id"] = evals["id"].str.replace("_validation", "_evaluation")

    if verbose:
        print("validation")

        print("evaluation")

    vals = vals.melt(id_vars=id_columns, var_name="d", value_name="demand")
    evals = evals.melt(id_vars=id_columns, var_name="d", value_name="demand")

    sales["part"] = "train"
    vals["part"] = "validation"
    evals["part"] = "evaluation"

    data = pd.concat([sales, vals, evals], axis=0)

    del sales, vals, evals

    data["d"] = extract_num(data["d"])
    data = data[data["d"] >= d_thresh]

    # delete evaluation for now.
    data = data[data["part"] != "evaluation"]

    gc.collect()

    if verbose:
        print("data")

    return data


def merge_calendar(data, calendar):
    calendar = calendar.drop(["weekday", "wday", "month", "year"], axis=1)
    return data.merge(calendar, how="left", on="d")


def merge_prices(data, prices):
    return data.merge(prices, how="left", on=["store_id", "item_id", "wm_yr_wk"])

data = reshape_sales(sales, submission, d_thresh=1941 - int(365 * 3.5))
del sales
gc.collect()

calendar["d"] = extract_num(calendar["d"])
data = merge_calendar(data, calendar)
del calendar
gc.collect()

data = merge_prices(data, prices)
del prices
gc.collect()

data = reduce_mem_usage(data)
data["sell_price"]=data["sell_price"].fillna(0)
data["sell_price"]=data["sell_price"].transform(lambda x: round(x,2))
data["frout"]=data["sell_price"].transform(lambda x: int(str(x).split('.')[1]))

def snap(df):
    CA1=df[df.snap_CA==1][df.state_id==0][df.sell_price!=0].groupby(["id"])["demand"].mean()
    CA0=df[df.snap_CA==0][df.state_id==0][df.sell_price!=0].groupby(["id"])["demand"].mean()
    CA=CA1-CA0
    CA.name="SNAP_CA_increase"
    df =pd.merge(df,CA,on="id",how='outer')
    df["SNAP_CA_increase"][df.snap_CA==0]=0
    df["SNAP_CA_increase"]=df["SNAP_CA_increase"].fillna(0)
    del CA1
    del CA0
    del CA
    TX1=df[df.snap_TX==1][df.state_id==1][df.sell_price!=0].groupby(["id"])["demand"].mean()
    TX0=df[df.snap_TX==0][df.state_id==1][df.sell_price!=0].groupby(["id"])["demand"].mean()
    TX=TX1-TX0
    TX.name="SNAP_TX_increase"
    df =pd.merge(df,TX,on="id",how='outer')
    df["SNAP_TX_increase"][df.snap_TX==0]=0
    df["SNAP_TX_increase"]=df["SNAP_TX_increase"].fillna(0)
    del TX1
    del TX0
    del TX
    WI1=df[df.snap_WI==1][df.state_id==2][df.sell_price!=0].groupby(["id"])["demand"].mean()
    WI0=df[df.snap_WI==0][df.state_id==2][df.sell_price!=0].groupby(["id"])["demand"].mean()
    WI=WI1-WI0
    WI.name="SNAP_WI_increase"
    df =pd.merge(df,WI,on="id",how='outer')
    df["SNAP_WI_increase"][df.snap_WI==0]=0
    df["SNAP_WI_increase"]=df["SNAP_WI_increase"].fillna(0)
    del WI1
    del WI0
    del WI
    return df

def add_demand_features(df):
    for diff in [0]: 
        shift = DAYS_PRED + diff
        df[f"demand_shift_t{shift}"] = df[df.sell_price!=0].groupby(["id"])["demand"].transform(
            lambda x: x.shift(shift))
        df[df.sell_price==0][f"demand_shift_t{shift}"] = df[df.sell_price==0][f"demand_shift_t{shift}"].fillna(0)

    for size in [7,30,60,90,180]:
        df[f"demand_rolling_std_t{size}"] = df[df.sell_price!=0].groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(size).std())
        df[df.sell_price==0][f"demand_rolling_std_t{size}"] = df[df.sell_price==0][f"demand_rolling_std_t{size}"].fillna(0)

    for size in [7,30,60,90,180]:
        df[f"demand_rolling_mean_t{size}"] = df[df.sell_price!=0].groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(size).mean())
        df[df.sell_price==0][f"demand_rolling_mean_t{size}"] = df[df.sell_price==0][f"demand_rolling_mean_t{size}"].fillna(0)
    
    return df

def add_time_features(df, dt_col):
    df[dt_col] = pd.to_datetime(df[dt_col])
    attrs = [
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "dayofweek",
    ]

    for attr in attrs:
        dtype = np.int16 if attr == "year" else np.int8
        df[attr] = getattr(df[dt_col].dt, attr).astype(dtype)

    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)
    return df

data = add_demand_features(data).pipe(reduce_mem_usage)
dt_col = "date"
data = add_time_features(data, dt_col).pipe(reduce_mem_usage)
data=snap(data).pipe(reduce_mem_usage)
data = data.sort_values("date")
print("start date:", data[dt_col].min())
print("end date:", data[dt_col].max())
print("data shape:", data.shape)

features = [
    "item_id",
    "dept_id", 
    "cat_id", 
    "store_id", 
    "state_id",
    "event_name_1",
    "event_type_1",
    "sell_price",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "SNAP_CA_increase",
    "SNAP_TX_increase",
    "SNAP_WI_increase",
    # demand features.
    "demand_shift_t28",
    "demand_rolling_std_t7",
    "demand_rolling_std_t30",
    "demand_rolling_std_t60",
    "demand_rolling_std_t90",
    "demand_rolling_std_t180",
    "demand_rolling_mean_t7",
    "demand_rolling_mean_t30",
    "demand_rolling_mean_t60",
    "demand_rolling_mean_t90",
    "demand_rolling_mean_t180",
    # time features.
    "year",
    "month",
    "week",
    "day",
    "is_weekend",
    "dayofweek",
    'quarter',
    "frout"
]
# prepare training and test data.
# 2011-01-29 ~ 2016-04-24 : d_1    ~ d_1913
# 2016-04-25 ~ 2016-05-22 : d_1914 ~ d_1941 (public)
# 2016-05-23 ~ 2016-06-19 : d_1942 ~ d_1969 (private)

mask = data["date"] <= "2016-04-24"

# Attach "date" to X_train for cross validation.
X_train = data[mask][["date"] + features].reset_index(drop=True)
y_train = data[mask]["demand"].reset_index(drop=True)
X_test = data[~mask][features].reset_index(drop=True)

# keep these two columns to use later.
id_date = data[~mask][["id", "date"]].reset_index(drop=True)
data.info()
del data
gc.collect()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

X_train.to_csv('meta_data/X_train.csv')
y_train.to_csv('meta_data/y_train.csv')
X_test.to_csv('meta_data/X_test.csv')
print("Done")