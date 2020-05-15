import zipfile
with zipfile.ZipFile("m5-forecasting-accuracy.zip","r") as zip_ref:
    zip_ref.extractall("m5-forecasting-accuracy-data")