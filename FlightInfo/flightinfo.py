# -*- coding: utf-8 -*-

import sys
import pandas as pd

DJI_data = pd.read_csv(sys.argv[1], low_memory=False)
DJI_data = DJI_data.iloc[:, [9, 10, 11, 73, 74, 77, 78]].dropna()
DJI_data = DJI_data.groupby("GPS:dateTimeStamp").mean().reset_index()
DJI_data["GPS:dateTimeStamp"] = pd.to_datetime(DJI_data["GPS:dateTimeStamp"], utc=True)
DJI_data["Image:Time"] = DJI_data["GPS:dateTimeStamp"] + pd.Timedelta(hours=int(sys.argv[2]), minutes=int(sys.argv[3]), seconds=int(sys.argv[4]))
DJI_data.to_csv("flightinfo.csv")