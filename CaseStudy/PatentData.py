#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np

patent = pd.read_csv("./sea_managing_raw.csv")
# print(patent.head())
#   국가코드 DB종류 특허/실용 구분 문헌종류 코드  ... Original FI[JP] Original F-term[JP] Original Theme Code [JP]   WIPS ON key
# 0   US   US        P      B2  ...             NaN                 NaN                      NaN  4.914000e+12
# 1   US   US        P      A1  ...             NaN                 NaN                      NaN  5.414000e+12
# 2   US   US        P      A1  ...             NaN                 NaN                      NaN  5.414000e+12
# 3   US   US        P      A1  ...             NaN                 NaN                      NaN  5.414000e+12
# 4   US   US        P      B2  ...             NaN                 NaN                      NaN  4.913050e+12
# print("\n")

# print(patent["NUMBER"].isnull().sum())
# 0
df_patent = patent[["NUMBER", "Original US Class All[US]"]]
# print(df_patent.head())
#       NUMBER                          Original US Class All[US]
# 0  11/644784                                134/166.R | 134/138
# 1  13/848244  106/627 | 106/600 | 106/631 | 106/632 | 106/63...
# 2  13/765924                                            204/554
# 3  13/767408                                  588/002 | 588/315
# 4  12/521154  210/709 | 210/711 | 210/712 | 210/713 | 210/71...
# print("\n")

# print(df_patent["Original US Class All[US]"].map(lambda x : x.split("|")).tolist())
edge_list = []
for data in zip(df_patent["NUMBER"].tolist(), df_patent["Original US Class All[US]"].map(
        lambda x : x.split("|")).tolist()):
        for value in data[1]:
            edge_list.append([data[0], value.strip()])
# print(edge_list[:10])
# [['11/644784', '134/166.R'],
#  ['11/644784', '134/138'],
#  ['13/848244', '106/627'],
#  ['13/848244', '106/600'],
#  ['13/848244', '106/631'],
#  ['13/848244', '106/632'],
#  ['13/848244', '106/634'],
#  ['13/848244', '210/702'],
#  ['13/765924', '204/554'],
#  ['13/767408', '588/002']]

df_edge_list = pd.DataFrame(edge_list)
# print(df_edge_list.head())
#            0          1
# 0  11/644784  134/166.R
# 1  11/644784    134/138
# 2  13/848244    106/627
# 3  13/848244    106/600
# 4  13/848244    106/631

df_edge_list["rating"]=1
print(df_edge_list.groupby([0,1])["rating"].sum().unstack().fillna(0).head(10))
# 1          004/114.1  004/144.1  004/222  004/459  004/679  ...  901/031  977/742  977/840  977/932  D25/113
# 0                                                           ...
# 05/583686        0.0        0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0      0.0
# 05/634273        0.0        0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0      0.0
# 05/636159        0.0        0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0      0.0
# 05/677447        0.0        0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0      0.0
# 05/701218        0.0        0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0      0.0
# 05/772460        0.0        0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0      0.0
# 05/801551        0.0        0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0      0.0
# 05/807560        0.0        0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0      0.0
# 05/809031        0.0        0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0      0.0
# 05/935641        0.0        0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0      0.0
# [10 rows x 2212 columns]
