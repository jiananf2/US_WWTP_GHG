#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Inventory of wastewater plants in the United States and their associated energy
usage and greenhouse gas emissions

The code is developed by:
    Jianan Feng <jiananf2@illinois.edu>
'''

import pandas as pd, geopandas as gpd

WWTP_TT = pd.read_csv('tt_assignments_2022.csv')

WWTP_info =  pd.read_csv('all_wwtps_data_070124.csv')
WWTP_info = WWTP_info[['CWNS_ID','STATE_CODE','LATITUDE','LONGITUDE']]

WWTP_info.rename(columns={'CWNS_ID':'CWNS_NUM',
                          'STATE_CODE':'STATE'},
                 inplace=True)

assert WWTP_TT.duplicated(subset='CWNS_NUM').sum() == 0
assert WWTP_info.duplicated(subset='CWNS_NUM').sum() == 0

WWTP_test = WWTP_TT.merge(WWTP_info, on='CWNS_NUM', how='inner')

assert len(WWTP_TT) == len(WWTP_test)

WWTP_TT = WWTP_test

non_continental = ['HI','VI','MP','GU','AK','AS','PR']

WWTP_TT = WWTP_TT[~WWTP_TT['STATE'].isin(non_continental)]

WWTP_TT = gpd.GeoDataFrame(WWTP_TT, crs='EPSG:4269',
                           geometry=gpd.points_from_xy(x=WWTP_TT.LONGITUDE,
                                                       y=WWTP_TT.LATITUDE))
WWTP_TT = WWTP_TT.to_crs(crs='EPSG:3857')

balnc_area_map = gpd.read_file('lpreg2/lpreg2.shp')
balnc_area_map = balnc_area_map.to_crs(crs='EPSG:3857')

balnc_area_result = gpd.sjoin_nearest(WWTP_TT, balnc_area_map)
balnc_area_result = balnc_area_result[['CWNS_NUM','PCA_REG']]
balnc_area_result.rename(columns={'PCA_REG':'balancing_area'}, inplace=True)

balnc_area_result.to_excel('WWTP_balancing_area.xlsx')