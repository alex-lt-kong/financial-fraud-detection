#!/usr/bin/python3

import datetime
from decimal import Decimal
import gzip
import json
import logging
import pandas as pd
import pathlib 
import os
import random
import re
import requests
import socket
import subprocess
from subprocess import Popen, PIPE
import sys
import threading
import time
import urllib.request


# user_agent should NOT include the following elements: 
# 1. Android/iOS/Symbian/BlackBerry: some sites could feed the spider with mobile version which could not be properly recognized
# 2. IE 9.0- versions: some sites report that these versions are not supported
user_agents = [
"Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36 OPR/52.0.2871.99",
"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/18.17763", 
"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.112 Safari/537.36",
"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36", 
"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/601.7.7 (KHTML, like Gecko) Version/9.1.2 Safari/601.7.7",
"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
"Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko",
"Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0", 
"Opera/9.80 (Windows NT 6.1; WOW64) Presto/2.12.388 Version/12.18", 
"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36",        # 20191128: Copied from Chromium on Mamsds-Laptop
"Mozilla/5.0 (X11; Linux x86_64; rv:68.0) Gecko/20100101 Firefox/68.0"                                              # 20191128: Copied from Firfox on Mamsds-Laptop
]

curl_cmd = [
"/usr/bin/curl", 
"--header", "DNT: 1",   # The DNT (Do Not Track) request header indicates the user's tracking preference. It lets users indicate whether they would prefer privacy rather than personalized content.
"--header", "Accept-Language: en-US,en;",
"--header", "Accept-Encoding:", # cannot specify gzip here or the following error will be raised: <class 'UnicodeDecodeError'>; Value: 'utf-8' codec can't decode byte 0x9c in position 1
"--header", "Accept: */*",
"--header", "Cache-Control: no-store", # To turn off caching
"--location", 
"--silent",     # Silent or quiet mode. Don't show progress meter or error messages
"--user-agent", "",
"--max-time", "180", 
"--limit-rate", "50K", 
"--url", ""]
curl_ua_index = 14
curl_url_index = 20
curl_gzip_index = 6
curl_gzip_enabled = ["Accept-Encoding:", "Accept-Encoding: gzip"]

indicators = ['stock_code',
              'company_name',
              'annual_gross_margin', 
              'asset_turnover_annual',
              'cash_share_annual', # cash per share 
              'current_ratio_annual', 
              'eps_basic_excl_extra_annual', # Earnings per share excluding extraordinary items
              'fcf_share_ttm', # free cash flow per share Trailing 12 months 
              'inventory_turnover_annual',
              'lt_debt_equity_annual', # long-term debt to equity
              'net_interest_coverage_annual', # Net interest coverage ratio
              'net_profit_margin_annual', 
              'operating_margin_annual', 
              'quick_ratio_annual', 
              'receivables_turnover_annual',
              'roe_ttm', # return on equity, trailing 12 months
              'roi_annual',  # return on investment
              'roa_rfy', #  return on assets
              'tangible_book_annual',
              'total_debt_cagr_5y', 
              'total_debt_equity_annual']

    
def update_reuters_stock_data(lower: int, upper: int, parent_dir: str):


#    sleep_time = 1
    sleep_time_lower = 3
    sleep_time = sleep_time_lower
    stock_code = lower
    stock_count = 0
    start_time = time.time()
    
    results = []
    
    while stock_code < upper:

        curl_cmd[curl_ua_index] = user_agents[random.randint(0, len(user_agents) - 1)]
        url = "https://www.reuters.com/companies/api/getFetchCompanyKeyMetrics/{0:04d}.HK".format(stock_code)
        curl_cmd[curl_url_index] = url
        curl_cmd[curl_gzip_index] = curl_gzip_enabled[0]

        process = Popen(curl_cmd, stdout=PIPE)
        (curl_output, err) = process.communicate()
        exit_code = process.wait()
        curl_output = curl_output.decode("UTF-8")# curl_output must be decoded first or its excerpt cannot be concatenated with '....'
        try:
            jsonstr = json.loads(curl_output) 
        except:
            stock_code -= 1
            print("[{}] Failed parsing cURL output from Reuters: {}".format(stock_code, curl_output))
      #  print(jsonstr['status']['code'] == 200)
        if 'status' in jsonstr and 'code' in jsonstr['status']:
            
            if jsonstr['status']['code'] == 200 or jsonstr['status']['code'] == 206:
                try:
                    company_info = []
                    company_info.append(str(stock_code))
                    if 'market_data' not in jsonstr or 'company_name' not in jsonstr['market_data']:
                        continue
                        print('Perhaps not a stock: return value is\n {}'.format(jsonstr))
                    company_info.append(jsonstr['market_data']['company_name'])
                    for i in range(2, len(indicators)):
                        company_info.append(float(jsonstr['market_data'][indicators[i]]))
                #   print(company_info)
                    stock_count += 1
                    if sleep_time > sleep_time_lower:
                        sleep_time -= 0.1
                    print('[{:04d}] good: stock_count: {}, company_name: {}'.format(stock_code, stock_count, jsonstr['market_data']['company_name']))
                    results.append(company_info)
                    
                except:
                    print(sys.exc_info())
                    results.append(sys.exc_info())
            else:
                    print('[{:04d}] not good: retrying..., status_code: {}'.format(stock_code, jsonstr['status']['code']))
                    stock_code -= 1
                    sleep_time += 0.1
                    if sleep_time > 2:
                        stock_code += 1
                        sleep_time = 0.1
                        err_msg = 'too many attempts, skipped.!'
                        print(err_msg)
                        results.append([err_msg])

        else:
            print('[{:04d}] bad: status_code not exist'.format(stock_code))
        
        stock_code += 1
        time.sleep(sleep_time)
        speed = (stock_code - lower) / (time.time() - start_time)
        print('Speed: {} stock codes / sec, ETA: {} min'.format(round(speed, 1), round((upper - stock_code) * 1.05 / speed / 60, 1)))

   # print(['stock_code'].extend(indicators))
    df = pd.DataFrame(results, columns = indicators) 
    #df.replace(to_replace =-99999.99, value = 0) 
#    df.loc[df['current_ratio_annual'] < -99999, 'current_ratio_annual'] = 0
#    df.loc[df['return_on_equity_ttm'] < -99999, 'return_on_equity_ttm'] = 0
#    df.loc[df['asset_turnover_annual'] < -99999, 'asset_turnover_annual'] = 0
#    df.to_csv('/tmp/hkex-raw.csv', index = False, header=True)
#    df['current_ratio_annual']=(df['current_ratio_annual']-df['current_ratio_annual'].mean())/df['current_ratio_annual'].std()
#    df['return_on_equity_ttm']=(df['return_on_equity_ttm']-df['return_on_equity_ttm'].mean())/df['return_on_equity_ttm'].std()
#    df['asset_turnover_annual']=(df['asset_turnover_annual']-df['asset_turnover_annual'].mean())/df['asset_turnover_annual'].std()
#    print(df)
    
    df.to_csv(os.path.join(parent_dir, 'hkex_{:04d}-{:04d}.csv'.format(lower, upper - 1)), index = False, header=True)
 
if __name__ == '__main__':
    # You cannot have too many threads here!
    parent_dir = pathlib.Path(__file__).parent.absolute()
    threads = []
    threads.append(threading.Thread(target=update_reuters_stock_data, args=(   1,  2800, parent_dir)))
    threads.append(threading.Thread(target=update_reuters_stock_data, args=(2849,  3000, parent_dir)))
    threads.append(threading.Thread(target=update_reuters_stock_data, args=(3199,  4000, parent_dir)))
    threads.append(threading.Thread(target=update_reuters_stock_data, args=(4329,  4400, parent_dir)))
    threads.append(threading.Thread(target=update_reuters_stock_data, args=(4799,  5000, parent_dir)))
    threads.append(threading.Thread(target=update_reuters_stock_data, args=(6029,  6200, parent_dir)))
    threads.append(threading.Thread(target=update_reuters_stock_data, args=(6399,  6750, parent_dir)))
    threads.append(threading.Thread(target=update_reuters_stock_data, args=(6799,  7200, parent_dir)))
    threads.append(threading.Thread(target=update_reuters_stock_data, args=(7399,  7500, parent_dir)))
    threads.append(threading.Thread(target=update_reuters_stock_data, args=(7599,  9000, parent_dir)))
    threads.append(threading.Thread(target=update_reuters_stock_data, args=(9399,  9500, parent_dir)))
    threads.append(threading.Thread(target=update_reuters_stock_data, args=(9599,  9800, parent_dir)))
    threads.append(threading.Thread(target=update_reuters_stock_data, args=(9849, 10000, parent_dir)))

    for thread in threads:
        thread.start()

"""
HKEX Stock Code Allocation Plan
https://www.hkex.com.hk/-/media/HKEX-Market/Products/Securities/Stock-Code-Allocation-Plan/scap.pdf
02800-02849: ETF
03000-03199: ETF
04000-04199: HKMA Exchange Fund Notes
04200-04299: HKSAR Government Bonds
04300-04329: Debt securities for professional investors only
04400-04599: Debt securities for professional investors only
04600-04699: Professional Preference Shares
04700-04799: Debt securities for the public
05000-06029: Debt securities for professional investors only
06200-06299: HDRs
06300-06399: Securities/HDRs which are restricted securities (RS) 
06750-06799: Bonds of Ministry of the Finance of the PRC
07200-07399: Leveraged and Inverse Products
07500-07599: Leveraged and Inverse Products
09000-09199: Exchange Traded Funds (traded in USD)
09200-09399: Leveraged and Inverse Products (traded in USD)
09500-09599: Leveraged and Inverse Products (traded in USD)
09800-09849: ETF (traded in USD)
"""
