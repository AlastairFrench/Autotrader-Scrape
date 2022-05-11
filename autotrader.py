#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 18:49:29 2020

@author: Alastair
"""
#import libraries
from bs4 import BeautifulSoup as Soup
import requests
from pandas import DataFrame
from os import path
import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import math
import statsmodels.formula.api as smf


#set directory
DATA_DIR = '/Users/Alastair/Desktop/python/autotrader'

#fuction to parse the specs of the listings in the HTML document
def parse_specs(spec):
    """
    Take in a li tag and get the data out of it in the form of a list of
    strings.
    """
    return [str(x.string) for x in spec.find_all('li', limit=5)]

def scrape_autotrader(page):
    """
    Scrapes the autotrader page
    """
    #scrape auto trader specs
    au_response = requests.get(f'https://www.autotrader.co.uk/car-search?postcode=L17%208XX&make=VOLKSWAGEN&model=POLO&homeDeliveryAdverts=include&quantity-of-doors=5&minimum-badge-engine-size=1.2&maximum-badge-engine-size=1.6&onesearchad=Used&onesearchad=Nearly%20New&onesearchad=New&advertising-location=at_cars&page={page}')
    
    au_soup = Soup(au_response.text)
        
    specs = au_soup.findAll('ul', class_='listing-key-specs')
    
    list_of_parsed_specs = [parse_specs(spec) for spec in specs[0:]]
    
    # put it in a dataframe
    df = DataFrame(list_of_parsed_specs)
    
    # replacing 'nones' and then shifting values across and dropping columns
    df.replace(to_replace=['None'], value=np.nan, inplace=True)
    
    shifted_df = df.apply(lambda x: pd.Series(x.dropna().values), axis=1).fillna('')

    dropped_df = shifted_df.drop(df.columns[[1, 4]], axis=1) 
    
    # cleaning data
    dropped_df[0] = dropped_df[0].str[:4]   
    dropped_df[2] = dropped_df[2].str.split(' ').str[0]
    dropped_df[2] = dropped_df[2].str.replace(',', '')
    dropped_df[[0, 2]] = dropped_df[[0, 2]].astype(int)
    
    # find price
    prices = [price.text for price in au_soup.findAll('div', class_='vehicle-price')]
        
    df_prices = DataFrame(prices)
    
    # cleaning data
    df_prices[0] = df_prices[0].str.replace(',', '')
    df_prices[0] = df_prices[0].str.replace('£', '')
    df_prices[0] = df_prices[0].astype(int)
    
    # merging dfs
    df = pd.merge(df_prices, dropped_df, left_index=True, right_index=True)

    df.columns = ['price', 'year', 'mileage', 'engine']
    
    #save as a .csv
    return df.to_csv(path.join(DATA_DIR, f'scrape_data_page_{page}.csv') index=False)


#loop to get .csv fiels for all pages for that model of vehicle
pages = list(range(1,101))

i=0
for page in pages:
    scrape_autotrader(page)
    i=i+1
    

#merging csvs
import pandas as pd
#set working directory
os.chdir('/Users/Alastair/Desktop/python/autotrader')

#find all csv files in the folder
#use glob pattern matching -> extension = 'csv'
#save result in list -> all_filenames
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ], ignore_index=True)

combined_csv.drop(combined_csv.columns[0], axis=1, inplace=True)

#export to csv new combined
combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')


#plotting data 
df = pd.read_csv(path.join(DATA_DIR, 'combined_csv.csv'))

df.corr()

g = sns.relplot(x='year', y='price', hue='engine', data=df)
g.fig.subplots_adjust(top=0.9) # adding a title
g.fig.suptitle('VW Polo: Price vs. Year')
g.set_xlabels('Year') # modifying axis
g.set_ylabels('Price (£)')
g.savefig('vw_polo_year_14L.png')  # saving

g = sns.relplot(x='mileage', y='price', hue='engine', data=df)
g.fig.subplots_adjust(top=0.9) # adding a title
g.fig.suptitle('VW Polo: Price vs. Year')
g.set_xlabels('Mileage') # modifying axis
g.set_ylabels('Price (£)')
g.add_legend()  # adding legend
g.savefig('vw_polo_mileage_1.4L.png')  # saving


#modelling to predict price of car from age and mileage
df = pd.read_csv(path.join(DATA_DIR, 'combined_csv.csv'))
#df = df.loc[(df['engine'] == '1.4L')]

#visualising trendlines
g = sns.regplot(x='year', y='price', data=df)
g = sns.regplot(x='mileage', y='price', data=df)

df['ln_price'] = np.log(df['price'])

#visualising trendlines
g = sns.regplot(x='year', y='ln_price', data=df)
g = sns.regplot(x='mileage', y='ln_price', data=df)

# run regression
model = smf.ols(formula='ln_price ~ year + mileage',data=df)
results = model.fit()
results.summary2()

def results_summary_to_dataframe(results):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })

    #Reordering...
    results_df = results_df[["coeff","pvals","conf_lower","conf_higher"]]
    return results_df

results_df = results_summary_to_dataframe(results) 

#input age and mileage of car
year = 2011
mileage = 90000

#use linear regression to predict price
ln_price = results_df.iat[0,0] + year*results_df.iat[1,0] + mileage*results_df.iat[2,0]
import math
price = math.exp(ln_price)
print("%.0f" % price)       


