import pandas as pd
import numpy as np
import math
import os
from datetime import date
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class ValExtract:

    def __init__(self, url):
        
        self.url = url
        self.data = self.load_data(url)
        self.val_date = pd.Timestamp(date.today().year, date.today().month, 1)
        self.no_dead_date = pd.Timestamp(2199, 1, 1)
        self.data_transformation(self.val_date)
        
        #self.snapshot = self.take_snapshot(self.val_date)

    def load_data(self, urls: str = None):
        
        df = pd.DataFrame()
        if urls != None:
            if isinstance(urls, str):
                #df = pd.read_csv(urls)
            #elif isinstance(urls, list):
            #    for url in urls:
            #        df.append(pd.read_csv(url))
                files = os.listdir(urls)
                files = [file for file in files if file.endswith('.csv')]
                print(files)
                for file in files:
                    deal_name = file.split('+')[0]
                    one_deal = pd.read_csv(urls + '\\' + file,
                                   usecols=['Inventory Date',
                                           'POL_NUMBER',
                                           'SLICE_NUMBER',
                                           'Date of Birth Life 1',
                                           'SEX',
                                           'Date of Birth Life 2',
                                           'SEX2',
                                           'ANN_ANNUITY',
                                           'ANN_ANNUITY_INCEPTION',
                                           'JL_INDICATOR',
                                           'Date of Death Life 1',
                                           'Date of Death Life 2',
                                           ],)
                    one_deal['Deal Name'] = deal_name
                    
                    df = df.append(one_deal, ignore_index=True)
                    print(file)

        return df
                    

    def data_transformation(self, val_date):


        # Aggregate policy slices into unique lives
        self.data['POL_NUMBER'] = self.data['POL_NUMBER'].astype(str).str.split('_',expand=True)[0]
        self.data['POL_NUMBER'] = self.data['POL_NUMBER'] + '_' + self.data['Date of Birth Life 1'] + '_' + \
                                  self.data['SEX']
        self.data = self.data.groupby('POL_NUMBER').agg({
                                                         'Inventory Date': 'first',
                                                         'Deal Name': 'first',
                                                         'Date of Death Life 1': 'first',
                                                         'Date of Birth Life 1': 'first',
                                                         'Date of Death Life 2': 'first',
                                                         'Date of Birth Life 2': 'first',
                                                         'SEX': 'first',
                                                         'SEX2': 'first',
                                                         'ANN_ANNUITY' : 'sum',
                                                         'ANN_ANNUITY_INCEPTION' : 'sum',
                                                         'JL_INDICATOR' : 'first'            
                                                        }).reset_index()
        

      # Converting DoD Life1 into timestamp in a new column and using this column as inputs to get month and year of death Life 1
        self.data['Date of Death Life 1'] = pd.to_datetime(self.data['Date of Death Life 1'], dayfirst = True)
        self.data['Date of Birth Life 1'] = pd.to_datetime(self.data['Date of Birth Life 1'], dayfirst = True)
        self.data['Date of Death Life 2'] = pd.to_datetime(self.data['Date of Death Life 2'], dayfirst = True)
        self.data['Date of Birth Life 2'] = pd.to_datetime(self.data['Date of Birth Life 2'], dayfirst = True)
        
        #self.data['Date of Death Life 1'].fillna(value = val_date, inplace = True)
        #self.data['Date of Death Life 2'].fillna(value = val_date, inplace = True)
        
        self.data['MoD_Life1'] = self.data['Date of Death Life 1'].dt.month
        self.data['YoD_Life1'] = self.data['Date of Death Life 1'].dt.year
        self.data['MoD_Life2'] = self.data['Date of Death Life 2'].dt.month
        self.data['YoD_Life2'] = self.data['Date of Death Life 2'].dt.year
        
        # Calculating current age (irrespective if policy has died or not)
        self.data['Curr_age'] = (val_date - self.data['Date of Birth Life 1']).dt.days /365.25
        self.data.loc[~self.data['Date of Birth Life 2'].isna(), 'Curr_Age_2'] = (val_date - self.data['Date of Birth Life 2']).dt.days/365.25
        #self.data.loc[self.data['Date of Birth Life 2'].isna(), 'Curr_Age_2'] = None
                        
        # Calculating death age for all policies (ignoring if policy are still alive at valuation date)
        self.data['Age_at_death'] = (self.data['Date of Death Life 1'] - self.data['Date of Birth Life 1']).dt.days / 365.25
        self.data['Age_at_death_2'] = (self.data['Date of Death Life 2'] - self.data['Date of Birth Life 2']).dt.days / 365.25
        

        self.data.loc[~self.data['Date of Birth Life 2'].isna(), 'Has Life 2'] = 1
        self.data.loc[self.data['Date of Birth Life 2'].isna(), 'Has Life 2'] = 0
        
        return self.data


    def take_snapshot(self, date):
        self.data['Life 2 Inclusion'] = 0
        self.data.loc[(self.data['Has Life 2'] == 1) &
                      (self.data['Date of Death Life 1'] < date) &
                      ((self.data['Date of Death Life 2'] > self.data['Date of Death Life 1']) |
                       (self.data['Date of Death Life 2'].isna())),
                      'Life 2 Inclusion'] = 1
        self.snapshot = self.data.copy()
        
        self.snapshot['Which_life'] = 1
        life_1_temp = self.snapshot[[#'Inventory Date',
                                     'POL_NUMBER',
                                     #'SLICE_NUMBER',
                                     'Which_life',
                                     'Date of Birth Life 1',
                                     'SEX',
                                     'ANN_ANNUITY',
                                     'Date of Death Life 1',
                                     'Curr_age',
                                     'Age_at_death',
                                     'Deal Name']]
        
        life_1_temp.rename(columns = {'Date of Birth Life 1':'Date of Birth',
                                      'Date of Death Life 1':'Date of Death'},inplace=True)
        
        
        life_2_temp = self.snapshot.loc[(self.snapshot['Has Life 2'] == 1) &
                                      (self.snapshot['Life 2 Inclusion'] == 1),
                                      [#'Inventory Date',
                                       'POL_NUMBER',
                                       #'SLICE_NUMBER',
                                       'Which_life',
                                       'Date of Birth Life 2',
                                       'SEX2',
                                       'ANN_ANNUITY',
                                       'Date of Death Life 2',
                                       'Curr_Age_2',
                                       'Age_at_death_2',
                                       'Deal Name']]
        life_2_temp['Which_life'] = 2
        life_2_temp.rename(columns = {'Date of Birth Life 2':'Date of Birth',
                                      'Date of Death Life 2':'Date of Death',
                                      'Curr_Age_2':'Curr_age',
                                      'Age_at_death_2':'Age_at_death'},inplace=True)
        snapshot_preprocess = life_1_temp.append(life_2_temp)
        
        snapshot_preprocess['Inforce'] = 0
        snapshot_preprocess.loc[(snapshot_preprocess['Date of Death'].isna() |   
                                (snapshot_preprocess['Date of Death'] >= date)), 'Inforce'] = 1
        snapshot_preprocess[f'Death in {date.strftime("%Y_%m")}'] = 0
        snapshot_preprocess.loc[(snapshot_preprocess['Date of Death'].dt.year == date.year) &   
                                (snapshot_preprocess['Date of Death'].dt.month == date.month),
                                f'Death in {date.strftime("%Y_%m")}'] =1
        return snapshot_preprocess
    
    # For average age of death distribution    
    def take_snapshot_ddistri(self, i_date, e_date):
        self.data['Life 2 Inclusion'] = 0
        self.data.loc[(self.data['Has Life 2'] == 1) &
                      (self.data['Date of Death Life 1'] < e_date) &
                      ((self.data['Date of Death Life 2'] > self.data['Date of Death Life 1']) &
                       (((self.data['Date of Death Life 2'].dt.year == self.data['Date of Death Life 1'].dt.year) & (self.data['Date of Death Life 2'].dt.month > self.data['Date of Death Life 1'].dt.month))|
                        (self.data['Date of Death Life 2'].dt.year > self.data['Date of Death Life 1'].dt.year))|
                       (self.data['Date of Death Life 2'].isna())),
                      'Life 2 Inclusion'] = 1
        self.snapshot = self.data.copy()
        
        self.snapshot['Which_life'] = 1
        life_1_temp = self.snapshot[[#'Inventory Date',
                                     'POL_NUMBER',
                                     #'SLICE_NUMBER',
                                     'Which_life',
                                     'Date of Birth Life 1',
                                     'SEX',
                                     'ANN_ANNUITY',
                                     'Date of Death Life 1',
                                     'Curr_age',
                                     'Age_at_death',
                                     'Deal Name']]
        
        life_1_temp.rename(columns = {'Date of Birth Life 1':'Date of Birth',
                                      'Date of Death Life 1':'Date of Death'},inplace=True)
        
        
        life_2_temp = self.snapshot.loc[(self.snapshot['Has Life 2'] == 1) &
                                      (self.snapshot['Life 2 Inclusion'] == 1),
                                      [#'Inventory Date',
                                       'POL_NUMBER',
                                       #'SLICE_NUMBER',
                                       'Which_life',
                                       'Date of Birth Life 2',
                                       'SEX2',
                                       'ANN_ANNUITY',
                                       'Date of Death Life 2',
                                       'Curr_Age_2',
                                       'Age_at_death_2',
                                       'Deal Name']]
        life_2_temp['Which_life'] = 2
        life_2_temp.rename(columns = {'Date of Birth Life 2':'Date of Birth',
                                      'Date of Death Life 2':'Date of Death',
                                      'Curr_Age_2':'Curr_age',
                                      'Age_at_death_2':'Age_at_death'},inplace=True)
        snapshot_preprocess = life_1_temp.append(life_2_temp)
        
        snapshot_preprocess['Inforce'] = 0
        snapshot_preprocess.loc[(snapshot_preprocess['Date of Death'].isna() |   
                                (snapshot_preprocess['Date of Death'] >= e_date)), 'Inforce'] = 1
        snapshot_preprocess[f'Death by {i_date.strftime("%Y_%m")}'] = 0
        snapshot_preprocess.loc[(snapshot_preprocess['Date of Death'].dt.year == i_date.year) &   
                                (snapshot_preprocess['Date of Death'].dt.month <= i_date.month),
                                f'Death by {i_date.strftime("%Y_%m")}'] =1
        
        snapshot_preprocess = snapshot_preprocess[snapshot_preprocess[f'Death by {i_date.strftime("%Y_%m")}'] ==1]
        return snapshot_preprocess        
        
    def count_all(self, data ,date):
        df_death = data[(data[f'Death in {date.strftime("%Y_%m")}'] ==1)]
        df_deathL1 = data[((data.Which_life == 1) & (data[f'Death in {date.strftime("%Y_%m")}'] ==1))]
        df_deathL2 = data[((data.Which_life == 2) & (data[f'Death in {date.strftime("%Y_%m")}'] ==1))]        
        df_alive = data[data.Inforce == 1]
        df_aliveL1 = data[((data.Which_life == 1) & (data['Inforce'] ==1))]
        df_aliveL2 = data[((data.Which_life == 2) & (data['Inforce'] ==1))]          
        grp_death = df_death.groupby('Deal Name')
        grp_deathL1 = df_deathL1.groupby('Deal Name')
        grp_deathL2 = df_deathL2.groupby('Deal Name')
        grp_alive = df_alive.groupby('Deal Name')
        grp_aliveL1 = df_aliveL1.groupby('Deal Name')
        grp_aliveL2= df_aliveL2.groupby('Deal Name')
        death = grp_death[f'Death in {date.strftime("%Y_%m")}'].count()
        
        if df_deathL1.empty:
            life1_death = 0
        else:
            life1_death = grp_deathL1[f'Death in {date.strftime("%Y_%m")}'].count()
            
        if df_deathL2.empty:
            life2_death = 0
        else:
            life2_death = grp_deathL2[f'Death in {date.strftime("%Y_%m")}'].count()
            
        alive = grp_alive.Inforce.count()
        aliveL1 = grp_aliveL1.Inforce.count()
        aliveL2 = grp_aliveL2.Inforce.count()
        norm_death = death / alive * 1000
        sum_age_death = grp_death.Age_at_death.sum()
        sum_age_inforce = grp_alive.Curr_age.sum()
        result_df = pd.DataFrame({'Date': date,
                                  'Inforce (BOM)': alive,
                                  'Death count': death,
                                  'Norm death': norm_death,
                                  'Sum age inforce': sum_age_inforce,
                                  'Sum age death': sum_age_death,
                                  'Life1 death': life1_death,
                                  'Life2 death': life2_death,
                                  'Life1 inforce': aliveL1,
                                  'Life2 inforce': aliveL2})
        return result_df        
        # result_df is deal specific table with metrics
        
    def get_aggregate(self, data):
        grp_df = data.groupby('Date')
        agg_death = grp_df['Death count'].sum()
        agg_l1_death = grp_df['Life1 death'].sum()
        agg_l2_death = grp_df['Life2 death'].sum()
        agg_alive = grp_df['Inforce (BOM)'].sum()
        agg_l1_alive = grp_df['Life1 inforce'].sum()
        agg_l2_alive = grp_df['Life2 inforce'].sum()
        agg_sum_age_death = grp_df['Sum age death'].sum()
        agg_sum_age_inforce = grp_df['Sum age inforce'].sum()
        agg_norm_death = agg_death / agg_alive * 1000
        agg_df = pd.DataFrame({'Inforce (BOM)': agg_alive,
                               'Death count': agg_death,
                               'Norm death': agg_norm_death,
                               'Sum age inforce': agg_sum_age_inforce,
                               'Sum age death': agg_sum_age_death,
                               'Life1 death': agg_l1_death,
                               'Life2 death': agg_l2_death,
                               'Life1 inforce': agg_l1_alive,
                               'Life2 inforce': agg_l2_alive})
        agg_df = agg_df.reset_index()
        return agg_df
    
    def make_ref_period(self, data, period):
        data.loc[:,'ref_period'] = 0              # 1 indicates reference period, 0 indicates active period
        data.loc[:,'Month'] = data.Date.dt.month
        period = period - 1
        data.loc[:period, 'ref_period'] = 1
        return data
        
    def cumulative(self, data, period):
        grp_df = data.groupby('Month')
        #data.loc[:,'Cumulative Inforce (BOM)'] = data['Inforce (BOM)'].cumsum()
        inforce = grp_df['Inforce (BOM)'].sum()
        life1_alive = grp_df['Life1 inforce'].sum()
        life2_alive = grp_df['Life2 inforce'].sum()
        death = grp_df['Death count'].sum()
        life1_death = grp_df['Life1 death'].sum()
        life2_death = grp_df['Life2 death'].sum()
        norm_death = death / inforce * 1000
        sum_age_inforce = grp_df['Sum age inforce'].sum()
        sum_age_death = grp_df['Sum age death'].sum()
        cumulative_df = pd.DataFrame({'Inforce (BOM)': inforce,
                               'Death count': death,
                               'Norm death': norm_death,
                               'Sum age inforce': sum_age_inforce,
                               'Sum age death': sum_age_death,
                               'Life1 death': life1_death,
                               'Life2 death': life2_death,
                               'Life1 inforce': life1_alive,
                               'Life2 inforce': life2_alive})
        cumulative_df = cumulative_df.reset_index()
        cumulative_df.loc[:,'Cumulative deaths'] = cumulative_df['Death count'].cumsum()
        cumulative_df.loc[:,'Average deaths'] = cumulative_df['Death count']/(math.floor(period/12))
        cumulative_df.loc[:,'Cumulative average deaths'] = cumulative_df['Average deaths'].cumsum()
        cumulative_df.loc[:,'Cumulative average inforce'] = cumulative_df['Inforce (BOM)'].expanding().mean()
        cumulative_df.loc[:,'Cumulative annualized norm death'] = cumulative_df['Cumulative deaths']/cumulative_df['Cumulative average inforce'] * 12 * 1000
        cumulative_df.loc[:,'Cum sum age death']= cumulative_df['Sum age death'].cumsum()
        return cumulative_df
    
    def get_metrics(self, active_df, ref_df, date, period):  
        inforce = active_df['Inforce (BOM)'][(period - 1)]
        avg_age_inforce = (active_df['Sum age inforce'][(period - 1)])/ inforce
        total_death = ref_df['Cumulative deaths'][11]
        avg_age_death = (ref_df['Cum sum age death'][11])/ total_death
        metrics_df = pd.DataFrame({'Date': date,
                                   'Active policy count': inforce,
                                   'Average age of active policies': avg_age_inforce,
                                   'Total death count (ref)': total_death,
                                   'Average age at death (ref)': avg_age_death},
                                  index =[0])
        return metrics_df



if __name__ == '__main__':
##    begin_date = input("Enter begin date: (format: yyyymm, eg: 201701)\n")
##    end_date = input("Enter end date: (format: yyyymm, eg: 201701)\n")
##    begin_year = int(begin_date[:4])
##    begin_month = int(begin_date[4:6])
##    end_year = int(end_date[:4])
##    end_month = int(end_date[4:6])
##    month_to_run = (end_year - begin_year) * 12 + (end_month - begin_month+1)
##    active_period = input("Enter active period in month:\n")
    pass

begin_date = input("Enter begin date: (format: yyyymm, eg: 201701)\n")
end_date = input("Enter end date: (format: yyyymm, eg: 201701)\n")
interim_date = input("Enter interim date [for average age of death distribution]: (format: yyyymm, eg: 201701)\n")
begin_year = int(begin_date[:4])
begin_month = int(begin_date[4:6])
end_year = int(end_date[:4])
end_month = int(end_date[4:6])
interim_year = int(interim_date[:4])
interim_month = int(interim_date[4:6])
month_to_run = (end_year - begin_year) * 12 + (end_month - begin_month+1)
active_period = input("Enter active period in month:\n")
active_period = int(active_period)
url = input("Enter source files url\n")

# Input beginning lower and ending upper (final age) limit [for average age of death distribution]:
begin_age = input("Enter beginning lower age limit, eg: 50 \n")
end_age = input("Enter ending upper age limit, eg: 101 \n")
begin_age = int(begin_age)
end_age = int(end_age)


full_data = ValExtract(url)

by_deal = pd.DataFrame(columns = ['Date', 'Inforce (BOM)', 'Death count', 'Norm death', 'Sum age inforce','Sum age death'])

# For producing DoA file
i = 0
while i < month_to_run:    
    month = min(i,i-12*math.floor(i/12)) + 1
    year = begin_year + math.floor(i/12)
    run_date = pd.Timestamp(year, month, 1)
    snap_df = full_data.take_snapshot(run_date)
    deal_df = full_data.count_all(snap_df, run_date)
    by_deal = by_deal.append(deal_df)
    i = i+1
    
by_deal = by_deal.reset_index()
by_deal.rename(columns = {'index':'Deal Name'}, inplace = True)
by_deal = by_deal.fillna(0)
by_agg = full_data.get_aggregate(by_deal)

writer = pd.ExcelWriter(rf'{full_data.url}\DoA.xlsx',engine = 'xlsxwriter') #update path
by_deal.to_excel(writer, sheet_name = 'By Deal')
by_agg.to_excel(writer, sheet_name = 'Aggregated')
writer.save()

ref_months = month_to_run - active_period
by_period = full_data.make_ref_period(by_agg, ref_months)

ref_df = by_period[by_period.ref_period == 1]
active_df = by_period[by_period.ref_period == 0]

ref_df = full_data.cumulative(ref_df, ref_months)
ref_df2 = ref_df[:active_period]
active_df = full_data.cumulative(active_df, ref_months)

# For producing average age of death distribution

i_date = pd.Timestamp(interim_year, interim_month, 1)
e_date = pd.Timestamp(end_year, end_month,1)
age_slice = []

i = begin_age
while i < end_age:
    age = i
    age_slice.append(age)
    i = i+5

snap_death_df = full_data.take_snapshot_ddistri(i_date, e_date)
snap_death_df['Age_band'] = pd.cut(snap_death_df['Age_at_death'], age_slice)    
grp_df = snap_death_df.groupby('Age_band')
death_counts = grp_df['Age_at_death'].count()
result_df = pd.DataFrame({'Death counts': death_counts})
result_df.reset_index
writer_death = pd.ExcelWriter(rf'{full_data.url}\Death distribution.xlsx',engine = 'xlsxwriter')
result_df.to_excel(writer_death, sheet_name = "Age death distribution")


# Output snap_death_df dataframe (by deal)
bydeal_df = snap_death_df.groupby('Deal Name')
deal_deathcounts = bydeal_df[f'Death by {i_date.strftime("%Y_%m")}'].sum()
result_bydeal_df = pd.DataFrame({'Death counts': deal_deathcounts}).reset_index()
result_bydeal_df.to_excel(writer_death, sheet_name = "Death distribution data")
writer_death.save()


# For metrics
metrics = full_data.get_metrics(active_df, ref_df, e_date, active_period)  #write metrics and save it into an excel file

writer2 = pd.ExcelWriter(rf'{full_data.url}\metrics.xlsx',engine = 'xlsxwriter') #update path
active_df.to_excel(writer2, sheet_name = 'Active period')
ref_df2.to_excel(writer2, sheet_name = 'Reference period')
metrics.to_excel(writer2, sheet_name = 'metrics')
writer2.save()

## For graphs
## 1) Cumulative vs Average Death (done)
#fig, ax = plt.subplots(figsize = (10,8))
#ax.plot(ref_df2.Month,ref_df2[['Cumulative average deaths']], 'r--',
        #active_df.Month, active_df[['Cumulative deaths']], 'b--')
#ax.set_ylim(0,ax.get_ylim()[1])
#ax2 = ax.twinx()
#ax2.plot(ref_df2.Month,ref_df2[['Average deaths']], 'r', 
         #active_df.Month, active_df[['Death count']], 'b')
#ax2.set_ylim(ax.get_ylim())
#ax.set_title('Reported Deaths (2020 vs 3-Yr Average)')
#ax.set_xlabel('Month')
#ax.set_ylabel('Number of deaths')
#ax2.set_ylabel('Number of deaths')
#lines = [Line2D([0],[0], color = 'red' , linestyle = '--'),
         #Line2D([0],[0], color = 'blue' , linestyle = '--'),
         #Line2D([0],[0], color = 'red' , linestyle = '-'),
         #Line2D([0],[0], color = 'blue' , linestyle = '-')]
#labels = ['3-Yr Average (Cumulative)','2020 (Cumulative)','3-Yr Average (Monthly)','2020 (Monthly)']
#plt.legend(lines, labels, bbox_to_anchor = (1.0,-0.07), ncol = 4, fancybox =False, shadow = False)
#plt.tight_layout()
#plt.savefig(rf'{full_data.url}\Cumulative & Average Death Graph.png')  #update path

## 2) Normalized death plot
#fig, ax3 = plt.subplots(figsize = (10,8))
#ax3.plot(ref_df2.Month,ref_df2[['Cumulative annualized norm death']], 'r--', 
         #active_df.Month,active_df[['Cumulative annualized norm death']], 'b--')
#ax3.set_ylim(0,ax3.get_ylim()[1])
##ax4 = ax3.twinx()
##ax4.plot(table_final1['Month'],table_final1[['Cum_norm_death']], 'r--', 
         ##table_final0['Month'],table_final0[['Cum_norm_death']], 'b--')
##ax4.set_ylim(ax3.get_ylim())
#ax3.set_title('Normalized Death Rate (2020 vs 3-Yr Average)')
#ax3.set_xlabel('Month')
#ax3.set_ylabel('Rate per thousand')
##ax4.set_ylabel('Percentage')
##lines2 = [Line2D([0],[0], color = 'red' , linestyle = '--'),
         ##Line2D([0],[0], color = 'blue' , linestyle = '--'),
         ##Line2D([0],[0], color = 'red' , linestyle = '-'),
         ##Line2D([0],[0], color = 'blue' , linestyle = '-')]
#lines2 = [Line2D([0],[0], color = 'red' , linestyle = '--'),
         #Line2D([0],[0], color = 'blue' , linestyle = '--')]
##labels = ['3-Yr Average (Cumulative)','2020 (Cumulative)','3-Yr Average (Monthly)','2020 (Monthly)']
#labels = ['3-Yr Average (Cumulative)','2020 (Cumulative)']
#plt.legend(lines2, labels, bbox_to_anchor = (1.0,-0.07), ncol = 6, fancybox =False, shadow = False)
#plt.tight_layout()
#plt.savefig(rf'{full_data.url}\Normalized Death Graph.png') #update path
