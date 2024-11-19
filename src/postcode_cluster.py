import pandas as pd 
import logging

cols =[
      'all_types_total_buildings',
      'all_res_heated_vol_h_total',
      'perc_large_houses',
      'perc_standard_houses',
      'perc_small_terraces',
      #  'perc_estates',
      'perc_all_flats',
      'perc_age_Pre-1919',
      'perc_age_1919-1999',
      'perc_age_Post-1999',
      'Average Household Size',
      'HDD',
      'HDD_winter',
      'CDD',
      'postcode_density',
      'perc_white',
      'perc_asian',
      'Perc_econ_employed',
      'perc_econ_inactive', 
      ]


def load_and_prepare_postcode_data( data,  subset=None):
    logging.info('Load postcode data and aggregate variables')
    # data = pd.read_csv(input_path)
    data = pre_process_pc(data)
    X = data[cols].copy() 
    data_cols = X.columns.tolist()
    if subset is None:
        return  X, data_cols
    else:
        return  X.iloc[0:subset], data_cols
    


def econ_settings():
   econ_act = ['economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Part-time',
 'economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Full-time',
 'economic_activity_perc_Economically active (excluding full-time students): In employment: Self-employed with employees: Part-time',
 'economic_activity_perc_Economically active (excluding full-time students): In employment: Self-employed with employees: Full-time',
 'economic_activity_perc_Economically active (excluding full-time students): In employment: Self-employed without employees: Part-time',
 'economic_activity_perc_Economically active (excluding full-time students): In employment: Self-employed without employees: Full-time',
 
 'economic_activity_perc_Economically active and a full-time student: In employment: Employee: Part-time',
 'economic_activity_perc_Economically active and a full-time student: In employment: Employee: Full-time',
 'economic_activity_perc_Economically active and a full-time student: In employment: Self-employed with employees: Part-time',
 'economic_activity_perc_Economically active and a full-time student: In employment: Self-employed with employees: Full-time',
 'economic_activity_perc_Economically active and a full-time student: In employment: Self-employed without employees: Part-time',
 'economic_activity_perc_Economically active and a full-time student: In employment: Self-employed without employees: Full-time',
   ]
   econ_inac = ['economic_activity_perc_Economically inactive: Retired',
 'economic_activity_perc_Economically inactive: Student',
 'economic_activity_perc_Economically inactive: Looking after home or family',
 'economic_activity_perc_Economically inactive: Long-term sick or disabled',
 'economic_activity_perc_Economically inactive: Other' ] 
   
   unemp = ['economic_activity_perc_Economically active and a full-time student: Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks',
   'economic_activity_perc_Economically active (excluding full-time students): Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks' 
   ]
   other = ['economic_activity_perc_Does not apply']

   list_cols = [econ_act, unemp, econ_inac, other] 
   names = ['Perc_econ_employed', 'perc_econ_unemployed', 'perc_econ_inactive', 'perc_econ_other' ] 
   return list_cols, names 


def eth_setting():
    white = [ 'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British', 
    'ethnic_group_perc_White: Irish', 
    'ethnic_group_perc_White: Gypsy or Irish Traveller',
    'ethnic_group_perc_White: Roma', 'ethnic_group_perc_White: Other White',]
    black = [ 'ethnic_group_perc_Black, Black British, Black Welsh, Caribbean or African: African',
 'ethnic_group_perc_Black, Black British, Black Welsh, Caribbean or African: Caribbean',
 'ethnic_group_perc_Black, Black British, Black Welsh, Caribbean or African: Other Black',] 
    asian = [ 'ethnic_group_perc_Asian, Asian British or Asian Welsh: Bangladeshi',
 'ethnic_group_perc_Asian, Asian British or Asian Welsh: Chinese',
 'ethnic_group_perc_Asian, Asian British or Asian Welsh: Indian',
 'ethnic_group_perc_Asian, Asian British or Asian Welsh: Pakistani',
 'ethnic_group_perc_Asian, Asian British or Asian Welsh: Other Asian',] 
    other = ['ethnic_group_perc_Does not apply',
 'ethnic_group_perc_Other ethnic group: Arab',
 'ethnic_group_perc_Other ethnic group: Any other ethnic group',] 
    mixed = [ 'ethnic_group_perc_Mixed or Multiple ethnic groups: White and Asian',
 'ethnic_group_perc_Mixed or Multiple ethnic groups: White and Black African',
 'ethnic_group_perc_Mixed or Multiple ethnic groups: White and Black Caribbean',
 'ethnic_group_perc_Mixed or Multiple ethnic groups: Other Mixed or Multiple ethnic groups',]
    list_cols = [white, black, asian, other, mixed]
    names = ['perc_white', 'perc_black', 'perc_asian', 'perc_asian', 'perc_ethnic_other', 'perc_mixed']
    return list_cols , names  


def type_setting1():
    large = ['Very large detached', 'Large detached','Large semi detached' ,  'Tall terraces 3-4 storeys']
    standard = ['Standard size semi detached',  'Standard size detached']
    large_flats = ['Very tall point block flats', 'Tall flats 6-15 storeys']
    med_flats = ['Medium height flats 5-6 storeys', '3-4 storey and smaller flats']
    small_terraces = ['Small low terraces', '2 storeys terraces with t rear extension', 'Semi type house in multiples']
    flats = large_flats + med_flats
    estates = ['Linked and step linked premises', 'Planned balanced mixed estates']
    unkn_typ =  ['all_unknown']
    outbuilds = ['Domestic outbuilding']
    list_cols = [large, standard,  small_terraces, estates,  flats, unkn_typ, outbuilds ]
    names = ['perc_large_houses', 'perc_standard_houses',  'perc_small_terraces', 'perc_estates', 'perc_all_flats', 'perc_unknown_typ', 'perc_outbuildings']
    return list_cols, names

def age_setting1():
    pre_1919 = ['Pre 1919']
    o1919_1999= ['1919-1944', '1945-1959', '1960-1979', '1980-1989', '1990-1999',]
    post_1999= ['Post 1999'] 
    unk = ['Unknown_age']
    age_cols = [pre_1919, o1919_1999, post_1999, unk]
    age_names = ['perc_age_Pre-1919', 'perc_age_1919-1999', 'perc_age_Post-1999', 'perc_age_Unknown']
    return  age_cols, age_names 


def pre_process_pc(data):
    type_cols, type_names = type_setting1() 
    age_cols, age_names = age_setting1() 
    eth_cols, eth_name = eth_setting() 
    econ_cols , econ_names = econ_settings()

    for i in range (len(type_cols)):
        typ = [x +'_pct' for x in type_cols[i] ]
        data[type_names[i]] = data[typ ].fillna(0).sum(axis=1)

    for i in range (len(age_cols)):
        age = [x +'_pct' for x in age_cols[i] ]
        data[age_names[i]] = data[age ].fillna(0).sum(axis=1)


    for i in range (len(eth_cols)):
        data[eth_name[i]] = data[eth_cols[i] ].fillna(0).sum(axis=1)


    for i in range (len(econ_cols)):
        data[econ_names[i]] = data[econ_cols[i] ].fillna(0).sum(axis=1)

    return data 
