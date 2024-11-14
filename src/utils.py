


def join_pc_map_three_pc(df, df_col,  pc_map  ):
    # merge on any one of three columns in pc_map 
    final_d = [] 
    for col in ['pcd7', 'pcd8', 'pcds']:
        d = df.merge(pc_map , right_on = col, left_on = df_col  )
        final_d.append(d)
    # Concatenate the results
    merged_final = pd.concat(final_d ).drop_duplicates()
    
    if len(df) != len(merged_final):
        print('Warning: some postcodes not matched')
    return merged_final 