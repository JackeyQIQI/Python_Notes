# delete column in a dataframe
del df["aa"]
del df["bb"]

# delete a varname form a list
var_list = list(df_sample)
var_list.remove('ID')
var_list.remove('Month')
