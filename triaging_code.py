
def get_true_risk(row):

    ### This is how we define High (4), Medium (3), and Low (2) risk for triage purposes
    if (row['active_rescue'] > 0 or row['ir_flag'] > 0):
        return 4 # high risk
    
    elif (
          ('suicidal_desire' in row['suicidality'] and 'suicidal_intent' in row['suicidality'] and 'suicidal_capability' in row['suicidality']) \
            or ('timeframe' in row['suicidality'])) \
            and (row['3rd_party'] !=1 and row['testing'] != 1 and row['prank'] != 1 and row['other'] != 1): 
        return 4 # high risk
    
    elif (row['suicide'] == 1 or row['self_harm'] == 1) \
        and (row['3rd_party'] !=1 and row['testing'] != 1 and row['prank'] != 1 and row['other'] != 1): 
        return 3 # medium risk
    
    # this was not available in the data dump i received     
    # elif (row['blank'] == 1): # CC survey not complete, ground truth unknown
    #     return 0
    
    else:
        return 2 # normal risk


