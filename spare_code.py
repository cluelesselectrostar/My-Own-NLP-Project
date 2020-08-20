bookNames = [
    'chocofact',
    'fox',
    'matilda'
]

#define dataframe
entity_dtm = pd.DataFrame(entityCounter)
entity_dtm["bookNames"] = bookNames
entity_dtm = entity_dtm.set_index("bookNames")

# add entry for sum, and filter sum>5
entity_dtm = entity_dtm.transpose()
entity_dtm['sum'] = entity_dtm.sum(axis = 1, skipna = True) 
entity_dtm = entity_dtm[entity_dtm['sum'] > 5]
entity_dtm = entity_dtm.sort_values(by='sum', inplace=True, ascending =False)
entity_dtm = entity_dtm.transpose()

# filter
entity_dtm