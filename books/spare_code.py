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


'''
texts = [
    "Penny bought bright blue fishes.",
    "Penny bought bright blue and orange fish.",
    "The cat ate a fish at the store.",
    "Penny went to the store. Penny ate a bug. Penny saw a fish.",
    "It meowed once at the bug, it is still meowing at the bug and the fish",
    "The cat is at the fish store. The cat is orange. The cat is meowing at the fish.",
    "Penny is a fish"
]
'''