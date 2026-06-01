"""
Standard UK Sector Subject Area (SSA) codes with full labels and short tile labels.
Codes 1-15 are the OfS/JNF standard. Code 99 is a local extension.
"""

SSA = {
    1:  {'label': 'Health, Public Services and Care',               'tile': 'Health'},
    2:  {'label': 'Science and Mathematics',                        'tile': 'Science'},
    3:  {'label': 'Agriculture, Horticulture and Animal Care',      'tile': 'Agriculture'},
    4:  {'label': 'Engineering and Manufacturing Technologies',      'tile': 'Engineering'},
    5:  {'label': 'Construction, Planning and the Built Environment','tile': 'Construction'},
    6:  {'label': 'Information and Communication Technology',        'tile': 'Digital & Tech'},
    7:  {'label': 'Retail and Commercial Enterprise',               'tile': 'Business'},
    8:  {'label': 'Leisure, Travel and Tourism',                    'tile': 'Leisure'},
    9:  {'label': 'Arts, Media and Publishing',                     'tile': 'Arts & Media'},
    10: {'label': 'History, Philosophy and Theology',               'tile': 'Humanities'},
    11: {'label': 'Social Sciences',                                'tile': 'Social Sciences'},
    12: {'label': 'Languages, Literature and Culture',              'tile': 'Languages'},
    13: {'label': 'Education and Training',                         'tile': 'Education'},
    14: {'label': 'Preparation for Life and Work',                  'tile': 'Foundation'},
    15: {'label': 'Business, Administration and Law',               'tile': 'Business & Law'},
    99: {'label': 'Sustainability',                                  'tile': 'Sustainability'},
}
