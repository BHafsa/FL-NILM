
fed_NILM = {
    'power': {'mains': ['active'], 'appliance': ['active']},
    'sample_rate': 10,
    'appliances': [],
    'artificial_aggregate': False,
    'DROP_ALL_NANS': True,
    'methods': {

    },
    'train': {
        'datasets': {
            'c_1': {
                'path': None,
                'buildings': {
                    12: {
                        'start_time': '2015-01-01',
                        'end_time': '2015-01-15'
                    }
                }
            },
            'client_2': {
                'path': None,
                'buildings': {
                    2: {
                        'start_time': '2015-01-01',
                        'end_time': '2015-01-15'
                    }
                }
            },
            'client_3': {
                'path': None,
                'buildings': {
                    3: {
                        'start_time': '2015-01-01',
                        'end_time': '2015-01-15'
                    }
                }
            },
            'client_4': {
                'path': None,
                'buildings': {
                    4: {
                        'start_time': '2015-01-01',
                        'end_time': '2015-01-15'
                    }
                }
            },
            'client_5': {
                'path': None,
                'buildings': {
                    5: {
                        'start_time': '2015-01-01',
                        'end_time': '2015-01-15'
                    }
                }
            },
            'client_6': {
                'path': None,
                'buildings': {
                    6: {
                        'start_time': '2015-01-01',
                        'end_time': '2015-01-15'
                    }
                }
            },
            'client_7': {
                'path': None,
                'buildings': {
                    7: {
                        'start_time': '2015-01-01',
                        'end_time': '2015-01-15'
                    }
                }
            },

            'client_9': {
                'path': None,
                'buildings': {
                    9: {
                        'start_time': '2015-01-01',
                        'end_time': '2015-01-15'
                    }
                }
            },

        }
    },
    'test': {
        'datasets': {
            'cl10': {
                'path': None,
                'buildings': {
                    10: {
                        'start_time': '2015-01-15',
                        'end_time': '2015-01-30'
                    },
                }
            },
            'cl11': {
                'path': None,
                'buildings': {
                    11: {
                        'start_time': '2015-01-15',
                        'end_time': '2015-01-30'
                    },
                }
            },

        },
        'metrics': ['mae', 'nde', 'rmse', 'f1score'],
    }
}