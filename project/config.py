#!/usr/bin/env python
'''Identify my 4 dogs using CNN based on Keras.


'''

import simplejson as json



def load_config():
    with open("config.json") as f:
        config_raw = json.load(f)

    # Remove comments
    config = dict()
    for k, v in config_raw.items():
        config[k] = config_raw[k]["value"]

    return config
