# -*- coding: utf-8 -*-
"""
Script to Parse an INI File and Generate Input File for cbns3d_cuda

This script reads an input `.ini` configuration file and parses its content to 
generate a corresponding input file required by the `cbns3d_cuda` simulation 
program. 

### Example Command:
```bash
python parser_ini.py rae2822.ini
"""

import configparser
import sys

if __name__ == "__main__":
    
    config = configparser.ConfigParser()
    config.optionxform = str

    f_name = sys.argv[1]
    config.read(f_name)

    for sec in config.sections():
        for key in config[sec]:
            print(f"{key} = {config[sec][key]}")

    with open('input.txt', 'w') as f:
        for sec in config.sections():
            for key in config[sec]:
                f.write(config[sec][key])
                f.write('\n')
