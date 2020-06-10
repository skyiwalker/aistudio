import ast

import sys

import types

import json

import numpy as np

import pandas as pd

from datetime import datetime

 

with open(sys.argv[1]) as f:

    p = ast.parse(f.read())

 

for node in p.body[:]:

    if type(node) not in [ ast.FunctionDef, ast.Import, ast.ImportFrom, ast.ClassDef ] :

        p.body.remove(node)

 

module = types.ModuleType("mod")

code = compile(p, "mod.py", 'exec')

sys.modules["mod"] = module

exec(code, module.__dict__)

# Data Loading

try:

    all_data = pd.read_csv("/curate/input/feature.csv")

except:

    all_data = pd.read_csv("/curate/input/feature.csv", encoding = "ISO-8859-1")

try:

    test_data = pd.read_csv("/curate/input/test_feature.csv")

except:

    test_data = pd.read_csv("/curate/input/test_feature.csv", encoding = "ISO-8859-1")

 

 

has_remove_abnormal = 0

has_pre_feature = 0

has_post_target = 0

 

for node in p.body[:]:

    if type(node) in [ ast.FunctionDef ] :

        if node.name == "_remove_abnormal" :

            has_remove_abnormal = 1

        if node.name == "_pre_feature" :

            has_pre_feature = 1

        if node.name == "_post_feature" :

            has_post_target = 1

 

if has_pre_feature == 0 :

    print ("there is no _pre_feature function defined")

    exit(1)

import mod

from mod import *

 

try :

    if has_remove_abnormal == 1 :

        all_data = mod._remove_abnormal( all_data )

except Exception as ex:

    print ("there are errors in executing _remove_abnormal()", ex)

    exit(1)

 

 

try :

    test_x = mod._pre_feature( all_data, test_data)

except Exception as ex:

    print ("there are errors in executing _pre_feature()", ex)

    exit(1)

 

from sklearn.externals import joblib

try :

    clf = joblib.load('/curate/input/model.pkl')

    predict = clf.predict(test_x)

except Exception as ex:

    print ("there are errors in loading ML model", ex)

    exit(1)

 

try :

    if has_post_target == 1 :

        predict = mod._post_target( all_data, predict )

except Exception as ex:

    print ("there are errors in executing _post_target()", ex)

    exit(1)

 

result = {}

result['result'] = predict.tolist()

 

try :

    with open('/curate/output/metadata/dm.json', 'w') as outfile:

        json.dump(result, outfile)

except Exception as ex:

    print ("there are errors in writing result", ex)

    exit(1)
