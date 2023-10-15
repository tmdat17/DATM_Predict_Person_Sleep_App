import joblib
import numpy as np
import pandas as pd

model_detect_sleep = joblib.load('./model_detect_sleep/SVC/SVC_KFold_K_50_C_100000_GAMMA_0.001.h5')

isSleep = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
if(len(isSleep) == 20):
            # input_model_sleep = ' '.join(map(number, isSleep))
            # print('input:  ', input_model_sleep)
            test_df = pd.DataFrame([isSleep])
            preds = model_detect_sleep.predict(test_df)
            if(preds[0] == 's'):
                print('preds:  SLEEP', )
            else: 
                print('preds:  WAKE', )
            isSleep = []