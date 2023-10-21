import os
LIMIT_LINE = 7
name_folder = f'{LIMIT_LINE}_line'
file_1 = f'../data_train_text/{name_folder}/{LIMIT_LINE}_line_lie_sleep_data.txt'
file_2 = f'../data_train_text/{name_folder}/{LIMIT_LINE}_line_lie_wake_data.txt'
file_name = f'../data_train_text/{name_folder}/{LIMIT_LINE}_line_merge_2_lie_to_train.txt'
if not os.path.isfile(file_name):
    open(file_name, 'w').close()
    

with open(file_1, 'r') as file:
        lines = file.readlines()
with open(file_name, 'a') as file:
    file.writelines(lines[:])
    
with open(file_2, 'r') as file:
    lines = file.readlines()
with open(file_name, 'a') as file:
    file.writelines(lines[:])