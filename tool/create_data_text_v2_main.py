import os
import random


k = 1
t = 0
count_0 = 0
count_1 = 0
total_0 = 0
total_1 = 0
LIMIT_LINE = 5
N_LINE = 2500
STATUS = 'sleep'
file_name = f'../text_full_video/full_lie_{STATUS}_at_home/full_lie_{STATUS}_at_home.txt'

print('creating folder:.......................................................')
directory_save_text = '../data_train_text'
os.makedirs(directory_save_text, exist_ok=True)
file_data_train = f'{directory_save_text}/{LIMIT_LINE}_line_lie_{STATUS}_data.txt'
if not os.path.isfile(file_data_train):
    open(file_data_train, 'w').close()


print('prepare folder:.......................................................')
with open(file_name, 'r') as file:
    count_0 = 0
    count_1 = 0
    # lines_processed = 0
    lines = file.readlines()
    for i in range(N_LINE):
        with open(file_data_train, 'a') as file:
            if(k % LIMIT_LINE == 0):
                if(STATUS == 'sleep'):
                    file.write(f'{lines[i].strip()} s\n')
                else: file.write(f'{lines[i].strip()} w\n')
            else:
                lines[i].strip()
                file.write(f'{lines[i].strip()} ')
        line = lines[i].strip()
        count_0 += line.count('0')
        count_1 += line.count('1')
        k += 1
        if k % LIMIT_LINE == 0 or i == len(lines) - 1:
            print(f'lan {t}\ncount 0: {count_0}, \ncount 1: {count_1} \n-------------------------------')
            total_0 += count_0
            total_1 += count_1
            count_0 = 0
            count_1 = 0
            t+=1
            # LIMIT_LINE = 0
            # k += 1
print(f'total 0: {total_0}\ntotal 1: {total_1}')

if N_LINE % LIMIT_LINE != 0:
    # Điều kiện N_LINE % LIMIT_LINE khác 0, nghĩa là cần xóa dòng cuối
    with open(file_data_train, 'r') as file:
        lines = file.readlines()
    with open(file_data_train, 'w') as file:
        file.writelines(lines[:-1])
