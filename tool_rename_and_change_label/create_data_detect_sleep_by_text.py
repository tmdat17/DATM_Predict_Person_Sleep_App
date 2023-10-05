import os
LIMIT_LINE = 5
k = 1
t = 0
count_0 = 0
count_1 = 0
total_0 = 0
total_1 = 0
file_name = '../text_full_video/full_lie_wake_at_home/full_lie_wake_at_home.txt'

print('creating folder:.......................................................')
directory_save_text = '../data_train_text'
os.makedirs(directory_save_text, exist_ok=True)
file_data_train = f'{directory_save_text}/lie_wake_data.txt'
if not os.path.isfile(file_data_train):
    open(file_data_train, 'w').close()


print('prepare folder:.......................................................')
with open(file_name, 'r') as file:
    count_0 = 0
    count_1 = 0
    # lines_processed = 0
    lines = file.readlines()
    for i in range(1500):
        with open(file_data_train, 'a') as file:
            if(k % LIMIT_LINE == 0):
                file.write(f'{lines[i].strip()} w\n')
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
