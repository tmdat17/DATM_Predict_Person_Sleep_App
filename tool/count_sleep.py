import statistics
LIMIT_LINE = 3
k = 1
t = 0
count_0 = 0
count_1 = 0
total_0 = 0
total_1 = 0
countFinal = 0
countCheck = 0
arr0 = []
file_name = '../text_full_video/full_lie_wake_at_home/full_lie_wake_at_home.txt'
# file_name = '../text_full_video/full_lie_wake_at_home/full_lie_wake_at_home.txt'
print("Chuan bi dem:  ")
with open(file_name, 'r') as file:
    count_0 = 0
    count_1 = 0
    lines_processed = 0
    lines = file.readlines()
    for i in range(len(lines)):
        line = lines[i].strip()
        count_0 += line.count('0')
        count_1 += line.count('1')
        k += 1
        if k % LIMIT_LINE == 0 or i == len(lines) - 1:
            print(f'lan {t}\ncount 0: {count_0}, \ncount 1: {count_1} \n-------------------------------')
            if(count_0 >= 7 and count_0 + count_1 == 12):
                countFinal+=1
            if(count_0 + count_1 == 12):
                arr0.append(count_0)
                countCheck+=1
            total_0 += count_0
            total_1 += count_1
            count_0 = 0
            count_1 = 0
            t+=1
            # LIMIT_LINE = 0
            # k += 1
print(f'total 0: {total_0}\ntotal 1: {total_1}')
print(f'countFinal: {countFinal}\ncountCheck 1: {countCheck}   --- ' , countFinal/countCheck*100 )
# print(arr0)
print(statistics.mean(arr0))
print(statistics.median(arr0))
print(min(arr0))
print(max(arr0))