import os
import sys
name_label = sys.argv[1]
# path = 'C:/Users/ASUS/Desktop/img_need_rename'
path = os.getcwd()

# name_folder = 'img_need_rename'
print("Chuan bi doi ten file tu dong: ")
i = 1;
for f in os.listdir(path):
    old_name = f
    if(old_name.split('.')[-1] != 'py'):
        new_name = str(name_label) + '_' + str(i) + '.' + old_name.split('.')[1]
        print(new_name)
        os.rename(old_name, new_name)
        i += 1
    else: print("file python: {}".format(old_name))