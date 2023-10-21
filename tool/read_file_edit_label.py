import os
import sys
label = sys.argv[1]

path = os.getcwd()

print("Chuan bi doc va sua label file tu dong: ")
i = 1;
for f in os.listdir(path):
    file_name = f
    if(file_name.split('.')[-1] != 'py'):
        with open(file_name, 'r') as file:
            lines = file.readlines()
            for i in range(len(lines)):
                data_text = lines[i]
                data_text = data_text.split(' ')
                data_new = str(label) + ' ' +  data_text[1] + ' ' + data_text[2] + ' ' + data_text[3] + ' ' + data_text[4]
                print("data_new: ", data_new)
                lines[i] = data_new
        with open(file_name, 'w') as file:
            file.writelines(lines)
    else: print("file python: {}".format(file_name))
