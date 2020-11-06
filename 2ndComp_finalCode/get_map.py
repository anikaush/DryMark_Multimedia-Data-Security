import os

# just for TESTING purposes, NOT USED in final version
test_filepath = '/home/zsolti/deeplearn/dev-dataset/dev-dataset-forged/dev_0004.tif'


def get_map(filepath):
    olddir = os.getcwd()
    os.chdir("./SUPPORT/PRNU")

    command = 'venv/bin/python final.py ' + filepath + '> /dev/null 2>&1'

    os.system(command)
    os.chdir(olddir)


if __name__ == '__main__':
    import sys

    path = sys.argv[1]
    # path = test_filepath # TESTING PURPOSES
    get_map(path)
