from os import listdir
from os.path import isfile, join, splitext
from get_map import get_map

# just for TESTING purposes, NOT USED in final version
test_path = '/home/zsolti/deeplearn/dev-dataset/dev-dataset-forged'


def get_maps(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    for file in onlyfiles:
        base_filename_split = splitext(file)
        if base_filename_split[1] == '.jpg' or base_filename_split[1] == '.tif':
            filepath = join(path, file)
            get_map(filepath)


if __name__ == '__main__':
    # import sys
    #
    # path = sys.argv[1]
    path = test_path  # TESTING PURPOSES
    get_maps(path)
