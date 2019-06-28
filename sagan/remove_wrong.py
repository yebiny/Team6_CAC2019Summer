from PIL import Image
import glob
import os

def check_file(path):
    with open(path, 'rb') as image_file:
        try:
            Image.open(image_file)
            return True
        except IOError:
            return False


def main():
    for each in glob.glob('/scratch/seyang/cat/*/*'):
        if not check_file(each):
            print(each)
            os.remove(each)

if __name__ == '__main__':
    main()
