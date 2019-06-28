#!/bin/python3
import os
import argparse
import urllib
import urllib.request
import glob
from tqdm import tqdm


# def download(url_list, out_dir):
def download(root, key, verbose):
    url_list = os.path.join('./data/', key + '.txt')
    out_dir = os.path.join(root, key)

    print('{} --> {}'.format(url_list, out_dir))
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    with open(url_list, 'r') as txt_file:
        for url in tqdm(txt_file.readlines()):
            url = url.rstrip('\n')
            filename = os.path.join(out_dir, os.path.basename(url))
            if os.path.exists(filename):
                continue

            if verbose:
                print(url)

            try:
                urllib.request.urlretrieve(url, filename)
            except urllib.error.HTTPError:
                pass
            except urllib.error.URLError:
                pass
            except UnicodeEncodeError:
                pass
            except ConnectionResetError:
                pass

def main():
    allowed_keys = ['egyptian', 'persian', 'siamese', 'tabby', 'tiger']

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/scratch/seyang/kias-cac/cat/')
    parser.add_argument('-k', '--key', choices=allowed_keys)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.key is None:
        for key in allowed_keys:
            download(args.root, key, args.verbose)
    else:
        download(args.root, args.key, args.verbose)


if __name__ == '__main__':
    main()
