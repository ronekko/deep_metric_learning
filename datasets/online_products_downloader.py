# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 20:23:23 2017

@author: sakurai
"""

import os
import contextlib
import time
from six.moves.urllib import request
from progressbar import ProgressBar, Percentage, Bar, ETA, FileTransferSpeed
import fuel

fuel_root_path = fuel.config.config["data_path"]["yaml"]
base_url = "ftp://cs.stanford.edu/cs/cvgl/"
filenames = ["Stanford_Online_Products.zip"]
urls = [base_url + f for f in filenames]

fuel_data_path = os.path.join(fuel_root_path, "online_products")
os.mkdir(fuel_data_path)

for filename in filenames:
    url = base_url + filename
    filepath = os.path.join(fuel_data_path, filename)

    with contextlib.closing(request.urlopen(url)) as f:
        expected_filesize = int(f.headers["content-length"])
        print(expected_filesize)
    time.sleep(5)

    widgets = ['{}: '.format(filename), Percentage(), ' ', Bar(), ' ', ETA(),
               ' ', FileTransferSpeed()]
    progress_bar = ProgressBar(widgets=widgets,
                               maxval=expected_filesize).start()

    def reporthook(count, blockSize, totalSize):
        progress_bar.update(min(count*blockSize, totalSize))

    request.urlretrieve(url, filepath, reporthook=reporthook)
    progress_bar.finish()

    downloaded_filesize = os.path.getsize(filepath)
    assert expected_filesize == downloaded_filesize, " ".join((
        "expected file size is {}, but the actual size of the downloaded file",
        "is {}.")).format(expected_filesize, downloaded_filesize)
