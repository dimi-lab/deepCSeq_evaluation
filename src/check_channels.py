## Usage:  python check_channels.py file.ome.tiff
import sys
import tifffile as tf
import pandas as pd
pd.set_option('display.max_rows', None)

import xml.etree.ElementTree as ET
inFile = sys.argv[1]
t = tf.TiffFile(inFile)
metaDataList = []
metadata = ET.fromstring(t.pages[0].description)
for ele in metadata.findall('.//*'):
    chMeta = dict(ele.attrib)
    #if 'Fluor' in chMeta:
    if 'Name' in chMeta:
        metaDataList.append(ele.attrib)

print(pd.DataFrame(metaDataList))
