import re
import pandas as pd
import os
import glob
xml_list=[]
filepath='C:\tensorflow2\models\research\object_detection\images'
for folder in ['test', 'train']:
    for filepath in glob.iglob('images/'+folder+'/*.txt'):
        base=os.path.basename(filepath)
        base1=os.path.splitext(base)[0]
        #print(base1)
        f=open(filepath, "r")
        x=f.readline()
        y=int(x)
        for i in range(y):
            z=f.readline()
            wordList = re.sub("[^\w]", " ", z).split()
            #print(wordList)
            value= ( base1 + ".jpg" , "1920", "1080", wordList[4],wordList[0],wordList[1],wordList[2],wordList[3])
            xml_list.append(value)
            #print(xml_list)
        f.close()
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv(('images/'+folder+'_labels.csv'), index=None)
    print('Successfully converted txt to csv.')
