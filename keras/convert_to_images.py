import pandas as pd
import numpy as np
from PIL import Image


df=pd.read_csv('dataset/fer2013.csv')


i=-1
for pixels in df.iloc[1:,1]:
	#print(pixels)
	i+=1	
	#if i < 28706 :
	#	continue
	img_str=pixels.split(' ')
	img_data=np.asarray(img_str, dtype=np.uint8).reshape(48,48)
	img=Image.fromarray(img_data)
	#print(df['Usage'][i])
	

	img.save('dataset/' + df["Usage"][i] + '/' + str(df["emotion"][i]) + '/img_' + str(i) + '.jpg')
	print('dataset/' + df["Usage"][i] + '/' + str(df["emotion"][i]) + '/img_' + str(i) + '.jpg')

