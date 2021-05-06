#!/usr/bin/env python
# coding: utf-8

# In[2]:

get_ipython().run_line_magic('matplotlib', 'inline')
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


# In[2]:


dataDir='..'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)


# In[3]:


# initialize COCO api for instance annotations
coco=COCO(annFile)


# In[4]:


# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())

print("--------------------------------------------")
print("--------------------------------------------")
nms = set([cat['supercategory'] for cat in cats])
print('COCO categories: \n{}\n'.format(' '.join(nms)))
nms=[cat['name'] for cat in cats]
print('COCO supercategories: \n{}'.format(' '.join(nms)))
print("--------------------------------------------")
print("--------------------------------------------")


# In[5]:


# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
imgIds = coco.getImgIds(catIds=catIds );
#imgIds = coco.getImgIds(imgIds = [5])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
print(img)

catIds = coco.getCatIds(catNms=['cat']);
imgIds = coco.getImgIds(catIds=catIds );
print("cat images are: {}".format(imgIds))


# In[6]:


# load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()


# In[7]:


# load and display instance annotations
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)


# In[8]:


# initialize COCO api for person keypoints annotations
annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
coco_kps=COCO(annFile)


# In[9]:


# load and display keypoints annotations
plt.imshow(I); plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)
coco_kps.showAnns(anns)


# In[10]:


# initialize COCO api for caption annotations
annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps=COCO(annFile)


# In[11]:


# load and display caption annotations
annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
plt.imshow(I); plt.axis('off'); plt.show()

