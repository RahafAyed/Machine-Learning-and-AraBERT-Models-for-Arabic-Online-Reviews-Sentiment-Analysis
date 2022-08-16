#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('git clone https://github.com/aub-mind/arabert')
get_ipython().system('pip install PyArabic farasapy fast-bert')


# In[ ]:


import pandas as pd
from arabert.preprocess import ArabertPreprocessor
from sklearn.model_selection import train_test_split




df_dataset = pd.read_csv('all.csv',header=0)

DATA_COLUMN = 'text'
LABEL_COLUMN = 'polarity'

df_dataset = df_AJGT[['text', 'polarity']]
df_dataset.columns = [DATA_COLUMN, LABEL_COLUMN]

label_map = {
    'Negative' : 0,
    'Positive' : 1
}



train_dataset, test_dataset = train_test_split(df_AJGT, test_size=0.2,random_state=42)
get_ipython().system('mkdir data')
train_dataset.to_csv("data/train.csv",index=True,columns=train_AJGT.columns,sep=',',header=True)
test_dataset.to_csv("data/dev.csv",index=True,columns=test_AJGT.columns,sep=',',header=True)
with open('data/labels.csv','w') as f:
  f.write("\n".join(map(str, df_AJGT['polarity'].unique())))


# In[ ]:


with open('data/labels.csv','w') as f:
  f.write("\n".join(map(str, df_AJGT['polarity'].unique())))


# In[ ]:


from fast_bert.data_cls import BertDataBunch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabert')

databunch = BertDataBunch('./data/', './data/',
                          tokenizer=tokenizer,
                          train_file='train.csv',
                          val_file='dev.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col='polarity',
                          batch_size_per_gpu=16,
                          max_seq_length=256,
                          multi_gpu=True,
                          multi_label=False,
                          model_type='bert',
                          )


# In[ ]:


import logging
import torch

from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device_cuda = torch.device("cuda")
metrics = [{'name': 'accuracy', 'function': accuracy}]

get_ipython().system("mkdir 'output'")
learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path='aubmindlab/bert-base-arabert',
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir='output',
						finetuned_wgts_path=None,
						warmup_steps=30,
						multi_gpu=False,
						is_fp16=False,
						multi_label=False,
						logging_steps=0)


# In[ ]:


learner.fit(epochs=5,
			lr=2e-5,
			validate=True, 	# Evaluate the model after each epoch
			schedule_type="warmup_linear",
			optimizer_type="adamw")


# In[19]:


# Load the TensorBoard notebook extension
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[20]:


import tensorflow as tf
import datetime


# In[21]:


# Clear any logs from previous runs
get_ipython().system('rm -rf ./logs/ ')


# In[25]:


get_ipython().run_line_magic('tensorboard', '--logdir output/tensorboard')


# In[27]:


learner.validate()

