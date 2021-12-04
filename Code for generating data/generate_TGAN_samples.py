import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
tf.disable_v2_behavior()
import warnings
warnings.filterwarnings("ignore")

### Dataset Loading
# load train dataset
census_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None)

# load test dataset --> skip first row since it's apparently weird
census_df_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', header=None, skiprows=1) 

census_df.columns = census_df_test.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 
                     'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
columns = census_df.columns

# drop final weight column
census_df.drop(['fnlwgt'], axis=1, inplace=True)
census_df_test.drop(['fnlwgt'], axis=1, inplace=True)


# label encoder (categorical -> numerical)
enc = preprocessing.LabelEncoder()
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

# encode dataset
census_df_enc = MultiColumnLabelEncoder(columns = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income']).fit_transform(census_df)


"""# Tabular GAN"""

# mount my drive (where tgan model is/will be stored)
#from google.colab import drive
#drive.mount('/content/drive')

# need to identify the continuous column indices from census_df 
# age, education_num, capital gain, loss, and hours are continuous 
continuous_columns = [0,3,9,10,11]

# Info for creating TGAN instance (and saving to my gdrive)
from tgan.model import TGANModel
model_path = 'models/tgan/census_fnlwgt_dropped'
drive_path = '/content/drive/My Drive/622/project'
model_path_in_drive = drive_path + '/' + model_path
retrain = True

# if we want to train/retrain, create new model
if retrain:
    # pass explicit params
    tgan = TGANModel(
        continuous_columns,
        output= drive_path + '/models/tgan/tgan_output_fnlwgt_dropped',
        gpu="cuda:0",             # was None
        max_epoch=10,             # 5 gives decent performance, hopefully 10 is better 
        steps_per_epoch=10000,    
        save_checkpoints=True,
        restore_session=True,
        batch_size=200,           
        z_dim=200,
        noise=0.2,
        l2norm=0.00001,
        learning_rate=0.001,
        num_gen_rnn=100,
        num_gen_feature=100,
        num_dis_layers=1,
        num_dis_hidden=100,
        optimizer='AdamOptimizer'
    )

# otherwise load saved model
else:
    tgan = TGANModel.load(model_path_in_drive)

if retrain:
    # reset this for retraining
    tf.reset_default_graph()

    # Fit the model to the census training data -- takes around 40 minutes per epoch!
    # so total about 3h 20 mins!
    tgan.fit(census_df)

    # save the TGAN model
    tgan.save(model_path_in_drive)

# generate new samples from the TGAN model --> intervals of 200 only
num_samples = 32600
tgan_census_df = tgan.sample(num_samples)

# save to csv
save_path = drive_path + '/csvs/tgan_samples.csv'
tgan_census_df.to_csv(save_path)