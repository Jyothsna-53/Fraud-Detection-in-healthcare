#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install imbalanced-learn')

get_ipython().system('pip install pickle5')


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder , OrdinalEncoder , LabelEncoder
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight


# In[3]:


Train=pd.read_csv("C:/Users/rapak/Downloads/archive (2)/Train-1542865627584.csv")
Train_Beneficiarydata=pd.read_csv("C:/Users/rapak/Downloads/archive (2)/Train_Beneficiarydata-1542865627584.csv")
Train_Inpatientdata=pd.read_csv(r"C:\Users\rapak\Downloads\archive (2)\Train_Inpatientdata-1542865627584.csv")
Train_Outpatientdata=pd.read_csv(r"C:\Users\rapak\Downloads\archive (2)\Train_Outpatientdata-1542865627584.csv")


# In[4]:


Train_Allpatientdata=pd.merge(Train_Outpatientdata,Train_Inpatientdata,
                              left_on=['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
       'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
       'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
       'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
       'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
       'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
       'ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid',
       'ClmAdmitDiagnosisCode'],
                              right_on=['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
       'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
       'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
       'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
       'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
       'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
       'ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid',
       'ClmAdmitDiagnosisCode']
                              ,how='outer')
Train_Allpatientdata.to_csv('Train_Allpatientdata.csv', index=False)


# In[5]:


Train_Allpatientdata=pd.merge(Train_Allpatientdata,Train_Beneficiarydata,on="BeneID")


# In[6]:


Train_Allpatientdata=pd.merge(Train_Allpatientdata,Train,on="Provider")


# In[7]:


Train_Allpatientdata.shape


# In[8]:


Train_Allpatientdata.head(10)


# In[9]:


Train_Allpatientdata.info()


# In[58]:


a=(Train_Allpatientdata["AttendingPhysician"]==Train_Allpatientdata["OperatingPhysician"])
b=(Train_Allpatientdata["OperatingPhysician"]==Train_Allpatientdata["OtherPhysician"])
c=(Train_Allpatientdata["AttendingPhysician"]==Train_Allpatientdata["OtherPhysician"])

print(a.sum())
print(b.sum())
print(c.sum())
print( (a+b).sum() ) # atten=oper=other


# In[59]:


def physician_same(row):
    atten_oper=row["AttendingPhysician"]==row["OperatingPhysician"]
    oper_other=row["OperatingPhysician"]==row["OtherPhysician"]
    atten_other=row["AttendingPhysician"]==row["OtherPhysician"]
    if atten_oper==True and oper_other==True:# atten = oper = other
        return 0
    elif atten_oper==True and oper_other==False:# atten = oper != other
        return 1
    elif atten_oper==False and oper_other==True:# atten != oper = other
        return 2
    else:# atten != oper != other
        return 3
    
phy_same=Train_Allpatientdata.apply(physician_same,axis=1)


# In[62]:


Train_Allpatientdata["phy_same"]=phy_same


# In[63]:


def physician_count(row,list_count):
    count=0
    for col in list_count:
        if pd.isnull(row[col]):
            continue
        else:
            count+=1
    return count
    
list_count=["AttendingPhysician","OperatingPhysician","OtherPhysician"]
phy_count=Train_Allpatientdata.apply(physician_count,axis=1,args=(list_count,))


# In[64]:


Train_Allpatientdata["phy_count"]=phy_count


# In[65]:


Train_Allpatientdata.head(4)


# In[66]:


startdate= pd.to_datetime( Train_Allpatientdata["ClaimStartDt"] )
enddate= pd.to_datetime( Train_Allpatientdata["ClaimEndDt"] )

period = ( enddate - startdate).dt.days
Train_Allpatientdata["period"] = period


# In[67]:


Train_Allpatientdata.info()


# In[68]:


## taking copy
copy1 = Train_Allpatientdata.copy()
copy1.to_csv('joined_dataset.csv', index=False)

# Add the new code for saving CSV and splitting the dataset here:

# Save the preprocessed dataset to a CSV file
copy1.to_csv('preprocessed_dataset.csv', index=False)

# Split the dataset into training and testing sets
train_ratio = 0.8  # You can adjust the ratio based on your preference
test_ratio = 1 - train_ratio

# Shuffle the data
copy1_shuffled = shuffle(copy1)

# Calculate the number of samples for training and testing
num_samples = len(copy1_shuffled)
num_train_samples = int(train_ratio * num_samples)

# Split the dataset
train_data = copy1_shuffled[:num_train_samples]
test_data = copy1_shuffled[num_train_samples:]

# Save the training and testing sets to separate CSV files
train_data.to_csv('train_dataset.csv', index=False)
test_data.to_csv('test_dataset.csv', index=False)


# In[69]:


cronic_cols_names=copy1.columns[ copy1.columns.str.startswith("ChronicCond") ]
cronic_cols=copy1[   cronic_cols_names   ]
cronic=cronic_cols.replace({2:0})
copy1[   cronic_cols_names   ]=cronic


# In[70]:


copy1["PotentialFraud"]=copy1["PotentialFraud"].replace({"Yes":1,"No":0})


# In[71]:


copy1["Gender"]=copy1["Gender"].replace({2:0})


# In[72]:


startadmt= pd.to_datetime( copy1["AdmissionDt"] )
enddatadmt= pd.to_datetime( copy1["DischargeDt"] )

periodadmt = ( enddatadmt - startadmt).dt.days
copy1["periodadmt"] = periodadmt
copy1["periodadmt"]=copy1["periodadmt"].fillna(0)


# In[73]:


copy1["RenalDiseaseIndicator"]=copy1["RenalDiseaseIndicator"].replace({"Y":1})


# In[74]:


birthdate=pd.to_datetime(copy1["DOB"])
enddate=pd.to_datetime(copy1["DOD"])

# cheack whether the patient dead or alife
def alife_function(value):
    if value==True:
        return 1
    else:
        return 0
alife = pd.isna(enddate).apply(alife_function)


# get the age of patient
max_date=enddate.dropna().max()
enddate[pd.isna(enddate)]=max_date
period=(((enddate-birthdate).dt.days/356).astype(int))

copy1["age"]=period
copy1["alife"]=alife


# In[75]:


copy1.head(2)


# In[76]:


copy1.info()


# In[77]:


#means and stds
def groupby(df,by,vars_to_group,methods,col_ident,as_index=True,agg=False):
    if agg:
        grouped=df.groupby(by=by,as_index=as_index)[vars_to_group].agg(methods)
        cols=['_'.join(col) for col in grouped.columns.values]
        cols=[col_ident+"_"+col for col in cols]
        grouped.columns=cols
        return grouped
    
    else:
        concat=df.groupby(by=by,as_index=as_index)[vars_to_group].transform(methods[0])
        cols=[ col_ident+"_"+col+"_"+methods[0] for col in concat.columns ]
        concat.columns=cols
        
        for method in methods[1:]:
            grouped=df.groupby(by=by,as_index=as_index)[vars_to_group].transform(method)
            cols=[col_ident+"_"+col+"_"+method for col in grouped.columns]
            grouped.columns=cols
            concat=pd.concat([concat,grouped],axis=1)
        
        return concat


# In[78]:


money_cols=["InscClaimAmtReimbursed","DeductibleAmtPaid","NoOfMonths_PartACov","NoOfMonths_PartBCov",
           "IPAnnualReimbursementAmt","IPAnnualDeductibleAmt","OPAnnualReimbursementAmt","OPAnnualDeductibleAmt"]


# In[79]:


provider_money=groupby(copy1,["Provider"],money_cols,["mean","std"],"provider",
                       True,False)


# In[80]:


banel_money=groupby(copy1,["BeneID"],money_cols,["mean","std"],"banel",
                       True,False)


# In[81]:


diag1_money=groupby(copy1,["ClmDiagnosisCode_1"],money_cols,["mean","std"],"diag1",
                       True,False)


# In[82]:


selected_cols_names=["phy_same","phy_count","period","periodadmt","age","alife","Provider","PotentialFraud"]
selected_cols=copy1[selected_cols_names]


# In[83]:


data=pd.concat([selected_cols,provider_money,banel_money,diag1_money],axis=1)


# In[84]:


grouped=data.groupby(by=["Provider","PotentialFraud"]).agg("mean").reset_index()


# In[85]:


grouped


# In[86]:


grouped=grouped.fillna(0)


# In[87]:


features=grouped.iloc[:,2:]
labels=grouped.iloc[:,1]


# In[88]:


from imblearn.over_sampling import SMOTE
oversample = SMOTE()
features, labels = oversample.fit_resample(features, labels)


# In[89]:


scaler = StandardScaler()
featuresstand=scaler.fit_transform(features)


# In[90]:


ff=compute_class_weight(class_weight="balanced",classes=np.unique(labels),y=labels)
cw=dict(zip(np.unique(labels),ff))

featuress,labelss=shuffle(featuresstand,labels)
xtrain,xtest,ytrain,ytest = train_test_split(featuress,labelss,test_size=0.1)


# In[91]:


xtrain=xtrain.astype(np.float32)
xtest=xtest.astype(np.float32)
ytrain=ytrain.astype(np.float32).to_numpy()
ytest=ytest.astype(np.float32).to_numpy()


ytrain=ytrain.reshape(ytrain.shape+(1,))
ytest=ytest.reshape(ytest.shape+(1,))

print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)


# In[92]:


inpt = tf.keras.layers.Input(shape=(xtrain.shape[1],))

d1=tf.keras.layers.Dense(246, activation='relu')(inpt)
d1=tf.keras.layers.Dense(123, activation='relu')(d1)

d2=tf.keras.layers.Dense(1,activation="sigmoid")(d1)

nural_network=tf.keras.Model(inputs=inpt,outputs=d2)


# In[93]:


nural_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='binary_crossentropy'
                    ,metrics=["accuracy"])

history_combined=nural_network.fit(xtrain,ytrain,validation_data=(xtest,ytest),batch_size=128,epochs=300,
                                  class_weight=cw)
with open("model_architecture.json", "w") as json_file:
    json_file.write(nural_network.to_json())

# Save the model weights to an HDF5 file
nural_network.save_weights("model.weights.h5")


# In[94]:


ytt=nural_network.predict(xtest)
dd=np.round(ytt)
print(classification_report(ytest,dd))
with open("model_architecture_pkl.pkl", "wb") as file:
    pickle.dump(nural_network.to_json(), file)


# In[ ]:




