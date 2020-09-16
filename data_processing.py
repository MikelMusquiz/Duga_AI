import numpy as np
import json


'''
Input:  params: list of the original parameters, categoricals: list with the categorical 
        parameters, changes: list of lists with the different options for each categorical

Output: A list of strings with the names of the parameters 
'''
def generate_parameters_name(params,categoricals,changes):
    for i, change in enumerate(categoricals):
        ind = params.index(change)
        params.remove(change)
        params[ind:ind] = changes[i]
    
    return params

# Read the data from the  file
with open('data.txt') as json_file:
    data = json.load(json_file)
    
# Get the length of the input data
ndata = len(data)
nparam_initial = len(data[0])

# A list that will contain the names of all the numerical parameters
numerical = []
# A boolean array with 'True' for numerical parameters and 'False' for categorical ones
numerical_ind = np.zeros([nparam_initial],'bool')
for i,param in enumerate(data[0]):
    if type(data[0][param]).__name__ !='str':
        numerical.append(param)
        numerical_ind[i] = True

# A list of categorical parameters
categorical = ['gun_name','npc1_name','npc1_mind','npc1_state','npc2_name','npc2_mind','npc2_state','npc3_name','npc3_mind','npc3_state']
# Lists with all the options for categorical parameters
guns_list = ['AK-47','DB Shotgun','Pistol','Knife', 'Brass Knuckles', 'Gauss rifle', 'SG Pistol', 'Light Knuckles', 'Rampage Knuckles', 'Sharp Knife', 'Light Knife', 'Modified Shotgun', 'TB Shotgun', 'AK-74', 'Ext Mag AK-47', 'Camo AK-47', 'Light AK-47', 'Anomaly Pistol', 'HP Pistol', 'Modded gauss', 'Bump gauss', 'Modded SGP',  'What??', 'Auto pistol']
npcs_list = ['idle soldier', 'patroul soldier', 'idle ninja', 'patroul ninja', 'hostile zombie', 'shy zombie', 'random', 'idle red', 'black idle', 'black patroul', 'idle green',  'idle blue', 'patroul sick', 'idle sick', 'shy sick', 'hostile blurry']
npc_minds_list = ['passive', 'hostile', 'shy']
npc_states_list = ['attacking', 'fleeing', 'idle', 'patrouling', 'searching', 'dying']

changes = [guns_list, npcs_list, npc_minds_list, npc_states_list, npcs_list, npc_minds_list, npc_states_list, npcs_list, npc_minds_list, npc_states_list]
# Generate the final list of strings of all the parameters together
parameters = generate_parameters_name(list(data[0].keys()),categorical,changes)

nparam = len(parameters)
data_np = np.zeros([ndata,nparam])

# Create a bool array with 'True' for the positions in which there are categorical parameters and
# 'False' for the positions with numerical ones
categ_ind = np.zeros([nparam])
categ_ind[6:30] = 1
categ_ind[34:59] = 1
categ_ind[61:86] = 1
categ_ind[88:113] = 1

# Create the final numpy table that contains all the data represented as numbers
for i in range(ndata):
    for j in range(nparam):
        # Put ones at the categorical values of this row         
        if parameters[j] in list(data[i].values()):
            data_np[i,j] =  1
    # Put the numerical values in their places
    data_np[i,np.logical_not(categ_ind)] = np.array(list(data[i].items()))[:,1][numerical_ind]

# Some angle values can be negative, I get the index of those and I make them positive
indexes = np.where(data_np[:,3]<0),3
data_np[indexes] = 360 + data_np[indexes]

# Calculate the dimensions of train and test data
ntrain = int(ndata*0.8)
ntest = ndata-ntrain

nout = 33

# Generate the train and test data
x_train = data_np[:ntrain,:]
y_train = x_train[1:,:nout]
x_train = x_train[:-1,:]
x_test = data_np[ntrain:,:]
y_test = x_test[1:,:nout]
x_test = x_test[:-1,:]


import keras
from keras import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
embed_dim = 32
lstm_out = 50

model = Sequential()
model.add(Embedding(ndata, embed_dim,input_length = nparam))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(output_dim=nout,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])


x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(lstm_out, input_shape=(1,nparam)))
model.add(Dense(output_dim=nout))
model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)
#
#
#print(model.summary())
#
#batch_size = 32
#num_epochs = 3
#
#model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs)

y_pred = model.predict(x_train)

