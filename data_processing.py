import numpy as np
import json

# Creates a list of strings with the names of the parameters 
def generate_parameters_name(data,changes,tochange):
    params = list(data[0].keys())
    for i, change in enumerate(tochange):
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

# Generate the final list of strings of all the parameters together
changes = [guns_list, npcs_list, npc_minds_list, npc_states_list, npcs_list, npc_minds_list, npc_states_list, npcs_list, npc_minds_list, npc_states_list]
parameters = generate_parameters_name(data,changes,categorical)

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

ntrain = int(ndata*0.8)
ntest = ndata-ntrain
x_train = data_np[:ntrain,:]
x_test = data_np[ntrain:,:]


import keras
from keras import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
embed_dim = 32
lstm_out = 200

model = Sequential()
model.add(Embedding(ndata, embed_dim,input_length = 114))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

batch_size = 32
num_epochs = 3

model.fit(x_train, batch_size=batch_size, epochs=num_epochs)


