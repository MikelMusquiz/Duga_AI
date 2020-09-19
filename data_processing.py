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

def generate_chunks(arr,chunk_len,x,y):
    for i in range(len(arr) - chunk_len):
        x_i = arr[i : i + chunk_len]
        y_i = arr[chunk_len + 1,:33]
        x.append(x_i)
        y.append(y_i)
    return x,y

# Delete all rows in which there is no close npc
def clean_useless_data(data,x_list,y_list):
    chunk_len = 100
    ndata = data.shape[0]
    i = 0
    while i < ndata:
        if data[i,59] < 200:
            first = i
            while i < ndata and data[i,59] < 200 :
                i = i + 1
            if i - first >= chunk_len:
                x_list,y_list = generate_chunks(data[first:i,:],chunk_len,x_list,y_list)
        else:
            i = i + 1
    return x_list,y_list
    

def create_numpy_table(filename):
    # Read the data from the  file
    with open(filename) as json_file:
        data = json.load(json_file)
        
    # Get the length of the input data
    ndata = len(data)
    nparam_initial = len(data[0])
    
    # Create a boolean array with 'True' for numerical parameters and 'False' for categorical ones,
    # it has the size of the original data
    numerical_ind = np.zeros([nparam_initial],'bool')
    for i,param in enumerate(data[0]):
        if type(data[0][param]).__name__ !='str':
            #numerical.append(param)
            numerical_ind[i] = True
    
    nparam = len(parameters)
    data_np = np.zeros([ndata,nparam])
    
    # Create a bool array with 'True' for the positions in which there will be categorical parameters and
    # 'False' for the positions with numerical ones, it has the size of the new numpy array
    categ_ind = np.zeros([nparam])
    categ_ind[6:30] = 1
    categ_ind[34:59] = 1
    categ_ind[61:86] = 1
    categ_ind[88:113] = 1
    
    # Create the numpy table that contains all the data represented as numbers
    for i in range(ndata):
        for j in range(nparam):
            # Put ones at the categorical values of this row         
            if parameters[j] in list(data[i].values()):
                data_np[i,j] =  1
        # Put the numerical values in their places
        data_np[i,np.logical_not(categ_ind)] = np.array(list(data[i].items()))[:,1][numerical_ind]
        
    return data_np

# A list of the categorical parameters
categorical = ['gun_name','npc1_name','npc1_mind','npc1_state','npc2_name','npc2_mind','npc2_state','npc3_name','npc3_mind','npc3_state']

# Lists with all the options for categorical parameters
guns_list = ['AK-47','DB Shotgun','Pistol','Knife', 'Brass Knuckles', 'Gauss rifle', 'SG Pistol', 'Light Knuckles', 'Rampage Knuckles', 'Sharp Knife', 'Light Knife', 'Modified Shotgun', 'TB Shotgun', 'AK-74', 'Ext Mag AK-47', 'Camo AK-47', 'Light AK-47', 'Anomaly Pistol', 'HP Pistol', 'Modded gauss', 'Bump gauss', 'Modded SGP',  'What??', 'Auto pistol']
npcs_list = ['idle soldier', 'patroul soldier', 'idle ninja', 'patroul ninja', 'hostile zombie', 'shy zombie', 'random', 'idle red', 'black idle', 'black patroul', 'idle green',  'idle blue', 'patroul sick', 'idle sick', 'shy sick', 'hostile blurry']
npc_minds_list = ['passive', 'hostile', 'shy']
npc_states_list = ['attacking', 'fleeing', 'idle', 'patrouling', 'searching', 'dying']

# List of the lists of changes
changes = [guns_list, npcs_list, npc_minds_list, npc_states_list, npcs_list, npc_minds_list, npc_states_list, npcs_list, npc_minds_list, npc_states_list]

example = {"pl_speed": 256, "pl_pos_x": 5, "pl_pos_y": 5, "pl_angle": 56.75, "pl_armor": 5, "pl_health": 35, "gun_name": "Brass Knuckles", "gun_reload": 0, "gun_mag": 0, "gun_bullets": 0, "npc1_ID": 17, "npc1_name": "hostile blurry", "npc1_mind": "hostile", "npc1_state": "patrouling", "npc1_dist": 504.0, "npc2_ID": 3, "npc2_name": "patroul ninja", "npc2_mind": "hostile", "npc2_state": "patrouling", "npc2_dist": 559.3612428475896, "npc3_ID": 2, "npc3_name": "idle ninja", "npc3_mind": "hostile", "npc3_state": "idle", "npc3_dist": 578.0181658045013}

# Generate the final list of strings of all the parameters together
parameters = generate_parameters_name(list(example.keys()),categorical,changes)


data_np = create_numpy_table('data_log_0.txt')

# Some angle values can be negative, I get the index of those and I make them positive
bad_angle_idx = np.where(data_np[:,3]<0)
data_np[bad_angle_idx] = 360 + data_np[bad_angle_idx]

x_list = []
y_list = []
x_list,y_list= clean_useless_data(data_np,x_list,y_list)
x_data = np.array(x_list)
y_data = np.array(y_list)
print("x_data shape:")
print(x_data.shape)
print("y_data shape:")
print(y_data.shape)

# Calculate the dimensions of train and test data
ndata = x_data.shape[0]
ntrain = int(ndata*0.8)
ntest = ndata-ntrain

# Shuffle the data
shuf_idx = np.random.permutation(range(ndata))
x_data = x_data[shuf_idx]
y_data = y_data[shuf_idx]

# Dived the data in train and test
x_train = x_data[:ntrain]
y_train = y_data[:ntrain]
x_test = data_np[ntrain:]
y_test = y_data[ntrain:]

nparam = x_train.shape[2]
nout = y_train.shape[1]

import keras
from keras import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
embed_dim = 32
lstm_out = 50
#
#x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
#x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(lstm_out, input_shape=(100,nparam)))
model.add(Dense(output_dim=nout))
model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
#×model.fit(x_train, y_train, epochs=30, batch_size=100, verbose=2)

y_pred = model.predict(x_train[8].reshape([1,100,114]))

