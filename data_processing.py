from tensorflow import keras
import numpy as np
import json


#guns_list = {'AK-47': 0,'DB Shotgun': 1 ,'Pistol': 2,'Knife': 3, 'Brass Knuckles': 4, 'Gauss rifle': 5, 'SG Pistol': 6, 'Light Knuckles': 7, 'Rampage Knuckles': 8, 'Sharp Knife': 9, 'Light Knife': 10, 'Modified Shotgun': 11, 'TB Shotgun': 12, 'AK-74': 13, 'Ext Mag AK-47': 14, 'Camo AK-47': 15, 'Light AK-47': 16, 'Anomaly Pistol': 17, 'HP Pistol': 18, 'Modded gauss': 19, 'Bump gauss': 20, 'Modded SGP': 21,  'What??': 22, 'Auto pistol': 23}
#npcs_list = {'idle soldier': 0, 'patroul soldier': 1, 'idle ninja': 2, 'patroul ninja':3, 'hostile zombie': 4, 'shy zombie': 5, 'random': 6, 'idle red': 7, 'black idle': 8, 'black patroul': 9, 'idle green': 10,  'idle blue': 11, 'patroul sick': 12, 'idle sick': 13, 'shy sick': 14, 'hostile blurry': 15}
#npc_minds_list = {'passive': 0, 'hostile': 1, 'shy': 2}
#npc_states_list = {'attacking': 0, 'fleeing': 1, 'idle': 2, 'patrouling': 3, 'searching': 4, 'dying': 5}

def generate_parameters_name(data,changes,tochange):
    params = list(data[0].keys())
    for i, change in enumerate(tochange):
        ind = params.index(change)
        params.remove(change)
        params[ind:ind] = changes[i]
    
    return params

guns_list = ['AK-47','DB Shotgun','Pistol','Knife', 'Brass Knuckles', 'Gauss rifle', 'SG Pistol', 'Light Knuckles', 'Rampage Knuckles', 'Sharp Knife', 'Light Knife', 'Modified Shotgun', 'TB Shotgun', 'AK-74', 'Ext Mag AK-47', 'Camo AK-47', 'Light AK-47', 'Anomaly Pistol', 'HP Pistol', 'Modded gauss', 'Bump gauss', 'Modded SGP',  'What??', 'Auto pistol']
npcs_list = ['idle soldier', 'patroul soldier', 'idle ninja', 'patroul ninja', 'hostile zombie', 'shy zombie', 'random', 'idle red', 'black idle', 'black patroul', 'idle green',  'idle blue', 'patroul sick', 'idle sick', 'shy sick', 'hostile blurry']
npc_minds_list = ['passive', 'hostile', 'shy']
npc_states_list = ['attacking', 'fleeing', 'idle', 'patrouling', 'searching', 'dying']


with open('data.txt') as json_file:
    data = json.load(json_file)
ndata = len(data)
nparam = len(data[0])

numerical = []
numerical_ind = np.zeros([nparam],'bool')
for i,param in enumerate(data[0]):
    if type(data[0][param]).__name__ !='str':
        numerical.append(param)
        numerical_ind[i] = True
    
categorical = ['gun_name','npc1_name','npc1_mind','npc1_state','npc2_name','npc2_mind','npc2_state','npc3_name','npc3_mind','npc3_state']
changes = [guns_list, npcs_list, npc_minds_list, npc_states_list, npcs_list, npc_minds_list, npc_states_list, npcs_list, npc_minds_list, npc_states_list]
parameters = generate_parameters_name(data,changes,categorical)

aux1 = len(guns_list)
aux2 = (len(npcs_list)-1)
aux3 = (len(npc_minds_list)-1)
aux4 = (len(npc_states_list)-1)
nparam = nparam + aux1 -1 + aux2*3 + aux3*3 + aux4*3 
data_np = np.zeros([ndata,nparam])

categ_ind = np.zeros([nparam])
categ_ind[6:30] = 1
categ_ind[34:59] = 1
categ_ind[61:86] = 1
categ_ind[88:113] = 1

print(parameters=='AK-47')

for i in range(ndata):
    #print(np.array(list(moment.items()))[:,1])
    for j in range(nparam):            
        if parameters[j] in list(data[i].values()):
            data_np[i,j] =  1

    aux = np.array(list(data[i].items()))[:,1][numerical_ind]

    data_np[i,np.logical_not(categ_ind)] = aux


