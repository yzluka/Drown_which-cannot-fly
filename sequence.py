import os

for i in range(0, 25):
    os.system('py Generator.py')
    os.system('move Belief_Info_Full.npy ' + 'Belief_Info_Full_' + str(i) + '.npy')
    os.system('move Belief_Info_Full.png ' + 'Belief_Info_Full_' + str(i) + '.png')
    os.system('move GT_Info_Full.npy ' + 'GT_Info_Full_' + str(i) + '.npy')
    os.system('move GT_Info_Full.png ' + 'GT_Info_Full_' + str(i) + '.png')
# os.system('py blurring.py')
# os.system('py Interpreter.py')
# os.system('py Normalizer.py')
