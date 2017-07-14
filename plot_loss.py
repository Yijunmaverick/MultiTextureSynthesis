import matplotlib.pyplot as plt

import json

def getXY(str):
   file = open(str)
   d = json.load(file)

   length = d["it"]

   x = list();
   y = list();

   for i in range (1, length):

     x.append(d["loss_history"][i][0])
     y.append(d["loss_history"][i][1])

   return (x,y)

# main
s1 = './data/train_out/loss_1000.json'

x1, y1 = getXY(s1)

plt.plot(x1, y1, label='')
legend = plt.legend(loc='upper center', shadow=True)

plt.show()

