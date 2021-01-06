import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

loss_info = pd.read_excel('64-output_df_loss.xlsx', index_col=0, engine='openpyxl')

fig = plt.gcf()
plt.figure

### Desired start epoch
start = int(1)

plt.plot(list(range(start, int(loss_info['epochs'].iloc[-1] + 1))), loss_info['loss'][start-1:])
plt.plot(list(range(start, int(loss_info['epochs'].iloc[-1] + 1))), loss_info['val_loss'][start-1:])

plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.yscale('log')
plt.xscale('log')
plt.legend(['Training', 'Validation'], loc='upper right')
axes = plt.gca()
# axes.set_ylim([0,1e-2])
plt.show()
fig.savefig('64-loss.png', dpi=500)
#%%

print(loss_info.min())
print(loss_info.idxmin()+1)
