import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

confusion_matrix = np.array([[2002, 10, 19, 50, 2, 6, 6, 2, 1, 3, 17, 0],
                             [11, 417, 0, 1, 0, 0, 106, 0, 0, 0, 0, 0],
                             [24, 1, 946, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                             [2, 0, 0, 84, 0, 0, 0, 0, 0, 0, 0, 0],
                             [2, 0, 0, 0, 55, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 133, 0, 0, 0, 0, 0, 0],
                             [0, 16, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 53, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 101, 43, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0, 0, 13, 130, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]])

labels = ['Other', 'Meal_Preparation', 'Relax', 'Eating', 'Work', 'Sleeping',
          'Wash_Dishes', 'Bed_to_Toilet', 'Enter_Home', 'Leave_Home', 'Housekeeping', 'Respirate']

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, cmap="viridis", fmt='g', cbar=True, xticklabels=labels, yticklabels=labels)
plt.title('Milan bi-LSTM')
plt.xticks(rotation=45)
plt.title('Confusion Matrix biLSTM Aruba')
plt.show()