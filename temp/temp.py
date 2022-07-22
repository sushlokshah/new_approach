from cv2 import add
import numpy as np
import matplotlib.pyplot as plt

base_function = np.ones((80))
add_function = np.sin(np.linspace(0,10,10))*1.5
add_function =  -1* np.exp(np.linspace(0,0.5,3))

def sampling_output(base_function):
    temp = np.zeros((base_function.shape[0]//2))
    grad = np.hstack([base_function[1:],base_function[0]]).reshape(base_function.shape) - base_function
    # grad = np.exp(grad)/np.sum(np.exp(grad))
    for i in range(len(temp)):
        if (i < len(temp) -2):
            temp[i] = base_function[2*i]*np.exp(grad[2*i]*0.8) + base_function[2*i+1]*np.exp(grad[2*i+1]*0.1) + base_function[2*i+2]*np.exp(grad[2*i + 2]*0.1)
            temp[i] = temp[i]/(np.exp(grad[2*i]*0.8) + np.exp(grad[2*i + 1]*0.1) + np.exp(grad[2*i + 2]*0.1))
        else:
            if(temp[i] == len(temp) - 1):
                temp[i] = base_function[2*i]*np.exp(grad[2*i]*0.9) + base_function[2*i+1]*np.exp(grad[2*i+1]*0.1)
                temp[i] = temp[i]/(np.exp(grad[2*i]*0.9) + np.exp(grad[2*i + 1]*0.1))
            else:
                temp[i] = base_function[2*i]

    return temp


base_function[20:23] =  base_function[20:23] + add_function*2
base_function[40:45] = base_function[40:45] + 4
base_function[60:63] = base_function[60:63] + np.exp(np.linspace(0,0.1,3))*2
# base_function[70:73] = base_function[70:73] - np.exp(np.linspace(0,0.8,3))
fig, ax = plt.subplots(4,2)
ax[0][0].plot(base_function)
ax[0][0].plot(base_function,"or")
ax[0][1].plot(base_function)
ax[0][1].plot(base_function,"or")

print(base_function.shape)

ax[1][0].plot(base_function[: :2])
ax[1][0].plot(base_function[: :2],"or")
samp_by_2 = sampling_output(base_function)#*0.7 + base_function[: :2]*0.3
ax[1][1].plot(samp_by_2)
ax[1][1].plot(samp_by_2,"or")
print(base_function[: :2].shape)

ax[2][0].plot(base_function[: :4])
ax[2][0].plot(base_function[: :4],"or")
samp_by_4 = sampling_output(samp_by_2)#*0.7 + samp_by_2[: :2]*0.3
ax[2][1].plot(samp_by_4)
ax[2][1].plot(samp_by_4,"or")

print(base_function[: :4].shape)

ax[3][0].plot(base_function[: :8])
ax[3][0].plot(base_function[: :8],"or")
samp_by_8 = sampling_output(samp_by_4)#*0.7 + samp_by_4[: :2]*0.3
ax[3][1].plot(samp_by_8)
ax[3][1].plot(samp_by_8,"or")


# ax[4][0].plot(base_function[: :16])
# ax[4][0].plot(base_function[: :16],"or")
# samp_by_16 = sampling_output(samp_by_8)
# ax[4][1].plot(samp_by_16)
# ax[4][1].plot(samp_by_16,"or")

print(base_function[: :8].shape)

plt.show()
