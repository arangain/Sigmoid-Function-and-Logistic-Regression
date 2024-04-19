# Explore the sigmoid function / Logistic function
# Explore logistic regression, which uses the sigmoid function


# (have (x86_64), need (arm64e)))
# Implemented alias x86='arch -x86_64 /bin/zsh --login'
#alias arm='arch -arm64 /bin/zsh --login'
#after opening open ~/.zshrc in terminal

# Now, to switch to x86, type x86 in terminal
# type arm to switch to arm64
# type arch to see architecture type


import numpy as np
import matplotlib.pyplot as plt
from plt_one_addpt_onclick import plt_one_addpt_onclick
from lab_utils_common import draw_vthresh
plt.style.use('./deeplearning.mplstyle')

#Example usage of exp() for exponents using NumPy
# Input is an array. 
input_array = np.array([1,2,3])
exp_array = np.exp(input_array)

print("Input to exp:", input_array)
print("Output of exp:", exp_array)

# Input is a single number
input_val = 1  
exp_val = np.exp(input_val)

print("Input to exp:", input_val)
print("Output of exp:", exp_val)

# Sigmoid function implementation
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """

    g = 1/(1+np.exp(-z))
   
    return g


#Testing output for various xalues of x
# Generate an array of evenly spaced values between -10 and 10
z_tmp = np.arange(-10,11)

# Use the function implemented above to get the sigmoid values
y = sigmoid(z_tmp)

# Code for pretty printing the two arrays next to each other
np.set_printoptions(precision=3) 
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])

# In the outputs, values on left are z, values on right are sigmoid(z)

# Plot the function using the matplotlib library
# Plot z vs sigmoid(z)
fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(z_tmp, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)

#Now let's apply logistic regression to the data example of tumor classification
#loading examples and initial values for parameters
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w_in = np.zeros((1))
b_in = 0

plt.close('all') 
addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=True)
#Click on 'run logistic regression' to find best logistic regression model for the training data
# the orange line is 'z' or w*x^(i)+b. it doesn't match the line in a linear regression model. improve results by applying a threshhold

#tick on the toggle 0.5 threshhold to show predictions if a threshhold is applied
#unlike the linear regression model, this one continues to make accurate predictions




plt.show(block=True) 
#when removing this code, the plots don't show up