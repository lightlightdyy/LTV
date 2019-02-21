

```python
import pandas as pd
import numpy as np
import re
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
from scipy.stats import ranksums
```


```python
path='/nfs/project/dengyuying/LTV/'
file_week='500_passenger_weeks_2017.csv'
data = pd.read_csv(path+file_week)
data.drop('Unnamed: 0', axis=1, inplace=True)
```


```python
def func(t):
    return [float(l) for l in re.split(string=t, pattern='[\s\[\]]') if l != '']

data_lst = data[['finish_order_num','pas_complaint_dri_nums','sum_coupon_spend']].applymap(func)
```


```python
xx=data_lst
yy=data_lst['finish_order_num']
DAYSINC=len(yy[0])  #52 weeks
fe_num=3
decay=6 #预测未来6周打车频次
```


```python
from sklearn.model_selection import train_test_split
X_train0, X_test0, y_train0, y_test0 = train_test_split(xx, yy, test_size=0.3)
```


```python
train=np.zeros(0)
for i in range(X_train0.shape[0]):
    tmp = X_train0.iloc[i,:]
    for j in range(len(tmp[0])):
        a=np.array(list((tmp[0][j],tmp[1][j],tmp[2][j])))
        train=np.append(train,a)
print(len(train))

X_train_tmp = train.reshape(X_train0.shape[0],DAYSINC,fe_num)
print(X_train_tmp.shape)

test=np.zeros(0)
for i in range(X_test0.shape[0]):
    tmp = X_test0.iloc[i,:]
    for j in range(len(tmp[0])):
        a=np.array(list((tmp[0][j],tmp[1][j],tmp[2][j])))
        test=np.append(test,a)
print(len(test))
X_test_tmp = test.reshape(X_test0.shape[0],DAYSINC,fe_num)
print(X_test_tmp.shape)
```

    54600
    (350, 52, 3)
    23400
    (150, 52, 3)



```python
length_x=51
length_y=1
X_train = X_train_tmp[:,0:length_x,:]
X_test = X_test_tmp[:,0:length_x,:]
print(X_train.shape,X_test.shape)

y_train = X_train_tmp[:,length_x:52,0:1].reshape(X_train0.shape[0],1)
y_test = X_test_tmp[:,length_x:52,0:1].reshape(X_test0.shape[0],1)
print(y_train.shape,y_test.shape)
```

    (350, 51, 3) (150, 51, 3)
    (350, 1) (150, 1)



```python
training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep

# LSTM Neural Network's internal structure
n_hidden = 3 # Hidden layer num of features  
n_classes = 1 # Total classes (should go up, or should go down)

tf.reset_default_graph()  #为了更改模型参数

# Training
learning_rate = 0.05  #0.005
lambda_loss_amount = 0.0015
training_iters = training_data_count * 500  # Loop 300 times on the dataset
batch_size = 50
display_iter = 30000  # To show test set accuracy during training

# Some debugging info
print("Some useful info to get an insight on dataset's shape and normalisation:")
print("(X shape, y shape, every X's mean, every X's standard deviation)")
print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")
```

    Some useful info to get an insight on dataset's shape and normalisation:
    (X shape, y shape, every X's mean, every X's standard deviation)
    (150, 51, 3) (150, 1) 1.2185080610021788 3.3231635944093023
    The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.



```python
def extract_batch_size(_train, step, batch_size):
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = []
    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s.append(_train[index])
    return batch_s
def one_hot(y_, n_classes=n_classes):
    y_ = np.array(y_).reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS
```


```python
def LSTM_RNN(_X, _weights, _biases):
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, n_input])
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=0.8, state_is_tuple=True) #不遗忘
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=0.9, state_is_tuple=True)
    lstm_cell_3 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    #lstm_cell_4 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
     #lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1], state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1,lstm_cell_2,lstm_cell_3], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)
    lstm_last_output = outputs[-1]  

    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']
 

```


```python
tf.reset_default_graph()  #为了更改模型参数
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


# Graph weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden,1]))  #mean=1.0
}
#biases = {
#    'hidden': tf.Variable(tf.random_normal([n_hidden])),
#    'out': tf.Variable(tf.random_normal([n_classes]))
#}

biases={
        'hidden':tf.Variable(tf.constant(0.1,shape=[n_hidden,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }

pred = LSTM_RNN(x, weights, biases)

# Loss, optimizer and evaluation
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
# L2 loss prevents this overkill neural network to overfit the data

# cross entropy loss function 交叉熵损失函数，分类问题 ; mean squared error 均方误差，回归问题
LOSS = tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(y, [-1]))) + l2
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(LOSS)

#saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
#module_file = tf.train.latest_checkpoint()    

#mse = tf.losses.mean_squared_error(y, pred)


# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

test_losses = []
test_MSE = []
train_losses = []
train_MSE = []

step = 1
while step * batch_size <= training_iters:
    batch_xs = extract_batch_size(X_train, step, batch_size)
    batch_ys = extract_batch_size(y_train, step, batch_size)

    # Fit training using batch data
    _, loss = sess.run(
        [optimizer, LOSS],
        feed_dict={
            x: batch_xs,
            y: batch_ys
        }
    )
    train_losses.append(loss)
    #train_MSE.append(MSE)

    # Evaluate network only at some steps for faster training:
    if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):

        # To not spam console, show training accuracy/loss in this "if"
        print("Training iter #" + str(step*batch_size) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) )

        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        prediction,loss= sess.run(
            [pred,LOSS],
            feed_dict={
                x: X_test,
                y: y_test
            }
        )
        test_losses.append(loss)
        #test_MSE.append(MSE)
        
        #rmse=np.sqrt(mean_squared_error(prediction,y_test))
        
        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) 
             )

    step += 1

print("Optimization Finished!")

```

    Training iter #50:   Batch Loss = 16.475519
    PERFORMANCE ON TEST SET: Batch Loss = 11.101472854614258
    Training iter #30000:   Batch Loss = 8.260573
    PERFORMANCE ON TEST SET: Batch Loss = 7.231837272644043
    Training iter #60000:   Batch Loss = 4.534149
    PERFORMANCE ON TEST SET: Batch Loss = 7.88058614730835
    Training iter #90000:   Batch Loss = 6.408723
    PERFORMANCE ON TEST SET: Batch Loss = 6.948380470275879
    Training iter #120000:   Batch Loss = 4.753021
    PERFORMANCE ON TEST SET: Batch Loss = 6.323665142059326
    Training iter #150000:   Batch Loss = 2.726341
    PERFORMANCE ON TEST SET: Batch Loss = 7.008629322052002
    Optimization Finished!



```python
predictions, final_loss = sess.run(
    [pred, LOSS],
    feed_dict={
        x: X_test,
        y: y_test
    }
)

test_losses.append(final_loss)

print("FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) )


# MAPE = np.mean(np.abs(y_test - predictions) / y_test)   #inf.  ture=0
MSE = np.mean((y_test - predictions)**2)
print(MSE)

#print(predictions-y_test)

#print(y_test)


```

    FINAL RESULT: Batch Loss = 7.284758567810059
    6.526355296272805



```python
import matplotlib.pyplot as plt
error = np.abs(y_test - predictions)
plt.figure()
plt.plot(list(range(len(predictions))), predictions, color='b',label='predict')
#plt.plot(list(range(len(y_test))), y_test,  color='y',label='true')
plt.plot(list(range(len(error))), error,  color='r',label='error')
plt.legend(loc='upper right')
plt.show()
```


![png](https://github.com/lightlightdyy/LTV/blob/master/images/output_12_0.png)



```python
plt.plot(list(range(len(error))), error,  color='r')

plt.show()
```


![png](https://github.com/lightlightdyy/LTV/blob/master/images/output_13_0.png)

