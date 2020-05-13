import tensorflow
import tflearn
import pickle

try:
    with open("data.pickle","rb") as f:
        words,labels,training,output = pickle.load(f)
except:
    print("data.pickel file not found\nMake sure you run start_or_update.py first")
else:
    x = int(input("Enter the no. of times to train: "))
    
    tensorflow.reset_default_graph()

    net = tflearn.input_data(shape=[None,len(training[0])])
    net = tflearn.fully_connected(net,8)
    net = tflearn.fully_connected(net,8)
    net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    model.fit(training,output,n_epoch=x,batch_size=8,show_metric=True)
    model.save("model.tflearn")
finally:
    print("Finished training")
    input("<Press any key to exit>")
