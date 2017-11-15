# fer2013 dataset:
# Training       28709

#Imports
from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np
import random

# emotion labels from FER2013:
emotion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
           'Sad': 4, 'Surprise': 5, 'Neutral': 6}
emo     = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']

#This function is taking pixel value from each row and doing some operations as mentioned below.
#Function is being performed for each row of pixel value continuously
def reconstruct(pix_str, size=(48,48)):
    #Declare empty pix_arr array
    pix_arr = []
    
    #The method split() returns a list of all the words in the string, using str as the separator (splits on all whitespace if left unspecified)
    #For each row of pixel values in given data, split as on whitespaces and append each value to pix_arr.
    #A new pix_arr is created for each row containing some number of values
    for pix in pix_str.split():
        pix_arr.append(int(pix))
    
    #Convert pix_arr list to array using numpy's asarray() function
    pix_arr = np.asarray(pix_arr)
    
    #return array reshaped as size mentioned in function parameter. An array of 48 arrays where each array contains 48 elements(As far as i could understand).
    #Note that this returns number of rows =  number of rows in our given dataset to this function.
    #It returns that many arrays.
    return pix_arr.reshape(size)

def emotion_count(y_train, classes, verbose=True):
    emo_classcount = {}
    
    #This code is used to indicate that Disgust is classified under Angry
    print ('Disgust classified as Angry')
    #loc is used to see wherever we find specific value, do something with it.
    #This code sets the value of 1 to 0 wherever it finds disgust in y_train(data.emotion column)
    #All disgusts are now set to Angry
    y_train.loc[y_train == 1] = 0
    #Since classification doesnt require disgust, remove it from the classes variable
    classes.remove('Disgust')
    print (y_train)
    #for "row_number, row in enumerate(cursor)"
    #For each class in classes(2 classes), read below
    for new_num, _class in enumerate(classes):
        #Both emotion and y_train contain numerical values for emotion.
        #Wherever you find y_train(emotion column value) == (value of class selected in for loop from emotion declared on top), replace it with new_num i.e. 0
        #That means, for Angry, whereever you find angry in dataset, make angry = 0
        #For Happy, wherever you find happy in dataset, make happy = 1
        #Now y_train will have two sets, where each set contains 3493 elements. In one set, angry is set to 0 and in another happy is set to 1
        y_train.loc[(y_train == emotion[_class])] = new_num
        #Count class_count in y_train(consider both sets) where y_train value = 0 and 1(in both sets).
        #Note: This should give two values for two sets. As per our set it gives 1339 for Angry in 1st set of y_train and 2154 for Happy in second set of y_train
        class_count = sum(y_train == (new_num))
        #If verbose is true, just print out row_number, class(ANGRY OR HAPPY), Number of respective samples.
        if verbose:
            print ('{}: {} with {} samples'.format(new_num, _class, class_count))
        #Generate two emo_classcount for Angry and for Happy
        #emo_classcount will contain two values i.e. {'Angry' : (0,1343)} & {'Happy' : (1,2150)}
        emo_classcount[_class] = (new_num, class_count)
    #y_train.values will print out all the values i.e. 0's and 1's in complete y_train
    return y_train.values, emo_classcount

def load_data(sample_split=0.3, usage='Training', to_cat=True, verbose=True,
              classes=['Angry','Happy'], filepath='C:/Users/Sai/Desktop/Facial_Expression_Recognition/Data_Generation/data/fer2013.csv'):
    #Read the file mentioned in the filepath
    df = pd.read_csv(filepath)
    
    # print df.tail()
    # print df.Usage.value_counts()
    
    #Store the dataframe with usage parameter(Parameter in function) as column value(Usage column in dataset)
    #df contains only rows with Usage=Training
    df = df[df.Usage == usage]
    
    #Create empty frames array
    frames = []
    
    #Append one more value to our classes array
    classes.append('Disgust')
    
    #for each value in classes, store dataframe class_df = (our current df where df's emotion column = our emotion array at top [_class]
    #This creates a sorted df of three emotions mentioned in our classes array i.e. Anger:0, Happy:3 and Disgust:1
    #class_df contains 3 frames for each of the emotion
    for _class in classes:
        class_df = df[df['emotion'] == emotion[_class]]
        frames.append(class_df)
    #data contains concatenated value of all frames one below other as axis=0. If axis = 1, concatenation would have been column wise.
    data = pd.concat(frames, axis=0)
    #"random" is used for random variable generator
    #"random.sample" is used to select k unique random elements from sequence or set.
    #list(data.index) will populate a list of 1st column i.e. index(Number) of each row... like 1,2,3,4.... index of each image
    #length of our data is number of rows i.e. 11646(print data.index). (11646x0.3) ~ 3493
    #Goal is to randomly pick 3494 elements from data i.e choose 3493 indexes.
    #Since population for our sample is data.index(Index values of images), new resulting list is list of indexes(valid indexes based on sample_split)
    rows = random.sample(list(data.index), int(len(data)*sample_split))
    
    #You can use .loc(label based) or .iloc(integer based). with .ix, we can decide to index positionally or via labels depending on datatype of index
    #Note that ix is deprecated now
    #This instruction replaces our previous data of 11646 indexes with new data of 3493 indexes as per selected rows in previous statement
    data = data.ix[rows]
    #Fill 1st {} with usage(function parameter)
    #Fill 2nd {} with classes(function parameter)
    #Fill 3rd {} with data.shape.data.shape will give us (Number of rows, Number of columns) for our data
    print ('{} set for {}: {}'.format(usage, classes, data.shape))

    #Use of pandas.DataFrame.apply function
    #DataFrame.apply(func, axis=0, broadcast=False, raw=False, reduce=None, args=(), **kwds)
    #Point to note is, by default axis = 0 i.e. apply function to each column. If axis = 1, then apply function to each row.
    #Now our data['pixels'] contain 48 x 48 arrays of pixel values provided after reconstruct function.
    data['pixels'] = data.pixels.apply(lambda x: reconstruct(x))
    
    #1st mat is object parameter for .array(). for each row in data.pixels mat is something.
    #X is array of all the images
    x = np.array([mat for mat in data.pixels]) # (n_samples, img_width, img_height)

    #print (x.shape)
    #Reshape returns an array containing the same data with new shape    
    #x.shape prints out (3493,48,48). x.shape[1] = 48 & x.shape[2] = 48
    #The previous statement indicated that our array x contains 3493 elements(arrays) where each element has 48 rows(arrays) and each row contains 48 elements(not arrays)
    #Note: For A.reshape(-1, 28*28), It means, that the size of the dimension, for which you passed -1, is being inferred.
    #Thus, previous statement means that "reshape A so that its second dimension has a size of 28*28 and calculate the correct size of the first dimension"
    #Note: reshape(-1) means it is unknown dimension(row unknown & column unknown) and we want numpy to figure out. What numpy will do is it will remove all the emenets in         arrays and put it in a single array.
    #Note: reshape(-1,1) means we have provided column as 1 but rows as unknown. Hence it will give out each element in separate array in each row.
    #Note: reshape(-1, 1, 48, 48) means (row unknown, column 1, each element contains 48 rows and 48 columns)
    X_train = x.reshape(-1, 1, x.shape[1], x.shape[2])
    print (X_train)
    #return of this function is y_train.values and emo_classcount - > #emo_classcount will contain two values i.e. {'Angry' : (0,1343)} & {'Happy' : (1,2150)}
    #y_train will contain - > y_train.values
    #new_dict will contain - > emo_classcount
    y_train, new_dict = emotion_count(data.emotion, classes, verbose)
    print (new_dict)
    #to_cat is given in function parameter. If its true, do something
    print (y_train)
    if to_cat:
        #to_categorical is present in keras. This converts class vector(integers) to binary class matrix
        #A classification model with multiple classes doesn't work well if you don't have classes distributed in a binary matrix.
        #You use to_categorical to transform your training data before you pass it to your model.
        #If your training data uses classes as numbers, to_categorical will transform those numbers in proper vectors for using with models. You can't simply train a 	classification model without that.
        #Note: Moral of all this is, it will create two columns in your variable, one for angry(0) and one for happy(1)
        #Wherever it finds angry, it will set 1 in that column(Angry column), wherever it finds happy, it will set 1 in that column(Happy column)
        y_train = to_categorical(y_train)
    #X_train is array of our images in training_set
    #y_train is array  of corresponding emotion i.e. value "1" in either of the emotion(Angry or Happy column as mentioned in previous step)
    #new_dict is more of a key value pair - > {'Happy': (1, 2200), 'Angry': (0, 1293)}
    return X_train, y_train, new_dict

def save_data(X_train, y_train, fname='', folder='C:/Users/Sai/Desktop/Facial_Expression_Recognition/Data_Generation/data/'):
    #Save an array to binary file in NumPy .npy format
    #save(file, arr, allow_pickle=True, fix_imports=True)
    #In short save X_train to this "folder + 'X_train' + fname" file. Same for the next statement.
    np.save(folder + 'X_train' + fname, X_train)
    np.save(folder + 'y_train' + fname, y_train)

#This statement means if file is being run directly do this...
#This statement makes sense when file is run using "python abcd.py" or clicking run arrow above in IDE
#In IDE when you click run arrow, your program runs this code.
if __name__ == '__main__':
    # makes the numpy arrays ready to use:
    print ('Making moves...')
    #Declare all classes in emo
    emo = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']
    '''
    X_train, y_train, emo_dict = load_data(sample_split=1.0,
                                           classes=emo,
                                           usage='PrivateTest',
                                           verbose=True)
    '''
    #sample_split indicates all the data as training data
    #classes variable indicates all the classes to be tested
    #Returned values will be stored to respective variables
    X_train, y_train, emo_dict = load_data(sample_split=1.0,
                                           classes=emo,
                                           usage='Training',
                                           verbose=True)
    print ('Saving...')
    #Use save_data function
    save_data(X_train, y_train, fname='_train')
    print (X_train.shape)
    print (y_train.shape)
    print ('Done!')
