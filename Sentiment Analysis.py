#%%
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
#import tensorflow_datasets as tfds

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Embedding
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC

#import data
amazon= open('C:/Users/velez/OneDrive/Desktop/School/D213/Task 2/sentiment+labelled+sentences/sentiment labelled sentences/amazon_cells_labelled.txt','r').read()
imdb= open('C:/Users/velez/OneDrive/Desktop/School/D213/Task 2/sentiment+labelled+sentences/sentiment labelled sentences/imdb_labelled.txt','r').read()
yelp= open('C:/Users/velez/OneDrive/Desktop/School/D213/Task 2/sentiment+labelled+sentences/sentiment labelled sentences/yelp_labelled.txt','r').read()

# put all 3 datasets into df
amazon=amazon.split('\n')
imdb=imdb.split('\n')
yelp=yelp.split('\n')
#remove final row
amazon=amazon[:-1]
imdb=imdb[:-1]
yelp=yelp[:-1]
#split on tabs
amazon=[x.split('\t') for x in amazon]
imdb=[x.split('\t') for x in imdb]
yelp=[x.split('\t') for x in yelp]
#new df containing all 3
mydata = []
for x in amazon:
    mydata.append(x)

for x in imdb:
    mydata.append(x)

for x in yelp:
    mydata.append(x)

df = pd.DataFrame(mydata)
print(df.shape)
#%%
#Exploratory data analysis
# Count character frequency 
from collections import Counter
s = df.to_dict(orient='dict')
freq = Counter(s)
print(dict(freq))
#%%
# Clean Data - replace new line with empty, lowercase row names, max length, repl spcl char, split into pieces based on spaces
df[1] = df[1].map(lambda x : x.replace('\n',''))
df[0] = df[0].str.lower()
maxlen = df[0].str.len().max()
df[0] = df[0].str.replace('[^0-9a-zA-Z.,-/ ]', '')
words = {word for line in df[0]
    for word in line.rsplit(",",1)[0].split()}
words
vocablen = len(words)

df.rename(columns =  {0:'text',1:'sentiment_label'}, inplace = True)
text = df.text.values

#Tokenizer, convert text to sequences of integers
tokenizer = Tokenizer(num_words=vocablen)
tokenizer.fit_on_texts(text)
encoded_docs = tokenizer.texts_to_sequences(text)

# Pad sequences with 0s beginning of rows
padded_sequence = pad_sequences(encoded_docs, maxlen = maxlen+1)
print(padded_sequence)
pdd_sqnc_df = pd.DataFrame(padded_sequence)
pdd_sqnc_df.to_csv('C:/Users/velez/OneDrive/Desktop/School/D213/Task 2/padded.csv')
print(tokenizer.word_index)
#%%
# split data 90/10 for train/test 
split_pct = 90
def split_train_test(x, y, split_percentage):
    random_seeds = np.random.rand(x.shape[0])
    split = random_seeds < np.percentile(random_seeds, split_percentage)
    return x[split], y[split], x[~split], y[~split]
x_train, y_train, x_test, y_test = split_train_test(padded_sequence, df['sentiment_label'],  split_pct)
print(x_train.shape)
print(x_test.shape)
print ('this is y_trains shape: ',y_train.shape)
print(y_test.shape)
print(x_test)
#%%
# this fixes the issue about the datatypes, looks like this and above similar looking code are necessary 
split_pct = 90
def split_train_test(x, y, split_percentage):
    random_seeds = np.random.rand(x.shape[0])
    split = random_seeds < np.percentile(random_seeds, split_percentage)
    return x[split], y[split],x[~split], y[~split]
x_train, y_train, x_test, y_test = split_train_test(padded_sequence, df['sentiment_label'], split_pct)
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

vocab_size = vocablen
embed_size = 128
x_train = pd.DataFrame(x_train)
df['text'] = padded_sequence.tolist()
df.to_csv(r'C:/Users/velez/OneDrive/Desktop/School/D213/Task 2/ReadyForAnalysis.csv', index = False, header=True)

model = Sequential()
model.add(Embedding(vocab_size, embed_size, input_shape = (x_train.shape[1],)))
model.add(LSTM(units=60, activation='tanh'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())
#%%
early_stopping_monitor = EarlyStopping(patience=2)
history = model.fit(x_train,y_train, epochs=4, batch_size=32, validation_data=(x_test,y_test))

def plot_learningCurve(history, epochs):
    epoch_range = range(1, epochs+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
#adjust number to match number of epochs
plot_learningCurve(history,4)

print(model.evaluate(x_test, y_test))
#%%
clf = SVC(random_state=0)
clf.fit(x_train, y_train)
SVC(random_state=0)
predictions = clf.predict(x_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()

model.save("Trained_Model.keras")
#%%

