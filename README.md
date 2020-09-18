# Isolated-Digit-Recognition
The project aims to classify spoken digits, using a GMM model.

# Methodology

The  dataset  is  split  into  train  and  test  sets.   The  training  dataset  comprising  of  digitsrecorded by different speakers are used to train GMM models,one for each digit. The clas-sification is to be based on the digit uttered. Each training audio is normalised,trimmed toremove silences, split into frames of 20ms each and mfcc,delta and double delta(13 dimen-sions each) is computed in each frame and appended to form a feature vector.The feature vectors of a particular digit spread over frames and audio files are appended to form a 39*N matrix (where N is the total number of feature vectors for a digit) which is fed to the GMM model for training.  Separate GMM models (comprising of 16 modes each) are trained for each digit.  The average log likelihood values(averaged over frames) for each testdata is computed against all the GMM models, out of which the maximum is evaluated and the test digit is classified as the digit associated with that particular model.  A corresponding confusion matrix is generated for the test dataset.

# Output

![Alt text](relative/path/to/img.jpg?raw=true "Title")
