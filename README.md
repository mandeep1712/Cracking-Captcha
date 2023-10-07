It is a challenge as well as a project to identify text captchas and give a computer-generated output that can be used as the input to automatically verify captcha locks in websites. This makes a machine crack the captcha without a human and thus deem captchas useless. This uses a CNN model in order to train the dataset

First, preprocess the dataset using the preprocessing code which converts the images into a standard size for processing. The dataset is retrieved from an article from blogging website 'Medium'. The preprocessing step also includes seperating all the characters in the image into single letters/characters which will be identified seperately.

Second, use a filler to add paddings to the images which are saved after preprocessing. Also, note that the number of characters in the text captcha should be 4 and any other combination should be discarded.

Third, the preprocessed single text dataset should be trained using CNN that inputs an image and outputs a text identification that can be recognised by computer as well as the human and it is looped until all the 4 texts are identified which then concatenate together to form the output.

Last, the trained model should be saved in some file and the saved model file should be used to predict some unknown inputs from the test set and thus test its accuracy.
