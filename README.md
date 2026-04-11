1 . munilagrad can now do Autodifferentiation
2 . added rnn to it 
3. added conv2D , flatten 

munilagrad not only supports scaler but also numpy array 
<!-- while doing that we implemented unbroadcasting for backward pass -->
what to do next -->
1 . add conv2D support and all the cnn releated stuff like polling , maxpool etc
2. add lstm , gru 
3. transformers 



now munilagrad can be installed using - pip install munilagrad
all this update is not there in pip install munilagrad
after adding cnn support will update the version form 1.0 -> 2.0


what has done
added conv2D , and trained the small custom dataset in (test_cnn.py) and work beautifully with nice cuvre loss going down ..