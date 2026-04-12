1 . munilagrad can now do Autodifferentiation
2 . added rnn to it 
3. added conv2D , flatten , maxpool2D

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
added conv2D ,maxpool2D and trained the small custom dataset in (test_cnn.py) and work beautifully with nice cuvre loss going down ..

now speed up training munilagrad 
have to implement img2col (which will reduce the nested for loops) ofcourse it will increase the memmory usage , but to get something , someone as to sacrifice and also by implementing this out munilagrad will depend on numpy more than py eventually we have to move toward the cuda 
and have to add mainly winograd convolution , which will reduce the computational multiplication and it will speed uo the training 