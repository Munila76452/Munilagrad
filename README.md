1 . munilagrad can now do Autodifferentiation
2 . added rnn to it 
3. added conv2D , flatten , maxpool2D
4. added img2col and col2img to optimise the code like before it takes 0.06 sec to train an cnn with 4 pic after this implemntation it takes 0.014 sec to train the same network , that an massive update so this is well documented in issue#1 and you can view how it is implmented ,and also its img2col.md file for full understanding 
5. munilagrad succesfully trained a mnist dataset with fial loss of 0.504
6.munilagrad can now record audio and can plot the spectograme (written from scratch and has same ~ match from in-buit scipy.spectograme fucn) 


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