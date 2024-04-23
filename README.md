I was able to get an accuracy of 71.82% with four styles: cubism, color field painting, mannerism late renaissance, and ukiyo-e

NEW MODEL IS UPLOADED TO GITHUB: art_model_4class_71_8.pt 

The reasoning for choosing these genres was that they are all fairly distinct in their styles and each of these styles has a particular component that I thought was would give a more diverse representation of art when put altogether: 

cubism - abstract with many details of shapes and figures

color field painting - mostly plain abstract with minimal details

mannerism late renaissance - human figures and portraits

ukiyo-e - landscapes and nature

I trained using 1100 examples per class and trained for 5 epochs. I did not change the model from how it was before (no dropout or batch norm), but I did adjust the scheduler (you can see the new one in the updated notebook that I uploaded to github).

I have created a confusion matrix (one with numbers as labels and one with the actual genre names as labels). 

NOTE: If you attempt to run just torch.load('art_model_4class_71_8.pt') you will get an error bc the model was trained on a gpu and you have a cpu (unless you are doing this in colab with gpu). Instead you will have to do model = torch.load('art_model_4class_71_8.pt',map_location=torch.device('cpu')).

Akhil - pls fill out the results and analysis of results section on the slides (include a confusion matrix from the repo, and use the uploaded model to find some correctly and incorrectly classified examples to talk about - maybe 1 correct and 1 incorrect from each genre?). You can probably just pick some images randomly from each genre and check if they get predicted properly. Talk to Teresa if you have trouble with preprocessing an image before inputting it to the model to get a prediction. In the slides when you are referring to images and their predicted genre, put the original image in the slide, not the preprocessed version.

Teresa - pls run the optimization with an example from each of those four genres with the new model and put the results in the slides. DONT FORGET TO UPDATE YOUR PREPROCESSING CODE TO MATCH THE NEW CODE WITH REFLECT PADDING. Unlike akhil (who will show original images), you should probably put the preprocessed image with its corresponding heatmap in the slides.

Btw, ^^^ these are all just my suggestions (what I would've said if I wasn't asleep) so if you have a better way to do something or you want to do something differently, I don't mind.

Hopefully the slides are all done before 11am and we can start planning what to say and recording at 11am. GOOD WORK YAWL!!
