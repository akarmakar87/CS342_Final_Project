I was able to get an accuracy of 71.82% with four styles: cubism, color field painting, mannerism late renaissance, and ukiyo-e

NEW MODEL IS UPLOADED TO GITHUB: art_model_4class_71_8.pt 

The reasoning for choosing these genres was that they are all fairly distinct in their styles and each of these styles has a particular component that I thought was would give a more diverse representation of art when put altogether: 

cubism - abstract with many details of shapes and figures

color field painting - mostly plain abstract with minimal details

mannerism late renaissance - human figures and portraits

ukiyo-e - landscapes and nature

I trained using 1100 examples per class and trained for 5 epochs. I did not change the model from how it was before (no dropout or batch norm), but I did adjust the scheduler (you can see the new one in the updated notebook that I uploaded to github).

I have created a confusion matrix (one with numbers as labels and one with the actual genre names as labels). 

Akhil - pls fill out the results and analysis of results section on the slides (include a confusion matrix, and use the uploaded model to find some correctly and incorrectly classified examples to talk about)

Teresa - pls run the optimization with an example from each of those four genres with the new model

Hopefully the slides are all done before 11am and we can start planning what to say and recording at 11am.
