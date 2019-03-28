Note: we add the notebooks used in GoogleColab for the sake of completeness since, without our Google Drive folder structure, the model cannot load data, etc.
Note2: we add two fine-tuning notebooks since we were not able to train properly maskrcnn. RetinaNet was trained but not with the proper parameters since we overshooted the minimum probably due to one (or both) of these reasons:
  * Learning rate was too high
  * We "only" freezed the backbone, ResNet50, all the other layers were trained.
Maybe we should only, train the classifier to avoid diverging from original weights at the beginning of training (we only want to tune the weights NOT change them significantly).
Feel free to change the paths to work for your case
