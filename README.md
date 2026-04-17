## Purpose of project:
Develop model capable of discovering clustering individual animals from images.
goal of lookup + discovery -> uses the identity database, group the remaining test images into individuals when they do not match any known identity
handle both known and unknown individuals


Feature extraction: -> pre-trained model: MegaDescriptor
competition follows discovery setting -> many test individuals not present in training, assign new images to previously unseen identities
training: images + identities -> 3 species -> lynx, salamandra, turtle carett 
test images: -> contains 4 species (not 3)
-> some images may correspond to known individuals, while others might be new individuals
-> when doing the pretraining i should also include in the pretraining the 4th class from test which is not included into training?


## Task purpose:
We have to do re-identification, it is different from just simple classification


## Resources and materials:
- [Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)[Code][https://github.com/michuanhaohao/reid-strong-baseline]
### Observations:
The results are obtained using global features from the ResNet50 backbone (we can change it to use MegaDescriptor - better suited for the task?).
Loss function used: [triplet loss]
Transformation method:  Random Erasing Augmentation (REA)
they are using a linear layer at the end to make a classification between all individuals (not enough for us) used **label smoothing** to prevent the model to overfit the training set and to keep itself adaptable. Other loss functions used: **ID LOSS**, **triplet loss**, **center loss**.
Bad idea to use both triplet loss + id loss togheter to optimize feature vector (section 3.5)
The last classification layer does not habe bias for their data set increases accuracy.
   

### Ideas inspired from paper:
initially thinking to have first a classification layer:
to have the 3 known categories from the dataset and an unknown category -> everything the model could not understand could be stored there. 




