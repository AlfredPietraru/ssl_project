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
### Paper 1:
[Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)[Code][https://github.com/michuanhaohao/reid-strong-baseline]
The results are obtained using global features from the ResNet50 backbone (we can change it to use MegaDescriptor - better suited for the task?).
Loss function used: [triplet loss]
Transformation method:  Random Erasing Augmentation (REA)
they are using a linear layer at the end to make a classification between all individuals (not enough for us) used **label smoothing** to prevent the model to overfit the training set and to keep itself adaptable. Other loss functions used: **ID LOSS**, **triplet loss**, **center loss**.
Bad idea to use both triplet loss + id loss togheter to optimize feature vector (section 3.5)
The last classification layer does not habe bias for their data set increases accuracy.   
#### Ideas paper1:
initially thinking to have first a classification layer:
to have the 3 known categories from the dataset and an unknown category -> everything the model could not understand could be stored there. 


### Paper 2:
[Deep Metric Learning][https://www.mdpi.com/2073-8994/11/9/1066]
deep metric learning — a subfield of machine learning focused on learning distance metrics through deep neural networks.
metric learning approaches -> creates a metric that can get closer
similar elements, and further away from one another the disimilar ones.

deep metric learning -> use a neural network to give similar embeddings to similar objects, and different embeddings to those disimilar.
I DO NOT SEE THE RELEVANCE OF THE PAPER - already this aspect is handled by the neural network for embeddings MegaDescriptor.   
#### Ideas paper 2:
After obtaining the embbeding maybe use PCA for collapsing them to have a smaller dimensions since thre are not so many identities to 
classify.


### Paper 3:
[Towards Open Set Deep Networks][https://arxiv.org/pdf/1511.06233]
Recognition in real world - is inherently open-set, create of OpenMAX -> estimates the probability of an input being from unknown class.
How to adapt deep networks to support open set recognition?

OpenMax Layer -> extends SoftMax layer (enabling to predict unknown class) -> likelyhood of system recognition failure(probability of given input belong to unknown class) 
drop the restriction for the probability for known classes to sum to 1 , reject inputs far from known inputs -> can handle unknown / unseen classes.
 EVT -> extreme value theory (branchs of statistics that focuses on modelling and understanding rare, extreme events )
THE PAPER IS FROM 2016 really old, good for understanding the task but really old...
penultimate layer of the network -> in our case the results after applying embedding model -> (called Activation Vector AV) 
#### Ideas paper 3:
never crossed my mind to check, does the embedding model properly separates the space -> for all 3 / 4 classes?
If it does there is no point in modifying the weights of the embedding model. 



