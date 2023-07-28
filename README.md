# Where's Waldo - Larger Sample Size, Less Privacy Risk

This repository provides the code for our paper's results.

### Where's Waldo effect
- privacyrisk.R creates the Where's Waldo effect Figure in section _The Where's Waldo effect_.

### Churn application
- dp_gan.py describes the fully connected GAN.
- predictions_resample.py describes section _Predicting real (out-of-sample) churn behavior._ and _Disparate impact of privacy protection on sensitive groups._
- data_sharing_application.py describes the section _Sharing protected data between departments._ 

### Pharmaceutical marketing application
- dp_panelgan.py describes the GAN with attention mechanism.
  
### Privacy attack
- privacy_attack.py provides the code for replicating our privacy attack with detailed comments.

We recommend using Google Colab [with a custom VM from Google Cloud Platform](https://research.google.com/colaboratory/marketplace.html#:~:text=The%20easiest%20way%20to%20connect,details%20of%20your%20Colab%20deployment.&text=Fill%20in%20the%20resulting%20dialog,VM%20configuration%20and%20click%20Connect.) (especially for the privacy attack). Most of the time, we used 64 CPUs and around 120GB of RAM.  
