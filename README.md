We support only single gpu execution on cuda devices.
If you want to run a clustering-method, you need to execute one of the (python) files in the root directory where the name ends with a "_MAIN".  Run the file in the command line interface. If executing the _MAIN.py applications, you need to pass some command line arguments to configure some required execution parameters. Let us begin with the instruction for the simplest starting file "MAIN.py":
You need to pass the following command line arguments with:
> python MAIN.py

### gpu:
On a multi-gpu machine, each gpu has an identifier starting with 0 and ending with the number of devices - 1. Usually, we operate on nvidia cuda devices. So the integer passed with -gpu argument becomes the device where the training process is executed (for instance,  0  means using "cuda:0" as device). Default is 0.
> python MAIN.py -gpu 0

### config: 
We communicate with the application via yaml-config files containing all modes, execution configurations, properties and parameters. The exact function and purpose we want to use is defined in the config file. The parameters in the config file are different for every MAIN starting file. For the exact application you like to run, the combination of MAIN.py file, config file (conf.yml) and optionally a list of dictionary encoding file is important. The config file defines the API for our programs.
> python MAIN.py -config config/SCAN.yml 

### prefix or p:
The prefix argument is basically the name and identifier you choose for the execution session and all output files produced by the script. For instance, if your prefix is "cluster01" the output model will be named "cluster01_best_model.pth" and the name of other output files is composed with the prefix you passed in the command line. In some starting files the prefix as
name and ID is that much important, it is defined as -p
> python MAIN -prefix runtime_name

### root_dir:
This is the path for the directory where the output directories and files are created
> python MAIN -root_dir RESULTS

### loss_track:
In MAIN.py, this option determines whether a pandas table is documents the current evaluation statistic for every epoch.
This means the average loss of the epoch (as total loss), the cluster comparison measures with the ground-truth of the evaluation dataset, summaries about confidences from the model prediction and "local consistency" and the entropy over the cluster-sizes distribution.
> python MAIN -loss_track  y 

Now we go to the MAIN file user manual. 
## single clustering process
Use MAIN.py if you want to train just one session or execution with the commands and parameters in the config file. This is the standard use case. We first list the implemented self-supervised deep learning clustering algorithms currently implemented here:

 - SCAN Clustering: https://arxiv.org/pdf/2005.12320.pdf, 
 - TWIST Loss: https://arxiv.org/pdf/2110.07402
 - Double (added SCAN-Loss with TWIST-Loss) 
 - TWIST-multihead (train multiple heads with TWIST Loss)
 - Double-multihead (train multiple heads with both losses)

These Training methods can only be started with MAIN.py or evaluate_MAIN.py
You can also run "Contrastive Clustering" Training with the  contrastive_clustering_MAIN.py file
 Let's take a closer look at the config file for controlling MAIN.py: 
 config/SCAN.yml
 

   

    # use setup parameter for definition of the clustering method
    setup: scan
    
    ### Technical Parameters, necessary but unimportant, stay FIXED 
    num_workers: 8
    
    ### Universal Parameters
    epochs: 200 # the code trains for every epoch, the whole datset is iterated and optimizes the model
    batch_size: 300 # The dataset is devided in pieces of batch_size and iterated by subsets of this size
    
    # if True, no gradients of the backbone that yields the feature representation 
    # for the cluster-head 	module are computed and the parameters stay fixed
    update_cluster_head_only: False 
    
    
    num_classes: 10 # classes used in the selected dataset
    # default available datasets: { stl-10, cifar-10, cifar-20 }
    train_db_name: stl-10 # the identifier for the dataset used for training the cluster-network
    val_db_name: stl-10 # the identifier for the dataset used for evaluating the trained cluster-network predictions
    
    ## nearest neighbors scan loss related
    # for model training with SCAN you need to compute the indices of the K nearest neighbors in the
    # embedding space and save it to a file named according to the 'neighbor_prefix' parameter,
    # the file you compute must match with the dataset you train the model with
    neighbor_prefix: scatnet_test
    # for validation phase, currently deactivated in the code
    neighbor_prefix_val: scatnet_test
    
    # parameter determines the number of nearest neighbor SCAN used in the training process
    # the neighbor indices file is used to access the neighbor index is adapted to this parameter
    num_neighbors: 20 
    
    # We adopted the SimCLR Implementation from the SCAN gitHub repository for an additional option
    to_augmented_dataset: False # not for used for clustering
    to_neighbors_dataset: True # required for Clustering with SCAN
    
    ## for stl10 dataset
    # split of stl-10 to use for training
    train_split: test # available: {train,test,both(train+test),unlabeled,train+unlabeled}
    # split of stl10 for validation
    val_split: test 
    
    # <dependent> use this parameters as combination
    # the augmentation_type sets the output-type and form of the Augmentation Function
    # Different methods prefer different output types. Otherwise, the augmentation_type defines
    augmentation_type: scan  # select {scan,twist}
    augmentation_strategy: ours # scan:{standard,simclr,ours}, twist:{simclr,standard,barlow,twist,scan,random,multicrop}
    
    augmentation_kwargs:
       local_crops_number: 0 # defaults to zero, only used for multicrop
       crop_size: 96 # set for stl-10
       normalize:
          mean: [0.485, 0.456, 0.406] # for stl-10
          std: [0.229, 0.224, 0.225] # for stl-10
       num_strong_augs: 4 # parameter for RandAugment
       cutout_kwargs: # for augmentation_type scan & augmentation strategy ours
         n_holes: 1
         length: 32
         random: True
    # <dependent>
    
    # !!! FIXED !!! for stl10 validation 
    transformation_kwargs: 
       crop_size: 96
       normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
          
    backbone: scatnet # id for network architecture
    pretrain_path: /home/blachm86/backbone_models/scatnet.pth # pretrained model file to load from the backbone directory
    feature_dim: 128 # network parameter
    hidden_dim: 128
    scatnet_args: # parameters for the "scatnet" backbone and pretrained model
        J: 2
        L: 16
        input_size: [96, 96, 3] 
        res_blocks: 30
        out_dim: 128
        
    # <dependent>
    num_heads: 10  # Some methods use multiple heads
    model_type: clusterHeads # head-architecture to use, if num_heads > 1, use clusterHeads
    # <dependent>
    
    model_args: # This parameter options is related to the HEAD-Architecture of the model
	    head_type: mlp # use {mlp,linear}
	    aug_type: default # use "multicrop" for the corresponding augmentation strategy
	    batch_norm: False # use batchnorm for each layer of the head
	    last_batchnorm: False # use a batch normalization for the final output, used for twist algorithm
	    last_activation: softmax # be carefull, SCAN already applies softmax function on the network output
	    drop_out: -1
	    
	# the train method is decisive for the Algorithm you want to use, all parameters in the config must be adapted to this options, especially the loss_type
    train_method: scan
    
    # each train_method has its own loss_type
    loss_type: scan_loss
    entropy_weight: 5.0 # parameter for the scan_loss, not used for another method 
    
	# use this as training parameters
    optimizer: adam # use one of {sgd,adam}
    optimizer_kwargs: # parameters for the optimizers
       lr: 0.0001
       weight_decay: 0.0001
    
    
    
    ## Scheduler
    scheduler: constant # for each epoch, the scheduler adjusts the learning rate
    scheduler_kwargs:
       lr_decay_rate: 0.1

We HIGHLY recommend to look in the file "functionality.py" because there you can see how the parameters are mapped to the program components and the object and code generated for the variables in the config file.
Next we list the files created from the training process.

 - the model parameters at the lowest loss:  [prefix]_best_model.pth
 - logger object that documents each training session of a runtime [prefix].session
 - An "Analysator Object" containing all computed outputs and statistics for a dataset [prefix]_ANALYSATOR
 - a png-image drawing a confusion matrix (optionally for SCAN-method) [prefix]_confusion_matrix.png
 - the log text file listing all training, execution and runtime parameters for checking  the exact condition that produced the outcome [prefix]_log.txt