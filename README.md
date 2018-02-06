### cosmoGAN

This code is to accompany "Creating Virtual Universes Using Generative Adversarial Networks" manuscript [arXiv:1706.02390](https://arxiv.org/abs/1706.02390).
The architecture is an implementation of the DCGAN architecture ([arXiv:1511.06434](https://arxiv.org/abs/1511.06434)).

- - - 
### How to train:  
```bash
git clone git@github.com:MustafaMustafa/cosmoGAN.git
cd cosmoGAN/networks
wget http://portal.nersc.gov/project/dasrepo/cosmogan/cosmogan_maps_256_8k_1.npy
```

That will download sample data (8k maps) for testing. You can download more data from [here](http://portal.nersc.gov/project/dasrepo/cosmogan/).  

To run:
```bash
python run_dcgan.py
```


### Load pre-trained weights:  
First download the weights:
```bash
cd cosmoGAN/networks
wget http://portal.nersc.gov/project/dasrepo/cosmogan/cosmoGAN_pretrained_weights.tar
tar -xvf cosmoGAN_pretrained_weights.tar
```

Then take a look at `load_and_use_pretrained_weights` notebook for how to run.  
