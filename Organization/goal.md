1. LSGAN, WGAN, GAN, CGAN, variable lossfunktions implementation with strategy pattern
2. Validation with differnet metrics
3. Image Generator 
4. extendable to each gan (WaveGAN, cycleGAN)
5. data manager 

-Trainer:
    strategies: loss functions, GANs, ...





### Next steps:
1. DCGAN dynamically adjustment to the desired size of the images. -> new dataloader for a new dataset
2. Testing on new data set 3 channels
3. CGAN with the same traits as normal gan
4. First evaluation pipeline: metrics, plots, gifs
5. directory managment where to save files and so...
6. Spectral normalization


oder doch erst cgan

falls schritt 2 nun heute klappt, dann fange ich mit einer klasse an die meine Files ordnet und ein dir für jeden Testlauf erstellt, dann dass man weiter Trainieren kann auch 
bei großen Datensätzen den bacthes nach 
dann cgan 

Auch wissenschaftlich alles dokumentieren in einer weiteren Klasse die plots erstellt und auch oben in die dirs reinspeichert (gradienten, losses, eval werte)