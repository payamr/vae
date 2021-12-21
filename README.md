## Convolutional VAE
This is adapted from tensorflow's VAE example: https://www.tensorflow.org/tutorials/generative/cvae

## Test on MNIST
Running `run_inference` generates the following images.
### Trained latent representations for 2^16 train datasets
![latent_scatter](https://user-images.githubusercontent.com/26334704/146868632-e2c0b42a-f54e-4f32-b207-13ef756cd26e.png)

### Generated images by varying mean and std between the train set values
![transition](https://user-images.githubusercontent.com/26334704/146868683-b44c9f0e-de29-4cc3-b6d3-f8dbb2f0a018.png)
