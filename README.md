# ESRGAN
ESRGAN, or Enhanced Super-Resolution Generative Adversarial Networks, is a state-of-the-art deep learning model designed for image super-resolution, aiming to generate high-quality, realistic images with enhanced detail and clarity from low-resolution inputs. It utilizes a combination of convolutional neural networks and adversarial training techniques to learn complex mappings between low and high-resolution image spaces, achieving significant improvements over traditional interpolation-based methods. ESRGAN has demonstrated remarkable capabilities in upscaling images across various domains, including photography, medical imaging, and satellite imagery, making it a valuable tool for enhancing visual content.

# Architecture
ESRGAN architecture consists of a generator network that learns to upscale low-resolution images using convolutional layers and residual blocks, coupled with a discriminator network trained to distinguish between generated high-resolution images and real high-resolution ones. This adversarial training setup encourages the generator to produce high-quality outputs that are indistinguishable from authentic high-resolution images, resulting in superior image enhancement.

# Few testing
## Input Image
![Org1](https://github.com/JoyBiswasgithub/ESRGAN/assets/138972138/cedb0665-3077-45d3-9e44-11c4359037c5)
shape = (225,225,3)
## Output Image
![Output1](https://github.com/JoyBiswasgithub/ESRGAN/assets/138972138/e80eeaab-fb40-4db4-99da-5701b7d853c9)
shape = (900,900,3)
