## Proposal

The proposed solution involves an advanced control system integrating an LED array with
a camera setup. This system leverages depth and normal maps of the subject to estimate
and manage changes in illuminance across the part’s surface. The goal is to achieve uniform
light distribution using model-predictive control techniques, ensuring consistency in
brightness across all images.
![image](https://github.com/user-attachments/assets/bc55c659-2e9f-491e-8659-a3774741458c)
The process of training the model involved the collection of diverse data sets using Blender,
a 3D rendering tool. This included capturing images before and after altering light intensities,
alongside obtaining depth and normal maps from the camera. These maps provided
crucial information about the scene’s geometry and surface properties. Depth maps helped
in understanding the spatial layout, while normal maps offered insights into surface orientation.
This comprehensive data collection was essential for the model to learn the relationship
between changes in LED intensity, depth, normals, and the resulting illuminance
delta maps, thus ensuring the training of an effective and robust model.
![image](https://github.com/user-attachments/assets/10df9ce7-c4b0-405d-9546-766d336d2cee)

## Result

### VAE pre-train Result:
![image](https://github.com/user-attachments/assets/cfe96ceb-3365-43df-9727-6213c06564fd)
### DCGAN train Result with adjustable 384 LED lights:
![image](https://github.com/user-attachments/assets/18e6462f-ad50-45f4-b797-a24b21d6e361)
