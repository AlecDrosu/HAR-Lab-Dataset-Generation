## Papers to Read to understand Validation
### Shruthi

1. [A tutorial on human activity recognition using body-worn inertial sensors](https://dl.acm.org/doi/10.1145/2499621)
2. [IMUTube: Automatic Extraction of Virtual on-body Accelerometry from Video for Human Activity Recognition](https://arxiv.org/abs/2006.05675)
3. [Human activity recognition in smart homes : tackling data variability using context-dependent deep learning, transfer learning and data synthesis](https://theses.hal.science/tel-03728064/)

### Other Papers
4. [A Survey of Human Activity Recognition in Smart Homes Based on IoT Sensors Algorithms: Taxonomies, Challenges, and Opportunities with Deep Learning](https://arxiv.org/abs/2111.04418)
5. [Joint Modeling of Event Sequence and Time Series with Attentional Twin Recurrent Neural Networks](https://arxiv.org/abs/1703.08524)

## Questions to know the answer to:
- What is the variability?
  1. Repetitive patterns: Since the generated data is based on a small sample of input data, the model may end up generating similar data points repeatedly. This could lead to overfitting on the input data, reducing the model's ability to generalize to new, unseen data.

  2. Limited diversity in the virtual dataset: If the input data is not diverse enough, the generated data may not capture the full range of possible scenarios and variations that could occur in a real-world setting. This limitation could impact the performance of the model when it encounters real data with different characteristics.

  3. Constraint functions and dataset specificity: If the constraint functions used in generating the virtual data are tailored specifically to the input dataset, they may not be easily applicable or adaptable to other datasets. This could limit the model's ability to generalize to different settings or scenarios.

  4. variability is important to consider because it helps assess the model's ability to capture the diverse range of scenarios and variations that can occur in real-world settings, and to generalize to new, unseen data.
- How can we quantify the data? (Validation I guess)
- Diverse Data
  1. To mitigate these issues, consider incorporating a diverse set of real-world data as input, incorporating techniques such as data augmentation to increase the diversity of the generated data, and using more general constraint functions that can adapt to different datasets and settings.
- modality
  
  1. Accelerometer data: Captures motion-based information from wearable devices, such as smartphones or smartwatches.
   
  2. Gyroscope data: Measures angular velocity, which can be useful for detecting rotational movements.
   
  3. Environmental sensors: Can provide information about temperature, humidity, or air quality, which may be relevant to some activities.
   
  4. Audio data: Can capture sounds produced by humans or their environment during activities.
   
  5. Video data: Provides visual information about the environment and the person's movements and interactions.

- Quantitative Metrics
  1. numerical metrics to evaluate the performance of the model
  2. Examples include: F-score, precision, recall, accuracy, and mean squared error.
- Qualitative Metrics
  1. Harder to define
  2. examine the strengths of the model 
  3. Check if the model is measuring all the activities
  4. Temporal patterns (Making food at 3am is odd)


train using original test on fake
train using fake test on original

combine datasets K-fold validation 80 - 20 split
granularity
durations
activity length
activity frequency
pdf instead of cdf
## Things to do:
- Evaluate on other datasets
- Try to fix the null value problem (remove them or get rid of most of them with the timestamps the same)

## For the presentation:

- Well defined motivation

" Developing a machine learning model for human activity recognition in smart homes generates synthetic sensor data to address the challenges of acquiring extensive real-world datasets. Leveraging a limited initial dataset, the model creates synthetic data to simulate longer periods, thus enabling advanced models to benefit from larger datasets and enhance performance. "

- map of progress
- what was achieved

# Other important links and so on

