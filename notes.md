
---------------

<h3>Prediction process</h3>

1. Predict bbox of video with yolov5 and establish the path of each sperm.
2. Calculate the sperm centroid and the characteristics associated with each sperm.
3. Classification of sperm as progressive or non-progressive.
4. Preprocessing of the resulting information and training of the model.
5. Validate the model using the metrics obtained.

---------------

<h3>Dataset for model progressive or non progressive</h3>
Features:

- Velocity: Average speed of the sperm (pixels/second).
- Straightness Ratio: Ratio of straight-line displacement to total path length.
- Angular Displacement: Change in direction over time (useful for detecting circular motion).
- Linearity: Measures how straight the trajectory is (e.g., using linear regression).
- Curvature: Measures how curved the trajectory is.
- Amplitude of Lateral Head Displacement (ALH): Side-to-side movement of the sperm head.
  - ALH measures the maximum lateral distance that the sperm head deviates from its average trajectory during its movement. In other words, it represents the amplitude of the oscillatory movements of the head as the sperm advances. It is expressed in micrometers (µm). Importance:
    - A high ALH indicates larger, more vigorous head movements, which may be associated with more active sperm.
    - A low ALH suggests more restricted movement, which may be related to a lower ability to progress and fertilize.
- Beat Cross Frequency (BCF): Frequency of the sperm head crossing its average path.
  - Beat Cross Frequency (BCF) is a metric used in the analysis of sperm motility. It is defined as the frequency with which the curvilinear trajectory of a sperm crosses its average trajectory. It is measured in hertz (Hz), which indicates the number of crossings per second. In other words, BCF reflects the speed and regularity of the oscillatory movements of the sperm tail and is related to its ability to move efficiently in a liquid medium. It is an important parameter in fertility studies, since it can influence the ability of the sperm to reach and fertilize the egg.
- Total Distance Traveled: Total path length of the sperm.
- Displacement: Straight-line distance from the start to the end point.
- Time Elapsed: Total time of the trajectory.