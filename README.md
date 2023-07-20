# Open Set Recognition Loss Optimization (OSR-Loss-Optimization)

The OSR-Loss-Optimization project is an exploration into enhancing the performance of traditional Open Set Recognition (OSR) models. The enhancement strategy focuses on manipulating model embedding space and fine-tuning of the model architecture.

## Project Description

OSR models often struggle to distinguish between known and unknown classes. Our approach addresses this issue by implementing Center Loss and Triplet Loss, which enable the model to more efficiently cluster data in the embedding space.

In addition to these techniques, we employ transfer learning models like RotNet to further improve the performance of our approach. By implementing a composite loss function that promotes clustered data, we've observed an improvement in model accuracy of up to 10%.

## Methodology

Our methodology involved:

1. Incorporating Center Loss and Triplet Loss techniques into conventional ResNet-based OSR models.
2. Utilizing transfer learning models, particularly RotNet, to further optimize our approach.
3. Experimenting and optimizing a composite loss function that encourages data clustering in the embedding space.

## License

This project is licensed under the terms of the MIT License.

## Contributions

The OSR-Loss-Optimization research is a collaborative effort between a Seoul Science High School (SSHS) research group and a Sungkyunkwan University supervisor.

Taehoon Hwang, SSHS Student
Yeongjun Kim, SSHS Student
Jinsu Park, SSHS Student

Prof. Heo Jaepil, Sungkyunkwan University

We express our appreciation to all participants in this research.