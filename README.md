# PULSE - AI bias detection and mitigation in sentiment analysis
This project was created in 48 hours by the Entourage Penguins team ([Camille Coeurjoly](https://github.com/Camille1992), [Leo Stahlschmidt](https://www.linkedin.com/in/leo-s-34680012b/), and [Sabina Firtala](https://github.com/sabinagio)) at the 6th edition of the Hackathon for Good.


## Problem Statement

> Currently, AI developers face considerable challenges in identifying and rectifying biases in AI datasets. Traditional methods for bias detection and correction are often time-consuming and complex, making them impractical for rapid development cycles. This delay in addressing biases can lead to their perpetuation in AI outputs, reinforcing existing prejudices and inequalities. The challenge is to streamline this process, enabling swift and efficient bias mitigation.

Read more about the challenge [here](https://www.hackathonforgood.org/hackathons/the-hague-6/rapid-bias-identification-and-correction-in-ai).

## Solution

Our team created PULSE, a Streamlit application which detects and mitigates racist AI modelling results for social media content moderation. PULSE detects bias in model outputs statistically and by calculating the [bias AUC scores as devised by Google Jigsaw](https://medium.com/sentropy/our-approach-to-machine-learning-bias-part-2-4f94b3f58ff9) and it mitigates it using both weight correction and adversarial debiasing. To detect whether a text includes a racial subgroup, we used [Holistic Bias descriptors](https://github.com/facebookresearch/ResponsibleNLP/tree/main/holistic_bias) developed by Meta. We present the [CrowdFlower dataset](https://github.com/t-davidson/hate-speech-and-offensive-language) as our use-case and welcome any suggestions for improvement.

*Note: The PULSE version currently in GitHub may still be missing some functionality. Please contact the repository owner (Sabina Firtala) regarding issues.*

## Data

@inproceedings{hateoffensive,
  title = {Automated Hate Speech Detection and the Problem of Offensive Language},
  author = {Davidson, Thomas and Warmsley, Dana and Macy, Michael and Weber, Ingmar}, 
  booktitle = {Proceedings of the 11th International AAAI Conference on Web and Social Media},
  series = {ICWSM '17},
  year = {2017},
  location = {Montreal, Canada},
  pages = {512-515}
}
