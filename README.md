# Somchai Fake News Detection on Social Media: with Content based Method

### Prerequisites

What things you need to install the software and how to install them:
   ```
   git clone https://github.com/neroswords/Fake_News_Detection.git
   ```

after you clone this repository, you should create your own enviroment first before do as this command. Then download glove.6B.50d.txt, you can download it from https://www.kaggle.com/watts2/glove6b50dtxt

this project use Python 3.7.1 for compile, then you can follow this command line.
   ```
   virtualenv venv (optional)
   venv\scripts\activate
   pip install -r requirements.txt
   ```
### Installation and Usage
after you do Prerequisites step,then follow this code.

for create new model you run this command.
   ```
   python classifier.py
   ```
for use application with UI, run this command.
   ```
   python prediction.py
   ```

Below is the Process Flow of the project:

<!-- <p align="center">
  <img width="600" height="750" src="https://github.com/nishitpatel01/Fake_News_Detection/blob/master/images/ProcessFlow.PNG">
</p> -->

### Performance
Below is the learning curves for our candidate models. 

**Logistic Regression Classifier**

<!-- <p align="center">
  <img width="550" height="450" src="https://github.com/nishitpatel01/Fake_News_Detection/blob/master/images/LR_LCurve.PNG">
</p>

**Random Forest Classifier**

<p align="center">
  <img width="550" height="450" src="https://github.com/nishitpatel01/Fake_News_Detection/blob/master/images/RF_LCurve.png">
</p> -->
