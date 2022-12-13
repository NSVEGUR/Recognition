# Recognition_ML

Folder structure:

- Data folder
  - For face_recognition you can use your own set of training and testing images of certain people.
  - For traffic_recognition you can get the data from [GTRSB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) dataset.

```bash
Project
│   README.md
│   LICENSE
│   .gitignore
│   face_recognition.ipynb
│   traffic_sign_recognition.ipynb
│
└───Cascades
│     haarcascade_frontalface_alt.xml
│
│
└───Data
    │
    └───Face_data
    │     │
    │     └───Train
    │     │       └─── Person-1
    │     │       │       1.jpeg
    │     │       │       2.jpeg
    │     │       │        ...
    │     │       └─── Person-2
    │     │       │
    │     │       └─── ...
    │     │
    │     └───Test
    │             └─── Person-1
    │             │       1.jpeg
    │             │       2.jpeg
    │             │        ...
    │             └─── Person-2
    │             │
    │             └─── ...
    │
    │
    │
    └───Traffic_sign_data
          │
          └───Train
          │       └─── Object-1
          │       │       1.png
          │       │       2.png
          │       │        ...
          │       └─── Object-2
          │       │
          │       └─── ...
          │
          └───Test
          │     1.png
          │     2.png
          │      ...
          └───Meta
                1.png
                2.png
                ...
```
