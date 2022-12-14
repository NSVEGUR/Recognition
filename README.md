# Recognition

---

Project based on recognition given in college. Here we discuss about two types of learnings facial recognition using machine learning and traffic sign recognition using both machine learning and deep learning.

Used:

- `PCA` for dimensionality reduction
- `KNN`, `LDA` and `LogisticRegression` as machine learning models
- `OpenCV` and `PIL` for image related works
- `CNN` using for deep learning on traffic signs

Folder structure:

- For face recognition,
  - you can use your own set of training and testing images of certain people.
- For traffic sign recognition,
  - you can get the data from [GTRSB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) dataset.

```bash
Project
│   README.md
│   LICENSE
│   .gitignore
│
└───Face_Recognition
│     │
│     └───Cascades
│     │     haarcascade_frontalface_alt.xml
│     │
│     └───Face_data
│           │
│           └───Train
│           │       └─── Person-1
│           │       │       1.jpeg
│           │       │       2.jpeg
│           │       │        ...
│           │       └─── Person-2
│           │       │
│           │       └─── ...
│           │
│           └───Test
│                   └─── Person-1
│                   │       1.jpeg
│                   │       2.jpeg
│                   │        ...
│                   └─── Person-2
│                   │
│                   └─── ...
│
└───Traffic_Sign_Recognition
      │        using_CNN.ipynb
      │        using_KNN.ipynb
      │        model.h5
      │
      └───Data
          │  Meta.csv
          │  Test.csv
          │  Train.csv
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
