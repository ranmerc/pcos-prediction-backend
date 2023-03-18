# PCOS Prediction API

PCOS Prediction API built with [FastAPI](https://fastapi.tiangolo.com/) and deployed on [render](https://www.render.com/).

Machine learning model built with [Google Colab](https://colab.research.google.com/drive/1ivZ_Pe6acEc6dR4_60bSRhVjUfXGhYNe).

Docs available at `/docs`.

## Running on Local Machine

1. Clone repository

```bash
git clone https://github.com/ranmerc/pcos-prediction-backend.git
```

2. Create Virtual Environment

```bash
# Creates Virtual Environment named venv
python -m venv venv
```

3. Activate venv

```bash
source venv/Scripts/activate
```

4. Install packages

```bash
pip install -r requirements.txt
```

5. Running Uvicorn Server

```bash
uvicorn main:app --reload
```

6. Deactivating Virtual Environment

```bash
deactivate
```

- Generating requirements.txt

```bash
pip freeze > requirements.txt
```

## How it works?

We pickle the trained model and StandardScaler object -

```python
import pickle
pickle.dump(model, open("model.sav", 'wb'))
pickle.dump(standard_scalar, open("sc.sav", 'wb'))
```

Then we load the model on server and make prediction using it -

```python
loaded_model = pickle.load(open("model.sav", 'rb'))
sc = pickle.load(open("sc.sav", 'rb'))

test_data = []
test_data = sc.transform([test_data])
res = loaded_model.predict(test_data)
```

## Reference

- Can not activate a virtualenv in GIT bash mingw32 for Windows on [Stack Overflow](https://stackoverflow.com/a/61290456)

- Is it bad to have my virtualenv directory inside my git repository? on [Stack Overflow](https://stackoverflow.com/a/6590783)

- Python Tutorial: VENV (Windows) - How to Use Virtual Environments with the Built-In venv Module by [Cory Schafer on Youtube](https://www.youtube.com/watch?v=APOPm01BVrk)

- Deploying FastAPI application to Render by [Akash R Chandran](https://blog.akashrchandran.in/deploying-fastapi-application-to-render)

- What and why behind fit_transform() and transform() in scikit-learn! by [Towards Data Science](https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe)
