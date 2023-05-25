from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import os


class Parameters(BaseModel):
    age: float
    marriageStatus: float
    weight: float
    bmi: float
    follicleNoR: float
    follicleNoL: float
    amh: float
    regularCycle: float
    cycleLength: float
    skinDarkening: float
    hairGrowth: float
    weightGain: float
    fastFood: float
    pimples: float


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
async def get_prediction(patientParameters: Parameters):
    predictionParams = [
        patientParameters.skinDarkening,
        patientParameters.hairGrowth,
        patientParameters.weightGain,
        patientParameters.fastFood,
        patientParameters.pimples,
        patientParameters.follicleNoR,
        patientParameters.follicleNoL,
        patientParameters.regularCycle,
        patientParameters.amh,
        patientParameters.weight,
        patientParameters.bmi,
        patientParameters.cycleLength,
        patientParameters.age,
        patientParameters.marriageStatus,
    ]

    loaded_model = pickle.load(open(os.path.join(os.getcwd(), "model.sav"), "rb"))
    sc = pickle.load(open(os.path.join(os.getcwd(), "sc.sav"), "rb"))

    predictionParams = sc.transform([predictionParams])
    res = loaded_model.predict(predictionParams)

    if res:
        return {"atRisk": True}
    else:
        return {"atRisk": False}
