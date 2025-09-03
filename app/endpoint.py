from pydantic import BaseModel

class BankMarketingFeatures(BaseModel):
    # 10 признаков для предсказания
    poutcome: str
    contact: str
    duration: int
    housing: str
    month: str
    previous: int
    pdays: int
    loan: str
    age: int
    day: int