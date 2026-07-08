from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_class: str = Field(
        ...,
        example="scratches"
    )

    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        example=0.9985
    )
    