from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="Question to ask about the uploaded documents",
    )


class SourceResponse(BaseModel):
    citation_id: str
    source: str
    page: int | None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]
    model: str
    usage: dict


class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_created: int
    message: str


class DocumentResponse(BaseModel):
    filename: str
