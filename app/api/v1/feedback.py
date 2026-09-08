from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.auth import AuthUser
from app.core.config import Settings, get_settings
from app.core.deps import get_current_user
from app.schemas.api_models import FeedbackRequest, FeedbackResponse
from app.services.email_service import send_feedback_email

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    body: FeedbackRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> FeedbackResponse:
    send_feedback_email(settings, body)
    return FeedbackResponse()
