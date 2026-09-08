import smtplib
from email.message import EmailMessage

from fastapi import HTTPException, status

from app.core.config import Settings
from app.schemas.api_models import FeedbackRequest

CATEGORY_LABELS = {"bug": "Bug", "feature": "Feature request", "other": "Other"}


def _build_body(feedback: FeedbackRequest) -> str:
    ctx = feedback.app_context
    lines = [feedback.message.strip(), ""]
    if ctx:
        lines.append("--- App context ---")
        if ctx.version:
            lines.append(f"Version: {ctx.version}")
        if ctx.platform:
            lines.append(f"Platform: {ctx.platform}")
        lines.append(f"Desktop app: {'yes' if ctx.is_desktop else 'no'}")
        if ctx.url:
            lines.append(f"URL: {ctx.url}")
        lines.append("")
    if feedback.console_errors:
        lines.append("--- Recent console errors ---")
        lines.extend(feedback.console_errors)
    return "\n".join(lines)


def send_feedback_email(settings: Settings, feedback: FeedbackRequest) -> None:
    if not settings.smtp_user or not settings.smtp_password or not settings.feedback_to_email:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback email is not configured (SMTP_USER/SMTP_PASSWORD/FEEDBACK_TO_EMAIL)",
        )

    label = CATEGORY_LABELS.get(feedback.category, "Feedback")
    subject_excerpt = feedback.message.strip().replace("\n", " ")[:60]
    msg = EmailMessage()
    msg["Subject"] = f"[POLARIS Feedback] {label} — {subject_excerpt}"
    msg["From"] = settings.smtp_user
    msg["To"] = settings.feedback_to_email
    msg.set_content(_build_body(feedback))

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as smtp:
            smtp.starttls()
            smtp.login(settings.smtp_user, settings.smtp_password)
            smtp.send_message(msg)
    except smtplib.SMTPException as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to send feedback email: {exc}",
        ) from exc
