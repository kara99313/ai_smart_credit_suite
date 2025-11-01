# utils/email_sender.py
from __future__ import annotations
import os, smtplib, mimetypes
from email.message import EmailMessage
from typing import List, Tuple

def _smtp_settings():
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USER")
    pwd  = os.getenv("SMTP_PASS")
    port = int(os.getenv("SMTP_PORT", "587"))
    tls  = os.getenv("SMTP_STARTTLS", "1") != "0"
    return host, user, pwd, port, tls

def send_mail_with_attachment(
    to_addr: str,
    subject: str,
    body: str,
    files: List[Tuple[str, bytes, str]] | None = None,
) -> tuple[bool, str]:
    """
    files: liste de (filename, content_bytes, mime)
    """
    host, user, pwd, port, tls = _smtp_settings()
    if not host or not user or not pwd:
        return False, "SMTP non configuré (variables: SMTP_HOST, SMTP_USER, SMTP_PASS)."

    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    for name, content, mime in (files or []):
        main, sub = (mime.split("/", 1) if "/" in mime else ("application", "octet-stream"))
        msg.add_attachment(content, maintype=main, subtype=sub, filename=name)

    try:
        with smtplib.SMTP(host, port) as s:
            if tls:
                s.starttls()
            s.login(user, pwd)
            s.send_message(msg)
        return True, "OK"
    except Exception as e:
        return False, str(e)
