import email
import os
from email.encoders import encode_base64
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# from smtplib import SMTP_SSL as SMTP 	# This invokes the secure SMTP protocol (port 465, uses SSL)
from mimetypes import guess_type
from pathlib import Path
from smtplib import SMTP, SMTP_SSL  # Use this for standard SMTP protocol   (port 25, no encryption)
from typing import List
import os


class Email(object):
    # Standard (non-secure) SMTP port - 25. SSL encrypted port - 465.
    SMTP_CONFIG = dict(
        server=os.environ.get("SMTP_SERVER"), 
        port=465,
        username=os.environ.get("SMTP_USERNAME"), 
        password=os.environ.get("SMTP_PASSWORD")
    )

    # typical values for text_subtype are plain, html, xml
    DEFAULT_CONTENT_TYPE = "plain"
    DEFAULT_SENDER = os.environ.get("SMTP_DEFAULT_SENDER")

    def __init__(self, SMTP_CONFIG=None, debug=False):
        if not SMTP_CONFIG:
            SMTP_CONFIG = self.SMTP_CONFIG

        self.connection = SMTP_SSL(host=SMTP_CONFIG["server"], port=SMTP_CONFIG["port"])
        self.connection.set_debuglevel(debug)
        self.connection.login(SMTP_CONFIG["username"], SMTP_CONFIG["password"])

    def send(self, subject, message, receivers, attachments: List[str], sender=None, content_type=None):
        if not content_type:
            content_type = self.DEFAULT_CONTENT_TYPE

        if not sender:
            sender = self.DEFAULT_SENDER

        # create html email
        html = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" '
        html += '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"><html xmlns="http://www.w3.org/1999/xhtml">'
        html += '<body style="font-size:12px;font-family:Verdana"><p>С уважением, Команда корневых причин.</p>'
        html += "</body></html>"
        msg = MIMEMultipart()
        # TODO: MIMEMultipart might become ambiguous. Fix it before proceeding further
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = ', '.join(receivers)
        # emailMsg['Cc'] = ", ".join(cc) Кого в копию поставить
        msg.attach(MIMEText(html, 'html'))

        for filepath in attachments:
            mimetype, encoding = guess_type(str(filepath))
            mimetype = mimetype.split('/', 1)
            attach = MIMEBase(mimetype[0], mimetype[1])
            with open(str(Path(filepath)), "rb") as fp:
                attach.set_payload(fp.read())
                encode_base64(attach)
                attach.add_header('Content-Disposition', 'attachment', filename=os.path.basename(filepath))
            msg.attach(attach)

        self.connection.sendmail(sender, receivers, msg.as_string())


__all__ = ["Email"]
