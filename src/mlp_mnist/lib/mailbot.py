"""
    Tiny class to send emails.
"""

import json
import smtplib

class Mailbot:
    login_data = None
    server = None

    def __init__(self, fname):
        self._load_login(fname)

    def _load_login(self,fname):
        with open(fname, "r") as fp:
            self.login_data = json.load(fp)

    def _open_connection(self):
        try:
            self.server = smtplib.SMTP_SSL(host=self.login_data["server_address"], port=self.login_data["port"])
            self.server.ehlo()
        except:
            raise ValueError("MAILBOT: Invalid server_address or port specified!")

    def _login(self):
        try:
            self.server.login(self.login_data["account"], self.login_data["password"])
        except:
            raise ValueError("MAILBOT: Invalid account and/or password specified!")

    def _close_connection(self):
        self.server.close()

    def send(self, message, subject=""):
        self._open_connection()
        self._login()
        to = [recipient["address"] for recipient in self.login_data["recipients"]]
        email_text = "From: {:s}\nTo: {:s}\nSubject: {:s}\n\n{:s}".format(self.login_data["account"], ", ".join(to), subject, message)
        self.server.sendmail(self.login_data["account"], to, email_text)
        self._close_connection()

"""
mailbot = Mailbot("mailbot.json")
mailbot.send("Loss = 561231\nAccuray = 12.1", "[BlueLagDebug] Training finished on unknown model")
"""