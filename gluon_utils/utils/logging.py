import logging
import csv
import io

class CsvFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.output = io.StringIO()
        self.writer = csv.writer(self.output)

    def format(self, record):
        self.writer.writerow(record.msg)
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()

class HistoryLogger():
    def __init__(self, filename, header):
        self.warning_log = logging.getLogger(__name__)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter('[HistoryLogger:%(levelname)s]: %(message)s'))
        self.warning_log.addHandler(ch)

        self.logger = logging.getLogger(__name__+"_HistoryLogger")
        self.logger.setLevel(logging.INFO)

        self.fh = logging.FileHandler(filename)
        self.fh.setLevel(logging.DEBUG)
        self.fh.setFormatter(CsvFormatter())
        self.logger.addHandler(self.fh)

        self.header = header
        self.update(*header)

    def update(self, *args):
        if len(args) != len(self.header):
            self.warning_log.warning("The supplied number of measurements is different than the recorded header. Expected {}, recieved {}.".format(len(self.header), len(args)))
        self.logger.info(args)