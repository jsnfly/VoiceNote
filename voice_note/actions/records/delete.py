from actions.records.records import Records
from shutil import rmtree

class Delete(Records):
    TRIGGER_WORDS = ["delete", "löschen"]

    def run(self, _):
        last_record = self.get_last_record()
        if last_record is not None:
            rmtree(last_record)
