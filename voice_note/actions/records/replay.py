import wave
from actions.records.records import Records

class Replay(Records):
    TRIGGER_WORDS = ["replay", "wiedergeben"]

    def run(self, _):
        last_record = self.get_last_record()
        if last_record is not None:
            with wave.open(str(last_record / 'sample.wav'), 'r') as wf:
                frames = wf.readframes(wf.getnframes())
            return {'audio': {'frames': frames, 'width': wf.getsampwidth(),'channels': wf.getnchannels(),
                              'rate': wf.getframerate()}}
