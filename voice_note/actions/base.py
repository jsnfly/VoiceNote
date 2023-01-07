from abc import ABC, abstractmethod

class Action(ABC):
    def __call__(self, decoding_result, response):
        if self.trigger_condition(decoding_result):
            result = self.run(decoding_result)
            if result is not None:
                response.update(result)
        return response

    @abstractmethod
    def trigger_condition(self, decoding_result):
        pass

    @abstractmethod
    def run(self, decoding_result):
        pass
