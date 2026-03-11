from abc import ABC,abstractmethod

#abstract class for the observer that is responsible for the logging, saving,....
class Observer(ABC):
    @abstractmethod
    def update(self,info):
        pass
    