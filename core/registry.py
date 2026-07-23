

class Registry:
    """Class to register class names from yaml into actual classes. This allows more dynamic behavior. 
    """
    def __init__(self,name):
        self.name = name
        self.register = {}
        
    def registry(self,name):
        def decorator(cls):
            self.register[name] = cls
            return cls
        return decorator

    def get(self,name):
        return self.register[name]
