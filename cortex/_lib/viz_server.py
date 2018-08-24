class VizServerSingleton(object):
    """
    Singleton to manage visualisation server
    opening and closing.
    """
    class __OnlyOne:
        def __init__(self):
            self.viz_process = None
    instance = None

    def __new__(cls):
        if not VizServerSingleton.instance:
            VizServerSingleton.instance = VizServerSingleton.__OnlyOne()
        return VizServerSingleton.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name):
        return setattr(self.instance, name)
