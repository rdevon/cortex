from cortex.handlers import Handler

class ArchHandler(object):
    def __init__(
            self,
            defaults=None,
            setup=None,
            build=None,
            Dataset=None,
            DataLoader=None,
            transform=None,
            train_routines=None,
            test_routines=None,
            finish_train_routines=None,
            finish_test_routines=None,
            doc=None,
            kwargs=dict(),
            signatures=[],
            info=dict(),
            eval_routine=None):
        self.defaults = defaults
        self.setup = setup
        self.build = build
        self.Dataset = Dataset
        self.DataLoader = DataLoader
        self.transform = transform

        self.train_routines = train_routines
        self.test_routines = test_routines or {}
        self.finish_train_routines = finish_train_routines or {}
        self.finish_test_routines = finish_test_routines or {}
        self.eval_routine = eval_routine

        self.doc = doc
        self.kwargs = kwargs
        self.signatures = signatures
        self.info = info

    def unpack_args(self, args):
        model = Handler()
        routines = Handler()
        for k, v in vars(args).items():
            for sig_k, sig_v in self.signatures.items():
                if k in sig_v:
                    if sig_k == 'model':
                        model[k] = v
                    else:
                        if sig_k not in routines:
                            routines[sig_k] = {k: v}
                        else:
                            routines[sig_k][k] = v

        return Handler(model=model, routines=routines)
