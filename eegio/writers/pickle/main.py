import pickle


class Save:
    def save_result(self, resultobj, fpath):
        pickle.dump(
            resultobj, fpath, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False
        )
