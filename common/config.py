class Config(dict):
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(e)

    def build_dict_from_keys(self, keys):
        return {key: self[key] for key in keys}
