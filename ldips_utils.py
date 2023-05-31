def template(name, dim, type, value):
    return {
        name: {
            "dim": dim,
            "type": type,
            "name": name,
            "value": value,
        }
    }


def dimensionless_template(name, value):
    return template(name, [0, 0, 0], "NUM", value)


def speed_template(name, value):
    return template(name, [1, -1, 0], "NUM", value)


def distance_template(name, value):
    return template(name, [1, 0, 0], "NUM", value)


def start_template(value):
    if value is None:
        return dict()
    return template("start", [0, 0, 0], "STATE", value)


def output_template(value):
    if value is None:
        return dict()
    return template("output", [0, 0, 0], "STATE", value)


class LDIPSSample(object):
    def __init__(self, obs, fn, prev_action=None, next_action=None) -> None:
        self.sample = {
            **start_template(prev_action),
            **fn(obs),
            **output_template(next_action),
        }

    def get(self, key):
        return self.sample[key]["value"]

    def set(self, key, value):
        self.sample[key]["value"] = value

    def __hash__(self) -> int:
        return hash(tuple(self.sample[k]["value"] for k in self.sample))
