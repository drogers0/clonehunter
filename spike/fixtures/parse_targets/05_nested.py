class Processor:
    def process(self, data):
        def _validate(item):
            return item is not None

        def _transform(item):
            return str(item).strip()

        return [_transform(x) for x in data if _validate(x)]

    def pipeline(self, data, steps):
        def apply_step(value, step):
            return step(value)

        result = data
        for step in steps:
            result = apply_step(result, step)
        return result
