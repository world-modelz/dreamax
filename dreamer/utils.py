import haiku as hk

def initializer(name: str) -> hk.initializers.Initializer:
    return {
        'glorot': hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        'he': hk.initializers.VarianceScaling(2.0, 'fan_in', 'uniform')
    }[name]