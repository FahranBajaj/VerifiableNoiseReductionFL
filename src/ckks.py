import tenseal as ts

def generate_ckks_context() -> ts.Context:
    """Create a new context for CKKS encryption"""
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096)
    context.global_scale = pow(2, 40)
    return context