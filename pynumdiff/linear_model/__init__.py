from ._linear_model import lineardiff, savgoldiff, polydiff, spectraldiff

__all__ = ['lineardiff'] # Things in this list get treated as direct members of the module by sphinx
# polydiff and savgoldiff are still imported for now for backwards compatibility but are not documented
# as part of this module, since they've moved
